// sp_ruleload.h — M9a 规则加载器 (header-only, sp_watch.h 惯例).
// 纪律 (spec §4.2): 按目录扫描逐个 dlopen; abi_ver 不匹配/缺符号/坏 ELF
// 一律 REJECT 并记日志, 宿主不倒; THR 外置 thr.conf, 规则启动拿自己的段.
// 加载期全防御; eval 期崩溃 = 进程死由 systemd 拉起 (spec §6 允许).
#ifndef SP_RULELOAD_H_
#define SP_RULELOAD_H_
#include <dirent.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "sp_rule.h"

namespace sprule {

inline std::string thr_section(const std::string& text, const char* name) {
  // 取 [name] 段原文 (到下一个 '[' 或 EOF), 不含段头行; 无段返回空
  std::string want = "[" + std::string(name) + "]";
  size_t pos = 0;
  while (true) {
    size_t b = text.find(want, pos);
    if (b == std::string::npos) return "";
    if (b != 0 && text[b - 1] != '\n') {  // 段名须整行匹配
      pos = b + 1;
      continue;
    }
    size_t s = b + want.size();
    if (s < text.size() && text[s] == '\n') s += 1;
    size_t e = text.find("\n[", s);
    return e == std::string::npos ? text.substr(s) : text.substr(s, e - s + 1);
  }
}

struct RuleMgr {
  std::vector<sp_rule_desc*> rules;
  std::vector<void*> handles;

  int scan(const char* dir, const char* thr_path, sp_rule_desc* out[],
           int cap, char* err, size_t errlen) {
    rules.clear();
    handles.clear();
    if (err && errlen) err[0] = 0;
    std::string thr;
    if (thr_path) {
      FILE* f = fopen(thr_path, "rb");
      if (f) {
        char buf[8192];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) thr.append(buf, n);
        fclose(f);
      }  // 无 thr 文件 = 全部走规则内建默认, 不是错
    }
    DIR* d = opendir(dir);
    if (!d) {
      if (err && errlen)
        snprintf(err, errlen, "opendir %s: errno=%d", dir, errno);
      return 0;
    }
    std::vector<std::string> sos;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
      std::string fn = e->d_name;
      if (fn.size() > 3 && fn.substr(fn.size() - 3) == ".so")
        sos.push_back(fn);
    }
    closedir(d);
    for (size_t i = 0; i < sos.size() && (int)rules.size() < cap; ++i) {
      std::string path = std::string(dir) + "/" + sos[i];
      void* h = dlopen(path.c_str(), RTLD_NOW);
      if (!h) {
        const char* dlerr = dlerror();  // dlerror 取一次即清, 先存
        fprintf(stderr, "ruleload: REJECT %s (%s)\n", path.c_str(),
                dlerr ? dlerr : "dlopen failed");
        continue;
      }
      auto q = (const sp_rule_desc* (*)(void))dlsym(h, "sp_rule_query");
      const char* why = 0;
      if (!q) why = "sp_rule_query missing";
      const sp_rule_desc* desc = q ? q() : 0;
      if (!desc) why = why ? why : "query returned null";
      if (!why && desc->abi_ver != SP_RULE_ABI) why = "abi mismatch";
      if (!why && (!desc->name || !desc->name[0])) why = "empty name";
      if (!why && desc->init &&
          desc->init(thr_section(thr, desc->name).c_str()) != 0)
        why = "init failed";
      if (why) {
        fprintf(stderr, "ruleload: REJECT %s (%s)\n", path.c_str(), why);
        dlclose(h);
        continue;
      }
      fprintf(stderr, "ruleload: loaded %s v%s (%s)\n", desc->name,
              desc->version ? desc->version : "?", sos[i].c_str());
      rules.push_back(const_cast<sp_rule_desc*>(desc));
      handles.push_back(h);
    }
    if (out)
      for (int i = 0; i < (int)rules.size() && i < cap; ++i)
        out[i] = rules[i];
    return (int)rules.size();
  }

  int rescan(const char* dir, const char* thr_path) {
    unload_all();
    sp_rule_desc* tmp[SP_RULE_HIST * 8];
    return scan(dir, thr_path, tmp, SP_RULE_HIST * 8, 0, 0);
  }

  void unload_all() {
    for (auto* r : rules)
      if (r->fini) r->fini();
    for (auto* h : handles) dlclose(h);
    rules.clear();
    handles.clear();
  }

  sp_rule_desc* at(int i) {
    return (i >= 0 && i < (int)rules.size()) ? rules[i] : nullptr;
  }
};

inline RuleMgr& mgr() {
  static RuleMgr m;
  return m;
}

}  // namespace sprule

// plan 固定签名的自由函数 (单例支撑; SIGHUP 重扫 = rescan)
inline int rule_scan(const char* dir, const char* thr_path,
                     sp_rule_desc* out[], int cap, char* err,
                     size_t errlen) {
  return sprule::mgr().scan(dir, thr_path, out, cap, err, errlen);
}
inline int rule_rescan(const char* dir, const char* thr_path) {
  return sprule::mgr().rescan(dir, thr_path);
}
inline void rule_unload_all() { sprule::mgr().unload_all(); }
inline sp_rule_desc* rule_at(int i) { return sprule::mgr().at(i); }

#endif  // SP_RULELOAD_H_
