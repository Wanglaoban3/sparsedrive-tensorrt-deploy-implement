// run_engine2 —— 双引擎链式 runner (backbone -> head, D2D 零拷贝)
// 用法: run_engine2 bb.engine hd.engine [plugin.so] in_dir1 [in_dir2]
//               --dump out_dir [--iters N] [--warmup W]
// bb 输出张量按名字直接作为 hd 输入设备地址 (Orin 统一内存, 零拷贝);
// 其余 hd 输入从 manifest.tsv (name\tdtype\tdims\tfile) 读文件上传。
// dump: hd 全部输出 -> out_dir/<name>.bin + manifest.tsv (与 run_engine 一致)
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << "\n";
    }
};

static std::map<std::string, std::vector<char>> load_manifest(
        const std::string& dir) {
    std::map<std::string, std::vector<char>> blobs;
    FILE* f = fopen((dir + "/manifest.tsv").c_str(), "r");
    if (!f) { std::cerr << "no manifest in " << dir << "\n"; return blobs; }
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        char* p = line;
        auto field = [&p]() {
            char* st = p;
            while (*p && *p != '\t' && *p != '\n') ++p;
            std::string out(st, p - st);
            if (*p) ++p;
            while (!out.empty() && (out.back() == '\r' || out.back() == ' '))
                out.pop_back();
            return out;
        };
        std::string name = field();
        field();                    // dtype (体积以文件为准)
        field();                    // dims
        std::string file = field();
        if (name.empty()) continue;
        FILE* b = fopen((dir + "/" + file).c_str(), "rb");
        if (!b) { std::cerr << "missing blob " << dir << "/" << file << "\n";
                  blobs.clear(); return blobs; }
        fseek(b, 0, SEEK_END);
        long sz = ftell(b);
        fseek(b, 0, SEEK_SET);
        std::vector<char> buf(sz);
        if (fread(buf.data(), 1, sz, b) != (size_t)sz) {
            fclose(b); blobs.clear(); return blobs;
        }
        fclose(b);
        blobs[name] = std::move(buf);
    }
    fclose(f);
    return blobs;
}

static const char* dtype_str(DataType t) {
    switch (t) {
        case DataType::kFLOAT: return "f32";
        case DataType::kHALF: return "f16";
        case DataType::kINT32: return "i32";
        case DataType::kINT8: return "i8";
        case DataType::kBOOL: return "b8";
        default: return "unk";
    }
}

static size_t dtype_size(DataType t) {
    switch (t) {
        case DataType::kFLOAT: return 4;
        case DataType::kHALF: return 2;
        case DataType::kINT32: return 4;
        case DataType::kINT8: return 1;
        case DataType::kBOOL: return 1;
        default: return 0;
    }
}

static long long volume(const Dims& d) {
    long long v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
    return v;
}

struct TensorBuf {
    void* dev = nullptr;
    size_t bytes = 0;
};

int main(int argc, char** argv) {
    const char* bbPath = nullptr;
    const char* hdPath = nullptr;
    const char* pluginSo = nullptr;
    const char* dumpDir = nullptr;
    const char* bbDumpDir = nullptr;
    std::vector<std::string> inDirs;
    int iters = 1, warmup = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dump" && i + 1 < argc) dumpDir = argv[++i];
        else if (a == "--dump-bb" && i + 1 < argc) bbDumpDir = argv[++i];
        else if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--warmup" && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (a.rfind("--", 0) == 0) { std::cerr << "unknown " << a << "\n"; return 2; }
        else if (!bbPath) bbPath = argv[i];
        else if (!hdPath) hdPath = argv[i];
        else if (!pluginSo && a.find(".so") != std::string::npos) pluginSo = argv[i];
        else inDirs.push_back(a);
    }
    if (!bbPath || !hdPath || inDirs.empty()) {
        std::cerr << "用法: " << argv[0] << " bb.engine hd.engine [plugin.so]"
                  " in_dir1 [in_dir2] --dump out_dir [--iters N] [--warmup W]\n";
        return 2;
    }
    if (pluginSo && !dlopen(pluginSo, RTLD_NOW)) {
        std::cerr << "dlopen(" << pluginSo << ") failed: " << dlerror() << "\n";
        return 1;
    }
    Logger logger;
    auto rt = std::unique_ptr<IRuntime>(createInferRuntime(logger));

    std::map<std::string, std::vector<char>> inputs;
    for (auto& d : inDirs) {
        auto m = load_manifest(d);
        if (m.empty()) { std::cerr << "bad manifest: " << d << "\n"; return 1; }
        for (auto& kv : m) inputs[kv.first] = std::move(kv.second);
    }
    std::cout << "manifest inputs: " << inputs.size() << "\n";

    auto loadEngine = [&](const char* path) -> std::unique_ptr<ICudaEngine> {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::cerr << "cannot open " << path << "\n"; return nullptr; }
        std::string buf((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
        return std::unique_ptr<ICudaEngine>(
            rt->deserializeCudaEngine(buf.data(), buf.size()));
    };
    auto bb = loadEngine(bbPath);
    auto hd = loadEngine(hdPath);
    if (!bb || !hd) { std::cerr << "deserialize FAILED\n"; return 1; }
    std::unique_ptr<IExecutionContext> bbCtx(bb->createExecutionContext());
    std::unique_ptr<IExecutionContext> hdCtx(hd->createExecutionContext());
    if (!bbCtx || !hdCtx) { std::cerr << "context create FAILED\n"; return 1; }
    std::cout << "bb IO " << bb->getNbIOTensors() << ", hd IO "
              << hd->getNbIOTensors() << "\n";

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ---- bb inputs: upload; bb outputs: allocate ----
    std::map<std::string, TensorBuf> bbBufs;
    for (int i = 0; i < bb->getNbIOTensors(); ++i) {
        const char* nm = bb->getIOTensorName(i);
        auto mode = bb->getTensorIOMode(nm);
        Dims d = bb->getTensorShape(nm);
        DataType dt = bb->getTensorDataType(nm);
        size_t bytes = (size_t)volume(d) * dtype_size(dt);
        TensorBuf& tb = bbBufs[nm];
        tb.bytes = bytes;
        if (cudaMalloc(&tb.dev, bytes) != cudaSuccess) {
            std::cerr << "cudaMalloc bb " << nm << " failed\n"; return 1;
        }
        if (mode == TensorIOMode::kINPUT) {
            auto it = inputs.find(nm);
            if (it == inputs.end()) {
                std::cerr << "bb input missing blob: " << nm << "\n"; return 1;
            }
            if (it->second.size() != bytes) {
                std::cerr << "bb input size mismatch " << nm << ": file "
                          << it->second.size() << " want " << bytes << "\n";
                return 1;
            }
            cudaMemcpyAsync(tb.dev, it->second.data(), bytes,
                            cudaMemcpyHostToDevice, stream);
        }
        bbCtx->setTensorAddress(nm, tb.dev);
        std::cout << "  bb " << (mode == TensorIOMode::kINPUT ? "in " : "out ")
                  << nm << " " << dtype_str(dt) << " bytes=" << bytes << "\n";
    }

    // ---- hd inputs: bb outputs wired D2D, rest uploaded; outputs alloc ----
    std::map<std::string, TensorBuf> hdBufs;
    for (int i = 0; i < hd->getNbIOTensors(); ++i) {
        const char* nm = hd->getIOTensorName(i);
        auto mode = hd->getTensorIOMode(nm);
        Dims d = hd->getTensorShape(nm);
        DataType dt = hd->getTensorDataType(nm);
        size_t bytes = (size_t)volume(d) * dtype_size(dt);
        TensorBuf& tb = hdBufs[nm];
        tb.bytes = bytes;
        if (mode == TensorIOMode::kINPUT) {
            if (bbBufs.count(nm)) {
                tb.dev = bbBufs[nm].dev;      // D2D 零拷贝交接
                std::cout << "  hd in  " << nm << " <- bb D2D\n";
                hdCtx->setTensorAddress(nm, tb.dev);
                continue;
            }
            auto it = inputs.find(nm);
            if (it == inputs.end()) {
                std::cerr << "hd input missing blob: " << nm << "\n"; return 1;
            }
            if (it->second.size() != bytes) {
                std::cerr << "hd input size mismatch " << nm << ": file "
                          << it->second.size() << " want " << bytes << "\n";
                return 1;
            }
            if (cudaMalloc(&tb.dev, bytes) != cudaSuccess) {
                std::cerr << "cudaMalloc hd " << nm << " failed\n"; return 1;
            }
            cudaMemcpyAsync(tb.dev, it->second.data(), bytes,
                            cudaMemcpyHostToDevice, stream);
        } else {
            if (cudaMalloc(&tb.dev, bytes) != cudaSuccess) {
                std::cerr << "cudaMalloc hd out " << nm << " failed\n";
                return 1;
            }
        }
        hdCtx->setTensorAddress(nm, tb.dev);
        std::cout << "  hd " << (mode == TensorIOMode::kINPUT ? "in " : "out ")
                  << nm << " " << dtype_str(dt) << " bytes=" << bytes << "\n";
    }
    cudaStreamSynchronize(stream);

    // ---- run ----
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    for (int i = 0; i < warmup; ++i) {
        bbCtx->enqueueV3(stream);
        hdCtx->enqueueV3(stream);
    }
    cudaStreamSynchronize(stream);
    float total_ms = 0;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(ev0, stream);
        bbCtx->enqueueV3(stream);
        hdCtx->enqueueV3(stream);
        cudaEventRecord(ev1, stream);
        cudaEventSynchronize(ev1);
        float ms;
        cudaEventElapsedTime(&ms, ev0, ev1);
        total_ms += ms;
        if (iters <= 5 || i == 0)
            std::cout << "iter " << i << ": " << ms << " ms\n";
    }
    std::cout << "MEAN " << total_ms / iters << " ms over " << iters
              << " iters\n";

    // ---- dump bb outputs (debug: boundary correctness) ----
    if (bbDumpDir) {
        mkdir(bbDumpDir, 0755);
        std::ofstream mf(std::string(bbDumpDir) + "/manifest.tsv");
        for (int i = 0; i < bb->getNbIOTensors(); ++i) {
            const char* nm = bb->getIOTensorName(i);
            if (bb->getTensorIOMode(nm) != TensorIOMode::kOUTPUT) continue;
            Dims d = bb->getTensorShape(nm);
            DataType dt = bb->getTensorDataType(nm);
            size_t bytes = bbBufs[nm].bytes;
            std::vector<char> host(bytes);
            cudaMemcpy(host.data(), bbBufs[nm].dev, bytes,
                       cudaMemcpyDeviceToHost);
            std::string fn = std::string(nm) + ".bin";
            std::ofstream f(std::string(bbDumpDir) + "/" + fn,
                            std::ios::binary);
            f.write(host.data(), bytes);
            mf << nm << "\t" << dtype_str(dt) << "\t";
            for (int k = 0; k < d.nbDims; ++k)
                mf << (k ? "," : "") << d.d[k];
            mf << "\t" << fn << "\n";
        }
        std::cout << "dumped bb to " << bbDumpDir << "\n";
    }

    // ---- dump hd outputs ----
    if (dumpDir) {
        mkdir(dumpDir, 0755);
        std::ofstream mf(std::string(dumpDir) + "/manifest.tsv");
        for (int i = 0; i < hd->getNbIOTensors(); ++i) {
            const char* nm = hd->getIOTensorName(i);
            if (hd->getTensorIOMode(nm) != TensorIOMode::kOUTPUT) continue;
            Dims d = hd->getTensorShape(nm);
            DataType dt = hd->getTensorDataType(nm);
            size_t bytes = hdBufs[nm].bytes;
            std::vector<char> host(bytes);
            cudaMemcpy(host.data(), hdBufs[nm].dev, bytes,
                       cudaMemcpyDeviceToHost);
            std::string fn = std::string(nm) + ".bin";
            std::ofstream f(std::string(dumpDir) + "/" + fn,
                            std::ios::binary);
            f.write(host.data(), bytes);
            mf << nm << "\t" << dtype_str(dt) << "\t";
            for (int k = 0; k < d.nbDims; ++k)
                mf << (k ? "," : "") << d.d[k];
            mf << "\t" << fn << "\n";
        }
        std::cout << "dumped to " << dumpDir << "\n";
    }
    return 0;
}
