// onnx2engine —— 板端 ONNX -> TensorRT engine 编译工具 (TRT 8.6)
// 用法: onnx2engine model.onnx out.engine [--fp16] [--ws-mb 256] [--plugins lib.so]
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // TRT_LOG_LEVEL=VERBOSE 可看断言前最后处理的层
        const char* env = getenv("TRT_LOG_LEVEL");
        Severity min = (env && !strcmp(env, "VERBOSE")) ? Severity::kVERBOSE
                                                        : Severity::kWARNING;
        if (severity <= min) std::cerr << "[TRT] " << msg << "\n";
    }
};

// ---------------------------------------------------------------------------
// 隐式 INT8 熵校准 (绕开 QDQ 路线): 样本目录与 run_engine 同 manifest.tsv 格式
//   name\tdtype\tdims\tfile
// ---------------------------------------------------------------------------
struct CalibSample { std::map<std::string, std::vector<char>> blobs; };

static bool load_sample_dir(const std::string& dir, CalibSample& s) {
    FILE* f = fopen((dir + "/manifest.tsv").c_str(), "r");
    if (!f) { std::cerr << "calib: no manifest in " << dir << "\n"; return false; }
    char line[1024];
    bool ok = true;
    while (fgets(line, sizeof(line), f)) {
        char* p = line;
        auto field = [&p]() {
            char* st = p;
            while (*p && *p != '\t' && *p != '\n') ++p;
            std::string out(st, p - st);
            if (*p) ++p;
            while (!out.empty() && (out.back() == '\r' || out.back() == ' ')) out.pop_back();
            return out;
        };
        std::string name = field();
        field();  // dtype
        field();  // dims (体积由文件大小决定)
        std::string file = field();
        if (name.empty()) continue;
        FILE* b = fopen((dir + "/" + file).c_str(), "rb");
        if (!b) { std::cerr << "calib: missing " << dir << "/" << file << "\n"; ok = false; break; }
        fseek(b, 0, SEEK_END);
        long sz = ftell(b);
        fseek(b, 0, SEEK_SET);
        std::vector<char> buf(sz);
        if (fread(buf.data(), 1, sz, b) != (size_t)sz) { fclose(b); ok = false; break; }
        fclose(b);
        s.blobs[name] = std::move(buf);
    }
    fclose(f);
    return ok;
}

class BlobCalibrator : public IInt8Calibrator {
public:
    BlobCalibrator(std::vector<CalibSample> samples, std::vector<char> cache)
        : samples_(std::move(samples)), cache_(std::move(cache)) {}

    int getBatchSize() const noexcept override { return 1; }

    bool getBatch(void* bindings[], const char* names[], int nb) noexcept override {
        if (cur_ >= (int)samples_.size()) return false;
        for (int i = 0; i < nb; ++i) {
            auto it = samples_[cur_].blobs.find(names[i]);
            if (it == samples_[cur_].blobs.end()) {
                std::cerr << "calib: sample " << cur_ << " missing input " << names[i] << "\n";
                return false;
            }
            if (dev_.find(names[i]) == dev_.end()) {
                void* d = nullptr;
                if (cudaMalloc(&d, it->second.size()) != cudaSuccess) return false;
                dev_[names[i]] = d;
            }
            if (cudaMemcpy(dev_[names[i]], it->second.data(), it->second.size(),
                           cudaMemcpyHostToDevice) != cudaSuccess)
                return false;
            bindings[i] = dev_[names[i]];
        }
        ++cur_;
        std::cerr << "calib: batch " << cur_ << "/" << samples_.size() << " fed\n";
        return true;
    }

    const void* readCalibrationCache(size_t& length) noexcept override {
        length = cache_.size();
        return cache_.empty() ? nullptr : cache_.data();
    }
    void writeCalibrationCache(const void* ptr, size_t length) noexcept override {
        cache_.assign((const char*)ptr, (const char*)ptr + length);
    }
    CalibrationAlgoType getAlgorithm() noexcept override {
        return CalibrationAlgoType::kENTROPY_CALIBRATION_2;
    }

    const std::vector<char>& cache() const { return cache_; }

private:
    std::vector<CalibSample> samples_;
    std::vector<char> cache_;
    std::map<std::string, void*> dev_;
    int cur_ = 0;
};

static void printDims(const char* tag, const char* name, const Dims& d) {
    std::cout << tag << " " << name << ": [";
    for (int i = 0; i < d.nbDims; ++i) std::cout << (i ? "," : "") << d.d[i];
    std::cout << "]\n";
}

int main(int argc, char** argv) {
    const char* onnxPath = nullptr;
    const char* outPath  = nullptr;
    const char* pluginSo = nullptr;
    const char* calibData = nullptr;
    const char* calibCache = nullptr;
    const char* f32Substr = nullptr;
    bool fp16 = false;
    bool int8 = false;
    bool strict = false;
    bool f32NotQ = false;
    bool noTf32 = false;
    size_t wsMB = 256;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--fp16")) fp16 = true;
        else if (!strcmp(argv[i], "--int8")) int8 = true;
        else if (!strcmp(argv[i], "--strict")) strict = true;
        else if (!strcmp(argv[i], "--f32-notq")) f32NotQ = true;
        else if (!strcmp(argv[i], "--no-tf32")) noTf32 = true;
        else if (!strcmp(argv[i], "--ws-mb") && i + 1 < argc) wsMB = strtoul(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--plugins") && i + 1 < argc) pluginSo = argv[++i];
        else if (!strcmp(argv[i], "--calib-data") && i + 1 < argc) calibData = argv[++i];
        else if (!strcmp(argv[i], "--calib-cache") && i + 1 < argc) calibCache = argv[++i];
        else if (!strcmp(argv[i], "--f32-substr") && i + 1 < argc) f32Substr = argv[++i];
        else if (!onnxPath) onnxPath = argv[i];
        else if (!outPath)  outPath  = argv[i];
    }
    if (!onnxPath || !outPath) {
        std::cerr << "用法: " << argv[0] << " model.onnx out.engine [--fp16] [--int8] [--strict]"
                  " [--f32-notq] [--no-tf32] [--ws-mb 256] [--plugins lib.so]"
                  " [--calib-data dir1,dir2] [--calib-cache f]\n";
        return 2;
    }

    // 自定义算子 plugin 库 (静态注册器在 dlopen 时挂到 registry)
    if (pluginSo) {
        if (!dlopen(pluginSo, RTLD_NOW)) {
            std::cerr << "dlopen(" << pluginSo << ") failed: " << dlerror() << "\n";
            return 1;
        }
        std::cout << "plugin loaded: " << pluginSo << "\n";
    }

    Logger logger;
    std::cout << "TRT lib: 0x" << std::hex << getInferLibVersion() << std::dec << "\n";

    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(logger));
    if (!builder) { std::cerr << "createInferBuilder failed\n"; return 1; }

    // 注意: 板上 8.6.12 runtime 实测仍要求显式置 explicit batch 位(OSS 8.6.1 头文件里
    // kEXPLICIT_BATCH=0 表示"默认即显式", 但这个 runtime 不认 —— 传 0 建出来是隐式网络)
    auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(1));
    auto config  = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    auto parser  = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!network || !config || !parser) { std::cerr << "network/config/parser create failed\n"; return 1; }
    std::unique_ptr<BlobCalibrator> calibrator;  // 生命周期需覆盖 build

    if (!parser->parseFromFile(onnxPath, static_cast<int>(ILogger::Severity::kWARNING))) {
        for (int i = 0; i < parser->getNbErrors(); ++i)
            std::cerr << "[ONNX error] " << parser->getError(i)->desc() << "\n";
        std::cerr << "ONNX parse FAILED: " << onnxPath << "\n";
        return 1;
    }
    std::cout << "ONNX parse OK\n";
    for (int i = 0; i < network->getNbInputs(); ++i)
        printDims("  input", network->getInput(i)->getName(), network->getInput(i)->getDimensions());
    for (int i = 0; i < network->getNbOutputs(); ++i)
        printDims("  output", network->getOutput(i)->getName(), network->getOutput(i)->getDimensions());

    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, wsMB << 20);

    // 按层名子串强制 FP32 (--f32-substr a,b,c): 时序 bank/TopK 等对半精度敏感的
    // 区域保持全精度, 其余照常走 FP16
    if (f32Substr) {
        std::vector<std::string> subs;
        std::string s(f32Substr);
        for (size_t pos = 0; pos <= s.size();) {
            size_t c = s.find(',', pos);
            if (c == std::string::npos) c = s.size();
            if (c > pos) subs.push_back(s.substr(pos, c - pos));
            pos = c + 1;
        }
        int nf32 = 0;
        for (int i = 0; i < network->getNbLayers(); ++i) {
            auto* L = network->getLayer(i);
            const char* nm = L->getName();
            bool hit = false;
            for (auto& t : subs)
                if (strstr(nm, t.c_str())) { hit = true; break; }
            if (!hit) continue;
            // 只动纯浮点层: 带任何 int32/bool 输出的层 (Constant/TopK/Cast/Cmp)
            // 强转 FLOAT 会在 validate 阶段炸掉
            bool pureFloat = L->getType() != LayerType::kCONSTANT &&
                             L->getType() != LayerType::kCAST;  // Cast 的目标类型来自属性, 强转必炸
            for (int j = 0; pureFloat && j < L->getNbOutputs(); ++j) {
                DataType ot = L->getOutputType(j);
                if (ot != DataType::kFLOAT && ot != DataType::kHALF) pureFloat = false;
            }
            if (!pureFloat) continue;
            L->setPrecision(DataType::kFLOAT);
            for (int j = 0; j < L->getNbOutputs(); ++j)
                L->setOutputType(j, DataType::kFLOAT);
            ++nf32;
        }
        config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
        std::cout << "f32-substr: " << nf32 << " layers forced to FP32\n";
    }

    // --f32-notq: 除 Q/DQ 邻接区域 (显式量化区) 外, 全部纯浮点层强制 FP32。
    // 用途: QDQ 图只在 backbone/neck 保留 INT8、head 免量化时, 让 head 以
    // FP32 而非 FP16 执行 (fp16 在 6 层 decoder 逐 op 舍入会混沌放大)。
    // Q/DQ 判定: kQUANTIZE 层输出 kINT8 的是 Q (记录其输入), 输出 kFLOAT
    // 的是 DQ (记录其输出)。层邻接 = 输入出自 DQ 或输出喂给 Q。
    if (f32NotQ) {
        std::set<std::string> dqOut, qIn;
        for (int i = 0; i < network->getNbLayers(); ++i) {
            auto* L = network->getLayer(i);
            if (L->getType() != LayerType::kQUANTIZE) continue;
            bool isQ = L->getNbOutputs() > 0 &&
                       L->getOutput(0)->getType() == DataType::kINT8;
            if (isQ) {
                for (int j = 0; j < L->getNbInputs(); ++j)
                    qIn.insert(L->getInput(j)->getName());
            } else {
                for (int j = 0; j < L->getNbOutputs(); ++j)
                    dqOut.insert(L->getOutput(j)->getName());
            }
        }
        int nf32 = 0;
        for (int i = 0; i < network->getNbLayers(); ++i) {
            auto* L = network->getLayer(i);
            if (L->getType() == LayerType::kQUANTIZE) continue;
            bool pureFloat = L->getType() != LayerType::kCONSTANT &&
                             L->getType() != LayerType::kCAST;
            for (int j = 0; pureFloat && j < L->getNbOutputs(); ++j) {
                DataType ot = L->getOutputType(j);
                if (ot != DataType::kFLOAT && ot != DataType::kHALF)
                    pureFloat = false;
            }
            if (!pureFloat) continue;
            bool inZone = false;
            for (int j = 0; !inZone && j < L->getNbInputs(); ++j)
                if (dqOut.count(L->getInput(j)->getName())) inZone = true;
            for (int j = 0; !inZone && j < L->getNbOutputs(); ++j)
                if (qIn.count(L->getOutput(j)->getName())) inZone = true;
            if (inZone) continue;
            L->setPrecision(DataType::kFLOAT);
            for (int j = 0; j < L->getNbOutputs(); ++j)
                L->setOutputType(j, DataType::kFLOAT);
            ++nf32;
        }
        config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
        std::cout << "f32-notq: " << nf32 << " layers forced to FP32 ("
                  << dqOut.size() << " DQ, " << qIn.size() << " Q taps)\n";
    }

    if (noTf32) {
        config->clearFlag(BuilderFlag::kTF32);
        std::cout << "builder flags: TF32 disabled\n";
    }

    if (strict) {
        // 禁止精度提升 (8.6.12 的 chooseHigherPrecision 对 int32/bool tensor 断言)
        config->setFlag(BuilderFlag::kSTRICT_TYPES);
        std::cout << "builder flags: STRICT_TYPES\n";
    }
    if (int8) {
        config->setFlag(BuilderFlag::kINT8);
        if (calibData || calibCache) {
            // 隐式量化路线: 熵校准 (图内无 QDQ 时唯一可行路线)
            std::vector<char> preCache;
            if (calibCache) {
                std::ifstream cf(calibCache, std::ios::binary);
                if (cf) preCache.assign(std::istreambuf_iterator<char>(cf),
                                        std::istreambuf_iterator<char>());
            }
            std::vector<CalibSample> samples;
            if (calibData && preCache.empty()) {
                std::string dirs(calibData);
                for (size_t pos = 0; pos <= dirs.size();) {
                    size_t c = dirs.find(',', pos);
                    if (c == std::string::npos) c = dirs.size();
                    std::string d = dirs.substr(pos, c - pos);
                    if (!d.empty()) {
                        CalibSample s;
                        if (!load_sample_dir(d, s)) return 1;
                        samples.push_back(std::move(s));
                    }
                    pos = c + 1;
                }
            }
            if (!preCache.empty())
                std::cout << "calib cache loaded: " << preCache.size() << " bytes\n";
            else
                std::cout << "implicit int8 calibration over " << samples.size()
                          << " sample(s)\n";
            calibrator = std::unique_ptr<BlobCalibrator>(
                new BlobCalibrator(std::move(samples), std::move(preCache)));
            config->setInt8Calibrator(calibrator.get());
            std::cout << "builder flags: INT8 (implicit entropy calibration)\n";
        } else {
            // QDQ 显式量化图必须开 kINT8 (图内已带 scale/zp, 无需校准)
            std::cout << "builder flags: INT8 (explicit QDQ)\n";
        }
    }
    if (fp16) {
        if (builder->platformHasFastFp16()) config->setFlag(BuilderFlag::kFP16);
        else std::cout << "note: no fast FP16 on this platform, --fp16 ignored\n";
    }

    std::cout << "building engine...\n";
    auto plan = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan) { std::cerr << "buildSerializedNetwork FAILED\n"; return 1; }

    // 校准缓存落盘 (下次 --calib-cache 直读, 免校准)
    if (calibrator && calibCache && !calibrator->cache().empty()) {
        std::ofstream cf(calibCache, std::ios::binary);
        cf.write(calibrator->cache().data(), calibrator->cache().size());
        std::cout << "calib cache saved: " << calibCache
                  << " (" << calibrator->cache().size() << " bytes)\n";
    }

    // 回读验证: 保证序列化产物能被 runtime 正常反序列化
    auto rt     = std::unique_ptr<IRuntime>(createInferRuntime(logger));
    auto engine = std::unique_ptr<ICudaEngine>(rt->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine) { std::cerr << "deserialize check FAILED\n"; return 1; }
    std::cout << "deserialize check OK: " << engine->getNbIOTensors() << " IO tensors\n";

    std::ofstream f(outPath, std::ios::binary);
    if (!f) { std::cerr << "cannot open output: " << outPath << "\n"; return 1; }
    f.write(static_cast<const char*>(plan->data()), plan->size());
    if (!f) { std::cerr << "write failed\n"; return 1; }
    std::cout << "=== OK: " << outPath << " (" << plan->size() << " bytes) ===\n";
    return 0;
}
