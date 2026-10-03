# -*- coding: utf-8 -*-
"""generate deploy/onnx2engine3.cpp: board onnx2engine.cpp + --f16-notq.

--f16-notq: every pure-float layer NOT adjacent to Q/DQ gets an explicit
kHALF precision constraint + kPREFER_PRECISION_CONSTRAINTS, to test whether
per-layer constraints can steer Myelin to fp16 tactics under kINT8.
Plugin layers are skipped (their IO contract is the plugin's own).
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)
SRC = r"<REPO>\deploy\artifacts" \
      r"\onnx2engine_board.cpp"
DST = r"<REPO>\deploy" \
      r"\onnx2engine3.cpp"

t = open(SRC, encoding="utf-8").read()

# 1) argparse variable
old = "    bool f32NotQ = false;\n"
assert t.count(old) == 1
t = t.replace(old, old + "    bool f16NotQ = false;\n")

# 2) argparse branch
old = '        else if (!strcmp(argv[i], "--f32-notq")) f32NotQ = true;\n'
assert t.count(old) == 1
t = t.replace(old, old +
              '        else if (!strcmp(argv[i], "--f16-notq")) f16NotQ = true;\n')

# 3) usage line
old = '" [--f32-notq] [--no-tf32] [--ws-mb 256] [--plugins lib.so]"\n'
assert t.count(old) == 1
t = t.replace(old, '" [--f32-notq] [--f16-notq] [--no-tf32] '
                   '[--ws-mb 256] [--plugins lib.so]"\n')

# 4) the f16-notq block, inserted before the noTf32 block
F16BLOCK = r"""
    // --f16-notq: 除 Q/DQ 邻接区域外, 全部纯浮点层强制 FP16 精度约束。
    // 用途: kINT8 下 Myelin float 区在本板退化 (见 e_T12), 尝试用逐层
    // kHALF 约束 + kPREFER_PRECISION_CONSTRAINTS 把 Myelin 拉回 fp16 tactic。
    // 插件层不动 (IO 契约由插件自己声明)。
    if (f16NotQ) {
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
        int nf16 = 0;
        for (int i = 0; i < network->getNbLayers(); ++i) {
            auto* L = network->getLayer(i);
            if (L->getType() == LayerType::kQUANTIZE ||
                L->getType() == LayerType::kPLUGIN ||
                L->getType() == LayerType::kCONSTANT ||
                L->getType() == LayerType::kCAST) continue;
            bool pureFloat = true;
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
            L->setPrecision(DataType::kHALF);
            for (int j = 0; j < L->getNbOutputs(); ++j)
                L->setOutputType(j, DataType::kHALF);
            ++nf16;
        }
        config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
        std::cout << "f16-notq: " << nf16 << " layers constrained to FP16 ("
                  << dqOut.size() << " DQ, " << qIn.size() << " Q taps)\n";
    }

    if (noTf32) {
"""
old = "    if (noTf32) {\n"
assert t.count(old) == 1
t = t.replace(old, F16BLOCK)

open(DST, "w", encoding="utf-8", newline="\n").write(t)
print("wrote", DST)
