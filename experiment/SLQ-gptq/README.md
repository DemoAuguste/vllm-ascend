# SLQ + GPTQ 混合精度量化（experiment/SLQ-gptq）

- 定位：主线量化方案（阶段 7）。**GPTQ** 作为逐层位宽量化算子（Hessian 二阶误差补偿），**SLQ**（Shapley/线性敏感度 + ILP）做全局混合精度位宽分配。
- GPTQ 算子移植并泛化自昇腾 [AMCT](https://gitcode.com/cann/amct)（Apache-2.0，`amct_pytorch/algorithms/quant/gptq.py`，Huawei Technologies 2026），位宽由 `{4,8}` 扩展到任意 int。
- 背景：阶段 5 证实纯 fake-quant 的 VQ（PCDVQ/SAGE）在低位崩溃（缺误差补偿）；GPTQ 的 Hessian 补偿正是缺失环节。见 [`../progress.md`](../progress.md)。

## 1. 目录结构

```
SLQ-gptq/
  gptq/            GPTQ 算子
    quant.py       INT 对称量化-反量化（per-channel / per-group，任意位宽）
    core.py        GPTQ（Hessian 累积 + 列式二阶补偿；CPU/NPU 通用）
  slq/             SLQ
    sensitivity.py 线性敏感度（逐层单点替换测 NLL）
    ilp.py         ILP 位宽分配（拉格朗日对偶二分 + scipy.milp 精确解）
  model_quant.py   全模型 GPTQ fake-quant（统一位宽 / 逐层 bit_map）
  ppl_eval.py      WikiText-2 PPL（基线 vs 量化）
  slq_allocate.py  SLQ 驱动：逐层多档位量化 → 敏感度 → ILP → bit_map.json
  tests/test_gptq.py
```

## 2. 快速开始

```bash
cd experiment/SLQ-gptq

# 单测（CPU）
python -m pytest tests/test_gptq.py -q

# ① 统一位宽 GPTQ PPL（NPU）
ASCEND_RT_VISIBLE_DEVICES=5 python ppl_eval.py --device npu \
    --model Qwen/Qwen3-0.6B --bits 4 --group-size 128

# ② SLQ 混合精度：分配 bit_map 后评测
ASCEND_RT_VISIBLE_DEVICES=5 python slq_allocate.py --device npu \
    --model Qwen/Qwen3-0.6B --bits 3,4,8 --target-avg 3.5 \
    --group-size 128 --out bits_06b.json
ASCEND_RT_VISIBLE_DEVICES=5 python ppl_eval.py --device npu \
    --model Qwen/Qwen3-0.6B --bit-map bits_06b.json
```

## 3. 已验证结果

- **GPTQ 单测**：6 passed（位宽泛化 {2,3,4,6,8}、误差补偿优于 RTN、dead 列处理、per-group）。
- **AMCT 参考算子一致性**：与 AMCT `gptq.py` 同构（Hessian 累积 `sqrt(2/n)`、damp、CPU Cholesky 求逆、128 列分块、逐列二阶补偿）。
- **Qwen3-0.6B GPTQ int4 g128**：WikiText-2 PPL **20.0419 → 22.8689（+14.11%）**，平均 rel 0.1748（对比 SAGE 2.25bpw 的崩溃 PPL≈1102）。

## 4. 与 AMCT 的差异

| 项 | AMCT | 本实现 |
|---|---|---|
| 位宽 | int4 / int8 | 任意 int（{2..8}） |
| 分组 | per-channel（int）、HiF4 按 64 列组 | per-channel 或 per-group 可配 |
| 依赖 | amct_pytorch 全栈（onnx 等可选） | 仅 torch，可独立运行 |
| 求解缓存 | 每 bit 重新观察 | Hessian/置换/求逆跨位宽复用（SLQ 多档位高效） |

> 仅密集 fake-quant（精度验证用）；真实压缩 kernel/packing 属部署工作。
