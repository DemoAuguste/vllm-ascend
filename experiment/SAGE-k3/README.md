# SAGE 参考实现（experiment/SAGE-k3）

- 论文：*SAGE: Pushing 2-Bit Vector-Quantized LLMs toward Real-World Usability*（arXiv:2610.14745v1, SJTU）
- 官方实现：https://github.com/Zincwolf/sage （GPL-3.0，Linux + NVIDIA + CUDA 12.8）
- 定位：**SAGE 的独立参考实现（NPU 友好）**，供 Qwen3.8-27B 混合精度量化（SAGE 低位档 + GPTQ-INT 高位档 + SLQ 分配）复用。方法可行性见 [`../SAGE-feasibility.md`](../SAGE-feasibility.md)。

> 官方实现依赖 `paroquant` 子模块 + 5 个 git patch + CUDA kernel，无法直接移植到 Ascend。
> 本实现按论文 Algorithm 1 / §3 公式与官方源码语义**从零重写**，保持算法等价，并在下文标注与官方的差异。

---

## 1. 一句话结论

SAGE = 球坐标解耦 VQ（符号显式 + 非负方向码本 + 共享半径网格）× **ESG 能量分层分组** × **ABS 自适应位切分** × **blockwise 误差补偿**。
编码载荷 **2.25 bpw**（`(v+log₂K+V)/v`）。本实现含 CPU demo/单测与 NPU 冒烟脚本；本地 CPU 已验证 13 项测试通过、blockwise 使端到端 MSE 降低约 43%。

---

## 2. 方法 → 实现映射

| 论文要素 | 公式 | 实现位置 |
|---|---|---|
| 球坐标分解 `w → (s, p, r)` | §3.1, Eq.1 | [`sage/spherical.py`](sage/spherical.py) `decompose` / `reconstruct_vectors` |
| 非负单位码本角参数化 | Eq.12-13 | [`sage/spherical.py`](sage/spherical.py) `angles_to_codewords` / `codewords_to_angles` |
| 共享半径网格量化 | Eq.2/14 | [`sage/spherical.py`](sage/spherical.py) `radius_grid_quantize` |
| ESG 能量分层交错置换 | Eq.5 | [`sage/grouping.py`](sage/grouping.py) `esg_permutation` / `energy_weights` |
| 加权球面 k-means | Eq.6 | [`sage/codebook.py`](sage/codebook.py) `weighted_spherical_kmeans` / `weighted_assignment` |
| ABS 逐矩阵选切分 | Eq.7-9 | [`sage/quantizer.py`](sage/quantizer.py) `SAGEQuantizer.prepare` |
| blockwise 误差补偿 | §3.4, Eq.10 | [`sage/blockwise.py`](sage/blockwise.py) `blockwise_optimize` |
| bpp 解析 | Eq.3/15 | [`sage/quantizer.py`](sage/quantizer.py) `SAGEQuantizer.bpp` |

关键公式：
```
每子向量 bit:  B = v + log₂K + V            (v=8, 默认 (K,V)=(64,4) → 18 bit)
payload bpw:   BPW = B / v = 1 + (log₂K+V)/8   → 2.25 bpw
半径网格:       µ_g = min r̃,  δ_g = (max r̃ − µ_g)/(2^V − 1),  r̂ = µ_g + q·δ_g
方向赋值:       z = argmax_k (Σ ω p c)² / (Σ ω c²)      (ω=1 时退化为余弦)
加权:           ω_i ∝ e_i^γ  (γ=0.75, 子向量内单位均值)
```

---

## 3. 目录结构

```
SAGE-k3/
  sage/
    config.py     SAGEConfig / Split（ABS 候选与超参）
    utils.py      device 解析、分块 argmax、softplus 逆、日志（NPU 友好）
    spherical.py  球坐标分解/重建、角参数化、共享半径网格
    grouping.py   通道能量估计 + ESG 交错置换 + 加权
    codebook.py   加权球面 k-means（非负正交象限）
    quantizer.py  SAGEQuantizer：拟合 / ABS / 编解码 / bpp
    blockwise.py  逐层 blockwise 误差补偿（STE）
  tests/test_sage.py   13 项单测（含模型级集成）
  demo.py              CPU 最小可验证 demo（A–F）
  model_quant.py       全模型 fake-quant（CPU/NPU）
  ppl_eval.py          WikiText-2 PPL 对比
  npv_verify.py        NPU 数值一致性 + 性能
  README.md            本文件
```

---

## 4. 快速开始

```bash
# 依赖：python>=3.9, torch(CPU 即可), numpy；（模型级脚本另需 transformers、datasets）
cd experiment/SAGE-k3

# 1) 单元测试（CPU）
python -m pytest -q            # 期望 13 passed

# 2) 数值 Demo（CPU）
python demo.py

# 3) 全模型 fake-quant 冒烟（CPU，示例）
python model_quant.py --device cpu --model Qwen/Qwen3-0.6B \
    --calib-seqs 4 --calib-len 256 --kmeans-iters 8 --epochs 5

# 4) NPU 环境验证（NPU 服务器）
ASCEND_RT_VISIBLE_DEVICES=5 python npv_verify.py

# 5) NPU 上跑 0.6B PPL（固定 2.0bpw 单切分 / ABS 三切分）
ASCEND_RT_VISIBLE_DEVICES=5 python ppl_eval.py --device npu --splits 64,2 --epochs 20
ASCEND_RT_VISIBLE_DEVICES=5 python ppl_eval.py --device npu --splits 64,4\;128,3\;256,2
```

`model_quant.py` 常用开关：`--no-esg`（消融）、`--no-adaptive-bits`（固定切分）、`--sequential`（序贯学生输入）、`--only mlp|attn`（分组消融）、`--limit-layers N`（快速冒烟）、`--calib-text <file>`（本地文本校准，免 datasets）。

---

## 5. NPU 实现与性能设计

对齐 `AGENTS.md` 的 NPU 性能要求：

1. **设备无关 + 惰性 `torch_npu`**：`sage/utils.py:resolve_device` 仅在指定 NPU 时才导入 `torch_npu`；CPU 环境零依赖，便于本地开发/CI。
2. **规避 `.item()` 热路径同步**：编码、k-means、半径量化全程张量化；收敛判据每轮至多一次标量同步（k-means 20 轮）。PCDVQ 中"NPU 上 2^a 次 `.item()` 灾难"的问题在本实现不存在。
3. **大矩阵分块**：方向赋值 `[N,8]@[8,K]` 按 N 分块（`chunk`），防 27B 单层（N≈11M）中间矩阵 OOM；`npv_verify.py` 已覆盖 27B MLP 规模吞吐测试。
4. **码本尺寸可控**：K≤256 常驻小张量，贴近官方 shared-memory 查表解码形态。
5. **blockwise 全张量化**：半径 STE、码本角、网格重算均向量化；每 epoch 只做一次离散评估（`discrete_mse`，无梯度）。
6. **显存回收**：逐层处理后 `torch.npu.empty_cache()`；校准激活与全精度输出缓存在 **CPU**，逐层搬到设备处理后再释放。
7. **可测性能**：`npv_verify.py` 报告编码吞吐（M 子向量/s）、ABS 拟合耗时、blockwise 每层耗时。

---

## 6. 与官方实现的差异（重要）

| 项 | 官方 | 本实现 | 影响 |
|---|---|---|---|
| 优化框架 | `paroquant` 子模块 + git patch | 独立 Python 重写 | 无 CUDA 依赖，可读性/可移植性优先 |
| blockwise 学生输入 | 序贯（前序量化块输出） | 默认 FP 捕获；`--sequential` 序贯 | 默认更快；序贯更贴论文，成本 O(L) 前向 |
| 数值精度 | FP16 网格仿真 + AMP | FP32 参考实现 | 更稳，非逐位一致 |
| 半径网格 | 训练期 FP16 网格参数 | FP32 网格 | bpp 记账一致（按 FP16 计开销） |
| codebook | 非负正交象限（sign 外提） | 同 | 一致 |
| 导出 | 密集 QDQ + 打包 CUDA 格式 | 仅密集 fake-quant checkpoint | 打包/kernel 属部署工作，超出本参考范围 |
| 支持模型 | qwen3 / qwen3_moe / qwen3_5 / Ministral-3 | 任意 HF `nn.Linear`（可过滤） | 更通用 |

**未实现的官方能力**（按本任务范围裁剪）：CUDA 打包格式与解码 kernel、MoE 专家分片、分布式（多卡）拟合、压缩导出。

---

## 7. 与 PCDVQ-k3 的衔接

| | PCDVQ-k3 | SAGE-k3（本实现） |
|---|---|---|
| 方向码本 | 固定 E8 格 + 贪心 FPS | 可学习（激活加权球面 k-means） |
| 幅值/半径 | chi-Lloyd-Max 全局码本 | 逐组共享均匀网格（V bit） |
| 数据感知 | 无（模型无关） | ESG 通道能量 + 逐矩阵 ABS |
| 误差补偿 | **无**（纯 fake-quant，导致 PPL 崩溃） | **blockwise**（输出 MSE + STE） |
| 位档 | a=14/15/16 × b=2 | (K,V)∈{(64,4),(128,3),(256,2)} 等 |

SAGE 与 PCDVQ 同属"方向+幅值"形状-增益分解，**接口一致（逐层 fake-quant 权重）**，可直接替换 PCDVQ 作为混合精度方案的**低位档算子**，并补齐其缺失的误差补偿。

---

## 8. 本地已验证结果（CPU，Python 3.10 / torch 2.9.1）

```
python -m pytest -q        → 13 passed
python demo.py（关键输出）
  A. 球坐标分解重建最大绝对误差 3.7e-09
  B. 角参数化往返偏差 3.5e-06；码本非负单位
  C. 半径网格 V=2/3/4 相对误差 0.124/0.053/0.025；零步长组正确退化
  D. ESG 置换合法；子向量内能量变异：相邻分组 0.022 → ESG 0.964
  E. ABS 三切分（均 2.25bpw）选激活加权误差最小者；total bpp 2.562
  F. 两层 MLP 端到端 MSE：无补偿 7.51e-03 → blockwise 4.29e-03（降低 42.9%）
```

F 项即"**有 blockwise 误差补偿后 2bit 级 VQ 不再崩溃**"的机制证据——本实现相对 PCDVQ 的核心增益。

---

## 9. 局限与后续

- **NPU 实测结果（2026-10-16，见 [../progress.md](../progress.md) 阶段 5）**：本实现已在 Ascend 910B3 上跑通 `npv_verify.py` / `model_quant.py` / `ppl_eval.py`。
  - 核心算法正确：合成高斯与真实权重（无 ESG）(64,4)/(256,2) 的 rel≈0.31/0.27、方向余弦≈0.95/0.97。
  - **纯 fake-quant（无微调）下 Qwen3-0.6B 在 2.0–2.75bpw 全部崩溃**（PPL 135–1.2e6）；Qwen3-4B（论文验证规模）no-esg 2.25bpw 为 PPL 13.20→36.66（+178%）。
  - **ESG 在 WikiText 小校准下有害**（rel 0.31→0.63），主线建议先 `--no-esg`。
- **校准数据**：官方用目标模型自采样（self-rollout，1920×2048）；本实现默认 WikiText-2，规模远小于官方 —— 需加自采样以复现官方精度。
- **块级序贯**：`--sequential` 为逐层重前向的朴素实现，成本 O(L)；可优化为单次传播。
- **部署 kernel**：Ascend 无 SAGE VQ 解码算子；真实压缩层属后续工作（精度验证走 fake-quant）。
- **位宽范围**：SAGE 为 ≤3bit 极低位方法，工作点约 2.0–2.5 bpw；高位档由 GPTQ-INT 承担（见 [`../SAGE-feasibility.md`](../SAGE-feasibility.md) §4）。
