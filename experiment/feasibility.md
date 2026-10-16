# PCDVQ + SLQ 混合精度量化可行性分析报告

- 日期：2026-10-16
- 目标模型：Qwen3.8-27B（`Qwen3_5ForConditionalGeneration`，64 层 hybrid：48×linear_attention + 16×full_attention，hidden 5120，vocab 248K，VLM）
- 任务目标：**精度损失 < 0.5% 前提下实现更低 bit 量化**（对照基准：常规 W4 量化）
- 结论先行：**可行，推荐"PCDVQ 低位宽档 + GPTQ 中高位宽档 + SLQ 敏感度驱动的 ILP 混合精度分配 + SLQ-TL 单点标定"的组合方案**；精度验证走 fake-quant BF16 推理路径，真实部署 kernel 列为后续独立工作。主要风险与对策见 §6。

---

## 1. 两方法技术要点（调研结论摘要）

### 1.1 SLQ（Statistically-Lossless Quantization, arXiv:2605.02404, COLM 2026, IST-DASLab）

- **核心**：把"无损"直接定义在模型输出统计量上，而非权重/输出重构 MSE。
  - **SLQ-DL**（分布无损）：量化模型与原模型下一 token 分布统计不可分（EAR≥0.99），达成 5.0–6.6 bpp
  - **SLQ-TL**（任务无损）：7 个 zero-shot 基准平均恢复 ≥99% BF16，达成 **3.3–4.7 bpp**（Qwen3.5-27B：**3.30 bpp / 99.1% 恢复**）
- **流水线**（官方代码 4 阶段）：
  1. `slq_quantize.py`：逐层 GPTQ（非对称 INT、group 128、每 group 一个 GPTQ block、单 Hessian 一次产出全部位宽 {2..8} 的量化权重、cal_bits=4 序贯传播）
  2. `slq_build_database.py`：**多 bitwidth Shapley 敏感度**（Algorithm 1：对每个目标位宽 b* 独立跑 P=50 个随机排列的二元博弈，记录组从 b_max 切到 b* 的边际 NLL/KL/EAR 退化）+ 线性 α 估计（高斯噪声注入 15 档，便宜 2 个数量级）
  3. `slq_search_dl.py`：ILP 多选背包（ortools SCIP）最小化总预测退化，参数量加权 bpp 预算约束下二分搜索最小 bpp
  4. `slq_build_checkpoint.py`：拼装 HF checkpoint（fake-quant，BF16 权重落在整数格点上，stock transformers/vLLM 直接加载）
- **SLQ-TL 标定**（Algorithm 2，避免 benchmark-in-the-loop）：锚点 = uniform-4bit 的 lm-eval 实测；ρ = 实测KL/预测KL 校正；反解 threshold 后 ILP 二分；2× guardrail 拒绝漂移过大配置
- **敏感度结构实证**：K/V 最敏感、MLP 最不敏感、首尾层比中间层敏感
- **复现注意**：EAR/KL 必须 top-10 截断 + 基线 top-10 索引上重归一化；实际 bpp = b + 0.156（g128 scale/zero 开销）

### 1.2 PCDVQ（Polar Coordinate Decoupled Vector Quantization, arXiv:2506.05432, Houmo AI/哈工大深圳）

- **核心**：权重方向比幅值敏感得多（LLaMA-2-7B 实证：方向聚类 zero-shot 掉 46.5% vs 幅值仅 2.3%）。因此：
  - **PCD**：k=8 维权重向量 → 极坐标（方向 + 模长），分别量化，方向给多 bit（a=14~16），幅值固定 b=2
  - **DACC**：码本由理论分布闭式构造，**零训练、模型无关、全模型共享**：
    - 方向码本：RHT 高斯化后方向在球面均匀分布 → E8 格（最密球堆积）方向池上贪心最远点采样（Algorithm 1）
    - 幅值码本：r ~ chi(k) 分布 → 一维 Lloyd-Max 最优标量量化（Algorithm 2）
  - 前置 **Standard Gaussian Regularization**：随机化 Hadamard（RHT）+ 每列 scale s=‖x‖/√p，使权重分布统一为 N(0,1)
- **结果**：2.125-bit LLaMA-2-7B Wiki2 PPL 5.68（FP16 5.12），优于 QuIP# 2-bit（6.19）、VPTQ 2.02（5.83）、GPTQ 2.125（12.62）；zero-shot 平均 60.44 vs FP16 62.24
- **论文缺陷（复现必须修正）**：
  1. 位算术自相矛盾：a=16,b=2,k=8 → 2.25 bit/w，论文却称 2.125（严格 2.125 需 a=15）
  2. E8 方向池未定义扩展方式（最小壳层仅 240 向量，2^14 码字需 norm²≤8 壳层共 26640 个）
  3. 极坐标公式 Eq.6 字面实现丢失 v_k 符号（应改标准超球坐标 atan2）
  4. 非官方 notebook 的 Hadamard 正则化有实质 bug（构造了 H 但从未相乘），不可照抄
  5. 微调细节未给出（沿用 QuIP# block-wise + e2e 微调，WikiText2/C4 校准）

---

## 2. 组合方案设计：PSQ（PCDVQ–SLQ hybrid Quantization）

### 2.1 总体架构

```
                 ┌────────────────────────────────────────────────┐
 校准数据        │  阶段A 逐层量化算子池构建                        │
 FineWeb-Edu ──► │  · GPTQ 非对称INT g128：bitwidths {3,4,6,8}      │
 (1024×8192)     │  · PCDVQ VQ：{(a=14,b=2)→2.0, (a=15,b=2)→2.125,│
                 │             (a=16,b=2)→2.25} bit/w              │
                 │  → 每层每档一份 fake-quant 权重库（磁盘）         │
                 └───────────────────┬────────────────────────────┘
                                     ▼
                 ┌────────────────────────────────────────────────┐
                 │  阶段B 敏感度数据库（SLQ Algorithm 1 / 线性估计）  │
                 │  组定义适配 Qwen3.8：                            │
                 │   full_attn 层: {qkv}, {o}, {gate+up}, {down}   │
                 │   linear_attn 层: {in_proj_qkv+a+b+z}, {out},   │
                 │                   {gate+up}, {down}             │
                 │  φ_m^(q)（q∈量化档）= 组切到该档的边际度量退化      │
                 └───────────────────┬────────────────────────────┘
                                     ▼
                 ┌────────────────────────────────────────────────┐
                 │  阶段C ILP 混合精度分配 + SLQ-TL 单点标定         │
                 │  min Σ c_{m,q}  s.t. 每组恰选一档                │
                 │       Σ params_m · bpp_q ≤ Budget               │
                 │  Budget 二分 → 最小平均 bpp @ 精度约束            │
                 └───────────────────┬────────────────────────────┘
                                     ▼
                 ┌────────────────────────────────────────────────┐
                 │  阶段D checkpoint 拼装 + 精度评测                 │
                 │  fake-quant BF16 → transformers/vllm-ascend     │
                 │  PPL(wikitext2/c4) + lm-eval zero-shot          │
                 └────────────────────────────────────────────────┘
```

### 2.2 关键设计决策

**D1：量化算子池 = PCDVQ（低档）∪ GPTQ-INT（高档），而非 PCDVQ 全档。**
理由：a) PCDVQ 的优势区间在 ≤2.25 bit（论文 Table 1-2，其 3bit+ 未验证）；b) SLQ 实证敏感组（K/V、首尾层）在 4-6bit 才能保精度，GPTQ-INT 在该区间成熟且有 vllm-ascend W4A16 kernel 路径；c) 两者都以"逐层 fake-quant 权重 + 组级敏感度"的接口进入 SLQ 框架，正交可组合。

**D2：PCDVQ 修正实现（偏离论文之处全部显式记录）：**
1. 极坐标：标准超球坐标 `φ_i = atan2(√(Σ_{j>i} v_j²), v_i)`（i<k-1），`φ_{k-1} = atan2(v_k, v_{k-1})`（值域 [-π,π]），保留符号信息
2. 方向匹配在**笛卡尔单位向量空间**用余弦相似度（几何意义明确，与非官方 notebook 一致；论文 Eq.7 字面为角度空间，两者结果不同，取前者并在报告中注明）
3. E8 池扩展：枚举 norm²≤8 全部 26640 个格点归一化（覆盖 2^14）；2^16 需 norm²≤12（117360 个，FPS 一次性离线成本仍为分钟级）
4. RHT 沿输入维（QuIP# 惯例 `W·H`，H 可吸收到激活侧/逆变换回权重侧二选一，fake-quant 路径直接在权重侧做逆变换，部署零开销）
5. 幅值码本 b=2（4 码字，chi(8) Lloyd-Max；非官方 notebook 的 k=8,b=3 输出 [1.4241, 1.8740, 2.2255, 2.5455, 2.8623, 3.1978, 3.5805, 4.0698] 可作单测金标准）
6. 位档取 a=14（2.0）、a=15（2.125）两档先行，a=16 视精度余量加档

**D3：敏感度估计 0.6B 用完整 Shapley、27B 用线性 α 估计为主 + 抽样 Shapley 校验。**
成本核算：Qwen3.8-27B 组数 M≈64×4+16×2=288（linear_attn 48 层×4 组 + full_attn 16 层×4 组，粗算）；Shapley 全量 = P×M×|Q| ≈ 50×288×6 ≈ 8.6 万次 512×2048 校准前向，27B 上完全不可行。线性估计 = 15×M≈4300 次噪声注入前向（t∈[0.01,0.3]），可行。SLQ 论文消融显示线性与 Shapley 在 bpp 中段差异有限；0.6B 上先做 Shapley vs 线性一致性验证，再决定 27B 校验抽样密度。
注：线性 α 估计针对"加性噪声"代理，对 VQ 量化器的适用性需在 0.6B 上与 Shapley 对比验证（风险 R3）。

**D4：Qwen3.8-27B 架构适配：**
- 量化目标（正则匹配 `model.layers.N.` 下的 Linear）：
  - full_attn 层：`self_attn.{q,k,v,o}_proj`、`mlp.{gate,up,down}_proj`
  - linear_attn 层（gated delta net）：`linear_attn.{in_proj_qkv, in_proj_a, in_proj_b, in_proj_z, out_proj}`、`mlp.{gate,up,down}_proj`
- 不量化：vision tower/merger、embed/lm_head、MTP 层（`mtp.*`）、norm、conv
- 分组（兼容 vllm-ascend fused kernel 约束）：{q+k+v}、{gate+up} 同组；linear_attn 的 {in_proj_qkv, in_proj_a, in_proj_b, in_proj_z} 同组
- in_proj_qkv 为融合 QKV 的宽 Linear（输入 5120），GPTQ 沿输入维量化无障碍
- 依赖 transformers ≥5.8（agent env 5.14.1 ✓）；`trust_remote_code` + key 前缀 `model.language_model.layers.` 重映射（VLM 嵌套结构）

**D5：精度目标定义（任务"精度损失<0.5%"的操作化）：**
- 主口径：**下游任务平均准确率相对损失 ≤0.5%**（即恢复率 ≥99.5%，比 SLQ-TL 的 99% 更严）
  - 评测集：ARC-Challenge/Easy、HellaSwag、PIQA、WinoGrande、OpenBookQA（HF 缓存已有全部数据集）
- 辅口径：WikiText-2/C4 PPL 相对增加（预期 2.x-bit 区间做不到 <0.5%，仅作参考记录）
- 若 0.5% 达不成 → ILP 提高 Budget 重搜（混合精度框架天然支持"精度换 bit"的滑动）

**D6：部署路径分两层：**
- **精度验证层（本任务范围）**：fake-quant BF16 checkpoint，transformers 直接加载 + vllm-ascend BF16 部署（与 FP 基准同 kernel，纯精度对照，无 kernel 开发风险）
- **真实压缩层（后续工作，超出本任务精度目标）**：PCDVQ 组需自定义 Ascend VQ kernel（位解包+双 gather+逆Hadamard）；GPTQ-INT 4bit 档可走 vllm-ascend 现有 W4A16（g128）路径。真实 bit 收益的 demo：4bit 档层数 × INT4 kernel + 2bit 档层数 ×（暂以 BF16 权重驻留+离线解包）——仅报告理论 bpp，不承诺加速

### 2.3 预期量化配置（基于 SLQ 论文敏感度结构的先验）

| 组类型 | 预期档位 | 依据 |
|---|---|---|
| full_attn K/V（16 层） | GPTQ-INT 6~8 bit | SLQ：K/V 最敏感 |
| linear_attn in_proj 组（48 层） | GPTQ-INT 4~6 bit | delta net 状态投影敏感度待实测 |
| o_proj / out_proj | GPTQ-INT 4 bit | 中等敏感 |
| MLP gate/up/down（64 层） | PCDVQ 2.0~2.25 bit | SLQ：MLP 最不敏感；PCDVQ 2bit 档质量远超 GPTQ-2bit |
| 首尾层 | 整体上调 1-2 档 | SLQ 实证 |

粗算（参数占比：MLP ≈ 3×5120×17408/2 ≈ 63%？实测再算）：MLP 全部 2.125 + 其余 4-6 bit → 加权平均 ≈ **2.9-3.3 bit/w**，对比 uniform-4bit (4.16 bpp 含开销) **降低 ~25%**，且低于 SLQ-TL 论文在 Qwen3.5-27B 的 3.30 bpp（我们的 0.5% 目标比其 1% 宽松一半，bpp 预期持平或略优）。

---

## 3. 可行性逐项分析

### Q1：PCDVQ 量化器能否嵌入 SLQ 框架？—— 可以，接口正交

SLQ 框架对量化算子的全部要求：① 给定层与位宽，产出 fake-quant 权重（`WeightStore` 落盘）；② 支持组级替换（GPUWeightCache 热切换）做 Shapley 前向；③ bpp 可解析计算。PCDVQ 满足全部三条（码本全局共享、逐层编码无状态、bpp=(a+b)/k+s 开销）。唯一接口差异：GPTQ 用 Hessian 序贯传播误差、PCDVQ 不用（贪心查表）。二者在同一层不同位宽档各自独立产出，互不干扰。**注意**：SLQ 的 `calibration_bitwidth=4` 序贯传播机制对 PCDVQ 档无意义（VQ 档不参与 Hessian 链），实现时对 PCDVQ 档跳过序贯传播、直接用 FP 输入编码。

### Q2：Shapley/线性敏感度对 VQ 档是否有效？—— 待 0.6B 实证，理论无障碍

Shapley 是度量驱动的黑盒（只看组切换前后 NLL/KL/EAR 变化），与量化器类型无关，天然有效。线性 α 估计假设"相对 Frobenius 误差 → 度量退化"近似线性（HIGGS 式），对 VQ 的误差结构（方向/幅值异构）近似质量未知——**0.6B 上做 Shapley vs 线性的相关性验证，若相关性差则 27B 降 P（如 P=8）+ 减档数做抽样 Shapley**。

### Q3：27B 计算与存储预算？—— 可行

| 项 | 预算 | 服务器现状 |
|---|---|---|
| 权重库磁盘 | 27B×2B×(4 INT 档+3 VQ 档+1 FP) ≈ 430GB | 4.8T 可用 ✓ |
| GPTQ 校准前向 | 1024×8192 tokens ≈ 8.4M，×2（收集+传播） | 8×910B3，单卡 64G；27B 需 TP2-4 或层分块（逐层处理时单层权重+Hessian ≈ 5120²×4B fp32 ≈ 100MB，无压力） |
| 敏感度（线性） | 15×M≈4300 次 512×2048 前向；27B 单次前向（TP4）≈ 1-2 min → **~5-7 天单路**；TP4 下吞吐×4 仍需 ~1.5 天 | 可并行 4 路空闲卡（2/4/5/6/7 选四）压缩到 ~10 小时 |
| PPL/lm-eval | 常规 | vllm-ascend BF16 部署后跑 |

**缓解**：线性估计的 15 噪声档可减到 8 档；敏感度校准集 512×2048 可减到 256×2048（SLQ 附录显示种子方差 ±0.0005 EAR，减半样本风险可控）。先在 0.6B 校准流程再放大。

### Q4：Qwen3.8 架构（hybrid linear attention + VLM）适配风险？—— 中等，有先例

- SLQ 官方已支持 Qwen3.5-27B（同族架构：VLM 嵌套 + hybrid），关键适配点（`language_model.layers.` 重映射、linear_attn 模块名）论文/代码有先例
- vllm-ascend 有 Qwen3.8-27B 官方教程（v0.30.0rc1）+ 专用镜像 `qwen3.8-a5`（已在拉取）→ 部署路径确定
- 0.6B（标准 Qwen3 dense）冒烟通过后，27B 的增量风险集中在 linear_attn 层的模块名/分组，属工程问题

### Q5：NPU 上跑 SLQ/PCDVQ 流水线？—— 无硬阻塞

- 全流水线 = HF transformers 前向 + 线性代数（Cholesky、matmul、gather），torch_npu 兼容；SLQ 代码的 CUDA 硬编码（empty_cache/NCCL env/flash-attn）逐处替换为 device 无关写法
- PCDVQ 编码是大规模 argmin/argmax 查表（[N×k] × [2^a×k] 余弦），27B 单层约 5120×17408/8 ≈ 11M 组 × 16384 码字——需分块 + 半精度矩阵乘，NPU matmul 可承担；FPS 码本构造一次性 CPU 即可
- RHT：非 2 幂维度（5120=2^10×5、17408=2^9×17）用 QuIP# 式分块 Hadamard（块大小 512/1024）

### Q6：精度目标 <0.5% 能否达成？—— 高置信度

- SLQ-TL 在 Qwen3.5-27B 上 99.1% 恢复 @3.30bpp（损失 0.9%）；我们目标 99.5% @ 预期 ~3.0-3.3bpp——**用 ~0-0.3bpp 的代价换 0.4pp 恢复率**，在 ILP 框架内就是调一下约束阈值，机制上直接支持
- PCDVQ 单独 2.125bit 时 LLaMA-2-7B zero-shot 相对损失 ~2.9%（62.24→60.44），说明纯 2bit 全模型不够；但混合方案中 PCDVQ 只承担最不敏感的 MLP 组（占参数大头），其局部退化由 Shapley/ILP 显式计入预算
- 剩余不确定性来自 VQ 误差在 Qwen3.8 权重分布上的实际表现（RHT 后近高斯假设对该模型成立与否）——0.6B 先验证 PPL 曲线

---

## 4. 与"不建议参考的已有工作"的关系

服务器上 `/opt/zh/Qwen3.8-27B-Quant`、`/opt/zh/SLQ` 等目录为既往量化工作（含 482GB Qwen3.5-27B 多位宽输出）。按任务要求**不复用其方法与结论**，仅复用：模型/tokenizer 文件、HF 数据集缓存（fineweb-edu/wikitext/c4/ARC/HellaSwag/PIQA/WinoGrande/OpenBookQA 均已缓存）、docker 镜像。所有代码在本仓库 `quant/` 下从零实现（PCDVQ 部分参照论文公式 + 修正，SLQ 部分参照论文 Algorithm 1-2 自行实现而非搬运官方仓库）。

## 5. 交付物与验收标准

| 阶段 | 交付件 | 验收标准 |
|---|---|---|
| 1. 环境验证 | 0.6B 单卡+双卡 vllm-ascend 冒烟日志 | 生成通顺文本；TP2 正常 |
| 2. 27B 部署 | qwen3.8-a5 容器内 TP≥2 serve + 问答验证 | BF16 基准推理正常；记录吞吐 |
| 3. PCDVQ 实现 | `quant/pcdvq/`（码本构造+编码器+单测） | 单测过（含 notebook 金标准码本数值）；0.6B 各档 PPL 曲线 |
| 4. 敏感度+ILP | `quant/slq/`（GPTQ 算子+Shapley/线性+ILP） | 0.6B 混合精度配置 + 恢复率 ≥99.5% @ <4bpp |
| 5. 27B 全量 | 混合精度 checkpoint + 评测报告 | **平均 bpp < 4.0（对照 uniform-4 的 4.156），zero-shot 平均相对损失 ≤0.5%** |
| 6. 报告 | experiment/ 下最终实验报告 | 含逐组 bit 分配表、误差分析 |

## 6. 风险清单

| # | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| R1 | PCDVQ 论文缺陷导致复现数字对不上 | 高 | 中 | 修正清单已列（§2.2 D2）；以 0.6B PPL 曲线为自身基准，不追论文绝对值 |
| R2 | 线性 α 对 VQ 档失效 | 中 | 高 | 0.6B Shapley vs 线性相关性验证；失效则 27B 用 P=8 抽样 Shapley（成本 ~7×M×|Q| 次前向，TP4 可承受） |
| R3 | RHT 高斯化对 Qwen3.8 权重不充分 | 低 | 中 | 检验变换后 kurtosis；不足则 per-channel scale 改 per-group |
| R4 | linear_attn 层量化敏感度异常（无先验） | 中 | 中 | 敏感度数据库天然覆盖；异常组自动分到高档 |
| R5 | 27B 敏感度计算时间超预算 | 中 | 中 | 减噪声档/减校准长度/TP4 并行（§Q3） |
| R6 | transformers 5.14 与 Qwen3.8 权重加载兼容问题 | 低 | 中 | 官方 transformers_version 5.8.0.dev0 ≤ 5.14；vllm-ascend 镜像内自带环境兜底 |
| R7 | 磁盘不足 | 低 | 低 | 430GB << 4.8T |
| R8 | VQ 真实部署 kernel 缺失 | 确定 | 低（本任务） | 明确划出任务范围（§2.2 D6）；fake-quant 验证精度 |

## 7. 结论

PCDVQ 与 SLQ 在方法学上**正交互补**（VQ 低位宽量化器 × 输出统计敏感度驱动的混合精度分配），接口上可直接组合；Qwen3.8-27B 的架构特殊性（hybrid linear attention + VLM）有 SLQ 官方对同族 Qwen3.5-27B 的支持先例与 vllm-ascend 官方部署教程背书；算力/存储预算在 8×910B3 上可行。**判定：可行，按 §5 阶段推进。**
