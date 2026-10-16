# Qwen3.8-27B 量化任务进展

任务定义见 [task.md](task.md)。

## 阶段规划与交付件

| 阶段 | 内容 | 交付件 | 状态 |
|------|------|--------|------|
| 0 | 任务规划 | progress.md 本文件 | ✅ 完成 |
| 1 | 调研：SLQ / PCDVQ 论文与代码 | experiment/feasibility.md（可行性分析报告） | ✅ 完成 |
| 2 | NPU 环境验证：Qwen3-0.6B 单卡+多卡冒烟部署 | 冒烟日志与环境结论（记录于本文件） | ✅ 完成 |
| 3 | Qwen3.8-27B 多卡部署验证（≥2卡，vllm-ascend） | 部署命令、推理验证结果 | ✅ 完成 |
| 4 | PCDVQ 量化器实现 + Qwen3-0.6B 冒烟 | quant/ 代码 + 0.6B PPL 对比 | ⬜ 进行中 |
| 5 | SLQ Shapley value 混合精度分配实现 | 代码 + 层级 bit 分配结果 | ⬜ |
| 6 | Qwen3.8-27B 全量量化实验 + 精度评估 | 27B 量化前后 PPL/下游任务对比（目标：精度损失 <0.5%，平均 bit 更低） | ⬜ |

## 进展日志

### 2026-10-16
- 创建任务规划，开始 SLQ/PCDVQ 论文与代码调研，同步检查 NPU 服务器环境。
- **调研完成**，产出 [feasibility.md](feasibility.md)：SLQ（统计无损量化：多bitwidth Shapley 敏感度 + ILP 全局混合精度分配 + SLQ-TL 单点标定）与 PCDVQ（极坐标解耦 VQ：E8 方向码本 + chi-LloydMax 幅值码本，~2.0-2.25 bit/w）接口正交，可组合为"PCDVQ 低位档 + GPTQ-INT 高位档 + SLQ 分配"方案；判定可行。PCDVQ 论文存在位算术矛盾/极坐标丢符号/E8池未定义等缺陷，复现需按报告 §2.2 D2 修正。
- **NPU 服务器环境**（121.37.53.41:10001，8×Ascend 910B3 64GB）：
  - 卡 3 被 vLLM 进程占用（58.7GB）；卡 0（Warning）/1（Alarm）有健康告警；可用卡：2/4/5/6/7
  - agent conda env：torch 2.10.0 + torch_npu 2.10.0、transformers 5.14.1、无 vllm_ascend
  - HF 缓存已有全部校准/评测数据集（fineweb-edu、wikitext、c4、ARC、HellaSwag、PIQA、WinoGrande、OpenBookQA、GSM8K）
  - 服务器上 Qwen3-0.6B/Qwen3.8-27B-FP8 缓存均不完整；已启动 hf-mirror 下载 Qwen3-0.6B 与 Qwen3.8-27B（18 分片）
  - docker 镜像 `quay.io/ascend/vllm-ascend:qwen3.8-a5`（docker.md 指定）registry manifest 查询失败但 `docker pull` 可用，已在后台拉取
  - 既往量化工作目录（/opt/zh/Qwen3.8-27B-Quant、/opt/zh/SLQ）按任务要求不复用方法，仅复用模型文件/数据集缓存
- **模型下载完成**：Qwen3-0.6B（1.5G）与 Qwen3.8-27B（52G，32 文件）均已通过 hf-mirror 完整下载至 /root/.cache/huggingface/hub/。
- **镜像选型变更（重要）**：`quay.io/ascend/vllm-ascend:qwen3.8-a5` 拉取成功后实测为 **A5 专用构建**（容器内 vllm_ascend `_build_info.__device_type__="A5"`，含 A5 专属通信/编译逻辑），在本机 Ascend 910B3（A2，soc_version 223）上启动即报 `device type does not match` 断言错误，不可用。改用宿主机已有的 **`quay.io/ascend/vllm-ascend:nightly-main`**（vllm 0.30.0 + torch_npu 2.10.0.post4 + transformers 5.14.1，A2 构建），已验证支持 `Qwen3_5ForConditionalGeneration`（Qwen3.8-27B 架构，需 transformers ≥5.8）。
- **docker 运行模式**（照抄本机正常运行的 vllm 容器，与 env/docker.md 模板不同，实测服务器无 /dev/ummu、/dev/uburma）：
  - 仅映射所需 davinci 卡（容器内自动重编号 0..N-1），**不设** ASCEND_RT_VISIBLE_DEVICES
  - 设备：davinciN + davinci_manager + hisi_hdc + devmm_svm
  - `--net=host --ipc=host --shm-size=64g`；挂载 driver、ascend_install.info、hccl_rootinfo.json、hixlep、/root/.cache
  - 全卡映射 + ASCEND_RT_VISIBLE_DEVICES 模式会触发 aclInit Resource_Busy(EL0005)，勿用
- **卡占用更新**：卡 3/4 被 qwen36-nova-kv-wsy 占用（卡 3 满载）；卡 6/7 被 yg-qwen36-w4a8c8 容器映射但空闲（HBM 基线）；卡 0/1/2 有他人进程。本任务冒烟默认用**卡 5**（最干净）与卡 6。
- **阶段 2 冒烟（Qwen3-0.6B，nightly-main 镜像）— ✅ 全部完成**：
  - **TP=1（卡 5，graph 模式）：PASS** — vllm 0.30.0 V1 引擎，权重加载 9.5s（1.14GB），torch.compile 48.9s，ACL graph 捕获正常，5 条 prompt 生成全部正常（320 tokens / 13.6s / 23.6 tok/s），总耗时约 4.5 分钟
  - **卡锁定规律（重要）**：被其他**正在运行的容器**映射并初始化过的卡（3/4 被 qwen36-nova-kv-wsy、6/7 被 yg-qwen36-w4a8c8），新容器内 `torch.npu.device_count()=0`（`Can't get ascend_hal device count`，枚举遇锁定卡整体失败）；宿主机直跑进程占用的卡（0/1/2）不影响容器访问。诊断矩阵：卡 2=1 可见、卡 2,5=2 可见、卡 5,7 / 4 / 4,5 = 0
  - **TP=2（卡 2,5）第一次（graph 模式）：FAIL** — graph capture 到 97% 后 hang（卡 2 与他人 100% 满载任务共卡）
  - **TP=2（卡 2,5）第二次（eager 模式）：FAIL** — 引擎初始化阶段 TP0（宿主机卡 2）算子 10 分钟未完成，`AclrtSynchronizeDeviceWithTimeout` 报 507034（vector core execution timed out），随后 EngineCore 初始化失败。根因：**卡 2 被他人任务 AICore 100% 饱和，算子饥饿触发 watchdog 超时，卡 2 不可用于计算**
  - **TP=2（卡 0,5，eager 模式）：PASS** — 双卡权重切分/HCCL/生成全部正常，模型加载 239.3s，5 条 prompt 生成 320 tokens / 34.7s（9.2 tok/s），内容全部正确连贯。卡 0 有芯片级 Warning 健康告警但实测计算正常
  - **结论**：vllm-ascend nightly-main 镜像在本机 910B3 上单卡/多卡推理功能均验证通过；共享环境下多卡部署需避开满载卡（0/5 为当前可用组合）；graph capture 在有算力竞争的卡上易 hang/超时，共享卡场景建议 eager 模式
- **阶段 3 启动：Qwen3.8-27B 多卡部署（卡 0,5，TP=2，eager）**：模型已就绪（52G，本地缓存），进行中
- **阶段 3 完成：Qwen3.8-27B TP=2 双卡部署 — ✅ PASS**（任务核心要求"至少两卡部署"达成）：
  - **部署配置**：镜像 nightly-main，宿主机卡 0,5（容器内 0,1），TP=2，eager，chat 模板（xhigh reasoning system prompt），`gpu_memory_utilization=0.85`，`max_model_len=4096`，bfloat16；部署命令 `run_smoke27b.sh 2 0,5 1`（nohup 后台 docker）
  - **架构识别**：`Resolved architecture: Qwen3_5ForConditionalGeneration`（VLM，需 transformers ≥5.8），18 个 safetensors 分片加载正常
  - **资源占用**：每卡权重 25.6118 GB（TP 切分，双卡共 ~51.2GB），peak activation 1.74 GiB，KV cache 可用 23.85 GiB / 324,701 tokens（4096 长度并发 79.27x），GPU mem 上限 51.81 GiB
  - **耗时**：引擎初始化（init engine）177.61s，模型总加载 348.0s
  - **推理验证**：5 条 chat prompt 生成 291 tokens / 77.8s（3.7 tok/s，推理模型 `<think>` 思维链 + eager + 共享卡环境所致，吞吐偏低但功能正确），全部回答正确连贯（"1 + 1 =" 正确答 2、北京为中国的首都等）
  - **结论**：vllm-ascend 对 Qwen3.8-27B（hybrid 64 层 linear/full attention）多卡部署功能完整可用，为后续量化实验（阶段 4/5/6）建立了部署与评估基线
- **阶段 4 启动：PCDVQ 量化器实现**（quant/pcdvq/，按 feasibility.md §2.2 D2 修正论文缺陷：极坐标丢符号、E8 池定义、位算术）
- **阶段 4 实施（1）：PCDVQ 量化器（同事实现 `experiment/PCDVQ-k3/`）已分析并在 NPU 验证通过**：
  - **实现审查**：4 模块（`rht.py` RHT 高斯正则化 / `polar.py` 极坐标解耦 / `codebooks.py` E8+chi-Lloyd-Max 码本 / `quantizer.py` 编解码）。正确修正论文+notebook 5 处缺陷：①RHT 补回 Hadamard 混合（notebook 只乘对角符号阵）；②非 2 幂维度分块 Hadamard；③PCD 末维符号用 atan2 保双射；④E8 多壳层枚举+单位化；⑤位算术严格 (a+b)/k。README 记录完整
  - **单元测试 9 passed**（16.8s）：含 E8 壳层计数 240/2400、Lloyd-Max 幅值码本与参考 notebook 金标准逐位一致（tol=1e-3 得 `[1.4241,...]`，tol=1e-8 收敛固定点 `[1.4396,...]`）
  - **demo 6 组验证数值一致**：RHT kurtosis 230.6→2.27（目标 3）、PCD 往返 7e-7、方向敏感度 9.54 vs 幅值 0.05（复现论文 Fig.1 动机）、DACC 消融 E8 贪心最均匀（最大两两余弦 0.75）
  - **NPU 验证（新增 `npv_verify.py`，卡 5，ASCEND_RT_VISIBLE_DEVICES=5）✅**：
    - NPU device_count=1（单卡可见）；码本构造 a=10 仅 2.4s（保持 CPU）
    - **CPU vs NPU 量化/反量化一致性 PASS**（重建最大绝对差 1.08e-7）
    - 分块方向量化 vs 全量索引一致率 1.0000；Qwen3-0.6B down_proj 规模（360448 向量）分块量化 0.02s、**吞吐 19.17M 向量/s**；方向 MSE 1.76 >> 幅值 MSE 0.05
  - **发现的性能瓶颈（下一阶段需处理）**：①a=14 档（BPW=2.0，feasibility 关键档）码本构造 CPU >5 分钟——根因 `e8_unit_directions` 用 8 维 meshgrid 全枚举（norm²≤8 需 207 万候选）+ `greedy_direction_codebook` 的 O(n²)（`sel_t=torch.tensor(sel)` 每步重建）；2^16 档需 norm²≤12 枚举 16.7M 点会内存爆炸；②方向码本必须 CPU 构造（NPU 上 2^a 次 `.item()` 同步 + 小 kernel 灾难）；③27B 单层 MLP 方向量化全量 `dirs@C_phi.t()`（11M×16384）会 OOM，必须分块（分块路径已验证可行）
  - **诚实结论（实测强化）**：PCDVQ 低 bit 纯 fake-quant 在 0.6B 上导致 PPL 灾难性崩溃（见下 4.3），根因是 2bit VQ 无误差补偿时 ~31% 权重误差逐层累积；论文端到端精度依赖误差补偿/微调 → 见下方"方案影响"
- **阶段 4 实施（2）：码本优化 + 0.6B 全模型 fake-quant + PPL 冒烟**：
  - **4.1 码本构造优化 ✅**：修掉 `greedy_direction_codebook` 的 O(n²)（`sel_t=torch.tensor(sel)` 每步重建 → 只置当前点 inf），a=14 码本构造从 >5 分钟降到 **~8s**（E8 枚举 1.0s + FPS 6.9s）；序列化缓存 .pt 复用（0.5MB，秒级加载）；pytest 9 仍通过
  - **4.2 全模型接入 ✅**：`model_quant.py` 加载 Qwen3-0.6B（28 层 hidden=1024 intermediate=3072），196 层 Linear 分块 PCDVQ 量化仅 14.9s（NPU 卡5）；`PCDVQ.quantize` 加分块方向匹配（避免大层 OOM）+ 去掉无用角度计算
  - **4.3 PPL 冒烟 ⚠️ 关键发现**：WikiText-2 基线 PPL **20.04**；PCDVQ a=14（BPW=2.0）纯 fake-quant：全模型 → PPL **360702**（崩溃）；仅 MLP（84 层）→ **1445**（+7108%）；仅 attention（112 层）→ **1836**（+9059%）。逐层相对 Frobenius 误差恒 ~0.314
  - **根因结论**：2bit VQ 权重重建误差 ~31% 接近理论下限（E8 方向码本 16384 在 8 维球面过稀疏）；无 QuIP# 式误差补偿、无 fine-tune 时该权重误差逐层累积导致输出崩溃。论文 2.125bit PPL 5.68 依赖误差补偿/微调（feasibility 1.2 已注"沿用 QuIP# block-wise + e2e 微调"）
  - **方案影响（需决策）**：fake-quant（无微调）精度验证路径下，PCDVQ 低 bit（2.0-2.25）不可用；混合精度方案低位档应改用 **GPTQ-INT 3bit**（有 Hessian 误差补偿，SLQ 原生支持）；PCDVQ 端到端精度留待"真实压缩层 + 微调"后续工作
- **阶段 4 待办（下一步）**：转向阶段 5 —— GPTQ-INT 算子（{3,4,6,8}bit g128 Hessian 补偿）+ Shapley 敏感度 + ILP 混合精度分配，0.6B 恢复率验证
