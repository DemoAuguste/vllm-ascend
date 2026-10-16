# Qwen3.8-27B模型量化任务

## 背景

实现Qwen3.8-27B的量化，在精度损失小于0.5%的情况下，实现更低的bit量化。

## 任务描述

1. 基于vllm-ascend实现Qwen3.8-27B多卡（至少两卡）部署，验证npu环境可用；
2. 分析参考论文，设计合适的模型量化方法，要求在精度损失小于0.5%的情况下实现更低的bit量化；
3. 先用Qwen3-0.6B模型进行小规模冒烟验证（单卡，多卡），验证通过后再进行Qwen3.8-27B全量实验。

## 量化方法设计论文参考

论文实现参考：
   - SLQ：https://arxiv.org/pdf/2605.02404，官方代码实现：https://github.com/IST-DASLab/SLQ
   - PCDVQ：https://arxiv.org/pdf/2506.05432，非官方参考实现（需要仔细分析是否正确）：https://github.com/danlee65071/pcdvq/blob/main/notebooks/pcdvq.ipynb

建议优先复现PCDVQ，这个方法能够实现更低bit的量化，结合SLQ的shapley value，实现模型的混合精度量化，使用SLQ-TL实现精度无损。
需要详细分析PCDVQ + SLQ可行性，先给出可行性分析。



## npu服务器环境信息

- npu环境说明：env\env.md
- docker启动命令说明：env\docker.md


## 模型

Qwen3.8-27B模型：https://huggingface.co/Qwen/Qwen3.8-27B
vllm-ascend部署Qwen3.8-27B模型说明：https://docs.vllm.ai/projects/ascend/en/v0.30.0rc1/tutorials/models/Qwen3.8-27B.html

## 其他说明

1. 先进行任务规划，将任务分解成多个阶段，每个阶段要有可验收的交付件；
2. 及时任务进展更新到experiment\progress.md；
3. 使用https://hf-mirror.com/下载必要的模型。npu环境上已有一些模型文件，你可以尝试复用；
4. npu环境上已有一些量化工作，不建议参考。你可以复用npu环境上的模型文件，但是当前没有Qwen3.8-27B，只有Qwen3.5-27B（不要混淆！！！）；
5. 如果存在多次尝试无法解决的任务，及时向用户报告，用户会帮你解决。