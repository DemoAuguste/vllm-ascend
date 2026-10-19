"""One-shot parallel GPTQ that saves per-layer weights at every bitwidth in --bitwidth_options."""
import argparse
import os
import time

import torch
import torch_npu  # noqa: F401  (NPU backend)
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.loading import MASKED_DATASETS, get_data, get_fineweb_edu_gptq
from src.quant import dist_utils
from src.quant.quantizer import Quantizer


def parse_args():
    parser = argparse.ArgumentParser(description="One-shot quantization with parallel GPTQ.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--quantizable_modules", type=str, required=True, help="Regex for modules to quantize.")
    parser.add_argument("--pre_block_modules", nargs="+", type=str, required=True, help="Modules before the blocks.")
    parser.add_argument("--block_modules", type=str, required=True, help="Name of the transformer block list.")
    parser.add_argument("--calibration_data", type=str, required=True, help="Dataset name or path to a .pt file.")
    parser.add_argument("--calibration_tokens", default=2**23, type=int)
    parser.add_argument("--calibration_sequence_length", default=None, type=int)
    parser.add_argument("--bitwidth_options", nargs="+", type=int, required=True)
    parser.add_argument(
        "--calibration_bitwidth", type=int, required=True, help="Bitwidth whose outputs calibrate later blocks."
    )
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--sym", action="store_true", help="Symmetric instead of asymmetric INT grid.")
    parser.add_argument("--perchannel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rel_damp", type=float, default=1e-2)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default=None, choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--model_class", type=str, default="AutoModelForCausalLM", help="VLM 需指定，如 Qwen3_5ForConditionalGeneration")
    parser.add_argument("--cpu_offload_modules", action="store_true")
    parser.add_argument("--cpu_offload_activations", action="store_true")
    parser.add_argument("--save_dir", type=str, required=True, help="Writes to <save_dir>/<model>/<cal_bits>bit.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.calibration_bitwidth not in args.bitwidth_options:
        raise ValueError(f"Calibration bitwidth {args.calibration_bitwidth} is not in bitwidth_options.")
    torch.manual_seed(args.seed)
    # torchrun sets the env:// rendezvous variables; plain python runs single-process.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="hccl", device_id=torch.device(f"npu:{os.environ['LOCAL_RANK']}"))
    world_size, rank = dist_utils.get_world_size(), dist_utils.get_rank()
    device = f"npu:{rank}"
    torch.npu.set_device(device)
    import transformers as _tf
    _cls = (_tf.AutoModelForCausalLM if args.model_class == "AutoModelForCausalLM"
            else getattr(_tf, args.model_class))
    model = _cls.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        dtype=args.dtype if args.dtype == "auto" else getattr(torch, args.dtype),
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        attn_implementation=args.attn_implementation,
    )
    if not args.cpu_offload_modules:
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    seq_len = args.calibration_sequence_length or model.config.max_position_embeddings
    if args.calibration_data == "fineweb_edu":
        data = get_fineweb_edu_gptq(args.calibration_tokens, seq_len, tokenizer, seed=args.seed)
    else:
        data = get_data(args.calibration_data, args.calibration_tokens, seq_len, tokenizer, train=True, seed=args.seed)
    if args.calibration_data in MASKED_DATASETS:
        data = data[0]
    # Each rank calibrates on its own slice of the data.
    per_rank = len(data) // world_size
    data = [([], {"input_ids": ids}) for ids in data[rank * per_rank : (rank + 1) * per_rank]]
    save_dir = os.path.join(args.save_dir, args.model_name_or_path.split("/")[-1], f"{args.calibration_bitwidth}bit")
    quantizer = Quantizer(
        model,
        data,
        quantizable_modules=args.quantizable_modules,
        pre_block_modules=args.pre_block_modules,
        block_modules=args.block_modules,
        save_dir=save_dir,
        obq_kwargs=dict(group_size=args.group_size, sym=args.sym, perchannel=args.perchannel, rel_damp=args.rel_damp),
        device=device,
        cpu_offload_modules=args.cpu_offload_modules,
        cpu_offload_activations=args.cpu_offload_activations,
    )
    t0 = time.perf_counter()
    quantizer.quantize(args.bitwidth_options, args.calibration_bitwidth)
    dist_utils.print_on_main(f"Quantization took {time.perf_counter() - t0:.0f} s.")
    if dist_utils.is_dist_available_and_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
