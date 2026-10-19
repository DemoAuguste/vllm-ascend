"""Block-wise GPTQ quantization of a model at several bitwidths, saving fake-quant weights and qparams per layer."""
import os
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from src.quant import dist_utils
from src.quant.fast_obq import FastOBQ
from src.quant.model_utils import ForwardInterrupt, InputCollector, select_layers, to
from src.quant.qparams import save_qparams, load_qparams
from src.quant.quant_utils import QLinear


def _barrier():
    if dist_utils.is_dist_available_and_initialized():
        dist.barrier()


class Quantizer:

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        quantizable_modules: str,
        pre_block_modules: List[str],
        block_modules: str,
        save_dir: str,
        obq_kwargs: Dict[str, Any],
        device: Optional[torch.device] = None,
        cpu_offload_modules: bool = False,
        cpu_offload_activations: bool = False,
    ) -> None:
        self.model = model
        self.data_loader = data_loader
        self.quantizable_modules = quantizable_modules
        self.pre_block_modules = pre_block_modules
        self.block_modules = block_modules
        self.save_dir = save_dir
        self.obq_kwargs = obq_kwargs
        self.device = device or next(model.parameters()).device
        self.cpu_offload_modules = cpu_offload_modules
        self.cpu_offload_activations = cpu_offload_activations

    @torch.no_grad()
    def quantize(self, bitwidth_options: List[int], calibration_bitwidth: int):
        device = self.device
        blocks = self.model.get_submodule(self.block_modules)
        pre_blocks = [self.model.get_submodule(name).to(device) for name in self.pre_block_modules]
        use_cache = getattr(self.model.config, "use_cache", None)
        self.model.config.use_cache = False
        blocks[0] = InputCollector(blocks[0].to(device), cpu_offload=self.cpu_offload_activations)
        for inp_args, inp_kwargs in self.data_loader:
            try:
                self.model(*to(inp_args, device=device), **to(inp_kwargs, device=device))
            except ForwardInterrupt:
                pass
        input_args, input_kwargs = blocks[0].input_args, blocks[0].input_kwargs
        blocks[0] = blocks[0].module
        _barrier()
        if self.cpu_offload_modules:
            for module in pre_blocks:
                module.cpu()

        for block_id, block in enumerate(blocks):
            dist_utils.print_on_main(f"Processing {self.block_modules} {block_id + 1}/{len(blocks)}")
            block = block.to(device)
            layers = select_layers(self.model, f"{self.block_modules}.{block_id}.", self.quantizable_modules)
            names = list(layers)

            # 断点续跑：若该块所有层的 cal-bit 权重已存在，直接载入，跳过 GPTQ
            all_saved = all(
                os.path.exists(os.path.join(self.save_dir, n, f"{calibration_bitwidth}.pth"))
                for n in names
            )
            if all_saved:
                for n in names:
                    bundle = load_qparams(os.path.join(self.save_dir, n), calibration_bitwidth)
                    ql = QLinear(bundle["qweight"], bundle["scales"], bundle["zeros"]).to(device)
                    parent_name, child_name = n.rsplit(".", 1)
                    setattr(self.model.get_submodule(parent_name), child_name, ql)
                dist_utils.print_on_main(f"  resumed block {block_id} ({len(names)} layers, no GPTQ)")
            else:
                handles = {name: FastOBQ(layer, bitwidth_options, **self.obq_kwargs)
                           for name, layer in layers.items()}
                hooks = [
                    layer.register_forward_hook(lambda _, inp, out, h=handles[name]: h.update(inp[0]))
                    for name, layer in layers.items()
                ]
                for inp_args, inp_kwargs in zip(input_args, input_kwargs):
                    block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
                for hook in hooks:
                    hook.remove()
                del hooks
                _barrier()

                for name, handle in handles.items():
                    self._quantize_layer(name, handle, calibration_bitwidth)
                # 释放旧层（已被 QLinear 替换）与 Hessian 引用
                for handle in handles.values():
                    handle.layer.cpu()
                    handle.H = None
                handles.clear()
                del handles, layers
                torch.npu.empty_cache() if hasattr(torch, "npu") else torch.cuda.empty_cache()

            # later blocks are calibrated on inputs from the quantized model
            for _i, (inp_args, inp_kwargs) in enumerate(zip(input_args, input_kwargs)):
                out = block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
                out = out[0] if isinstance(out, (tuple, list)) else out
                if self.cpu_offload_activations:
                    out = out.cpu()
                if len(inp_args) > 0:
                    inp_args[0].data = out
                elif "hidden_states" in inp_kwargs:
                    inp_kwargs["hidden_states"] = out
                else:
                    raise ValueError("Unsupported block input format.")
                if "npu" in str(device) and _i % 8 == 7:
                    torch.npu.empty_cache()
            if self.cpu_offload_modules:
                block.cpu()
            torch.npu.empty_cache() if hasattr(torch, "npu") else torch.cuda.empty_cache()
            if hasattr(torch, "npu"):
                dist_utils.print_on_main(
                    f"  block {block_id} done, npu alloc {torch.npu.memory_allocated() / 2**30:.1f} GiB")

        if use_cache is not None:
            self.model.config.use_cache = use_cache

    def _quantize_layer(self, name: str, handle: FastOBQ, calibration_bitwidth: int):
        layer_save_dir = os.path.join(self.save_dir, name)
        for bits, (qweight, scales, zeros) in handle.quantize().items():
            qlayer = QLinear(qweight, scales, zeros, bias=handle.layer.bias)
            # every rank holds the same result; only rank 0 writes it
            if dist_utils.is_main():
                os.makedirs(layer_save_dir, exist_ok=True)
                torch.save(qlayer.get_weight().cpu(), os.path.join(layer_save_dir, f"{bits}.pth"))
                save_qparams(layer_save_dir, bits, qweight, scales, zeros, handle)
            if bits == calibration_bitwidth:
                parent_name, child_name = name.rsplit(".", 1)
                setattr(self.model.get_submodule(parent_name), child_name, qlayer)
