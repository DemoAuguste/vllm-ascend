"""补丁 8：quantizer 支持断点续跑、显存清理，避免 27B 逐块累积 OOM。"""
import pathlib

p = pathlib.Path("/opt/zh/slq-official/src/quant/quantizer.py")
s = p.read_text()

# 1) import load_qparams
OLD_IMPORT = "from src.quant.qparams import save_qparams"
NEW_IMPORT = "from src.quant.qparams import save_qparams, load_qparams"
assert OLD_IMPORT in s
s = s.replace(OLD_IMPORT, NEW_IMPORT)

# 2) 替换每个 block 的处理逻辑
START = "            layers = select_layers(self.model, f\"{self.block_modules}.{block_id}.\", self.quantizable_modules)"
END = '            (torch.npu.empty_cache() if hasattr(torch, "npu") else torch.cuda.empty_cache())'
i0 = s.index(START)
i1 = s.index(END, i0) + len(END)

NEW = """            layers = select_layers(self.model, f"{self.block_modules}.{block_id}.", self.quantizable_modules)
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
                    f"  block {block_id} done, npu alloc {torch.npu.memory_allocated() / 2**30:.1f} GiB")"""

s = s[:i0] + NEW + s[i1:]
p.write_text(s)
print("patched quantizer (resume + cleanup)")
