import os
import time


def main():
    import torch
    import torch_npu  # noqa: F401
    from vllm import LLM, SamplingParams

    tp = int(os.environ.get("SMOKE_TP", "1"))
    tag = os.environ.get("SMOKE_TAG", f"tp{tp}")
    eager = os.environ.get("SMOKE_EAGER", "0") == "1"
    model = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3-0.6B")
    max_len = int(os.environ.get("SMOKE_MAX_LEN", "4096"))
    gpu_mem = float(os.environ.get("SMOKE_GPU_MEM", "0.7"))
    max_tokens = int(os.environ.get("SMOKE_MAX_TOKENS", "64"))
    chat = os.environ.get("SMOKE_CHAT", "0") == "1"
    print(
        f"[smoke] model={model} TP={tp}, npu count={torch.npu.device_count()}, "
        f"eager={eager}, chat={chat}",
        flush=True,
    )

    t0 = time.time()
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        max_model_len=max_len,
        gpu_memory_utilization=gpu_mem,
        trust_remote_code=True,
        enforce_eager=eager,
    )
    print(f"[smoke] model loaded in {time.time() - t0:.1f}s", flush=True)

    prompts = [
        "The capital of China is",
        "Water boils at a temperature of",
        "Large language models are",
        "1 + 1 =",
        "Write a haiku about autumn:",
    ]
    sp = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=max_tokens)
    t0 = time.time()
    if chat:
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        outs = llm.chat(msgs, sp)
    else:
        outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[smoke] generation done: {ntok} tokens in {dt:.1f}s ({ntok / dt:.1f} tok/s)", flush=True)

    ok = True
    for o in outs:
        txt = o.outputs[0].text.strip().replace("\n", " ")
        print(f"[smoke] PROMPT: {o.prompt!r:40s} -> {txt[:120]!r}", flush=True)
        if len(txt) < 5:
            ok = False
    print(f"[smoke:{tag}] RESULT: {'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
