"""补丁 7：slq_quantize.py 支持 --model_class（27B 为 Qwen3_5ForConditionalGeneration）。"""
import pathlib

p = pathlib.Path("/opt/zh/slq-official/slq_quantize.py")
s = p.read_text()

OLD_ARG = '    parser.add_argument("--attn_implementation", type=str, default=None, choices=["eager", "sdpa", "flash_attention_2"])'
NEW_ARG = OLD_ARG + '\n    parser.add_argument("--model_class", type=str, default="AutoModelForCausalLM", help="VLM 需指定，如 Qwen3_5ForConditionalGeneration")'
assert OLD_ARG in s
s = s.replace(OLD_ARG, NEW_ARG)

OLD_LOAD = """    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,"""
NEW_LOAD = """    import transformers as _tf
    _cls = (_tf.AutoModelForCausalLM if args.model_class == "AutoModelForCausalLM"
            else getattr(_tf, args.model_class))
    model = _cls.from_pretrained(
        args.model_name_or_path,"""
assert OLD_LOAD in s
s = s.replace(OLD_LOAD, NEW_LOAD)
p.write_text(s)
print("patched slq_quantize --model_class")
