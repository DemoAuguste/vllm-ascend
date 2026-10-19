"""补丁 4：把 calibration_data 在输出文件名里改为 stem，避免路径分隔符。"""
import pathlib

p = pathlib.Path("/opt/zh/slq-official/slq_build_database.py")
s = p.read_text()
old = 'tag = f"topk{args.top_k}_{args.calibration_data}_{args.calibration_samples}x{args.calibration_seq_length}"'
new = ('tag = (f"topk{args.top_k}_{Path(args.calibration_data).stem}_"\n'
       '           f"{args.calibration_samples}x{args.calibration_seq_length}")')
assert old in s, "tag line not found"
p.write_text(s.replace(old, new))
print("patched tag")
