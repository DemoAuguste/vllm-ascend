"""补丁 3：放开 --calibration-data 的 choices，允许传入本地 .pt 校准集。"""
import pathlib
import re

ROOT = pathlib.Path("/opt/zh/slq-official")
for rel in ["slq_build_database.py", "slq_search_dl.py", "slq_search_tl.py"]:
    p = ROOT / rel
    if not p.exists():
        continue
    s = p.read_text()
    # 去掉 choices=['c4', 'wikitext2', ...] 限制（保留参数）
    s2 = re.sub(r"(add_argument\(\s*['\"]--calibration-data['\"][^)]*?)choices=\[[^\]]*\]",
                r"\1", s, flags=re.S)
    if s2 != s:
        p.write_text(s2)
        print("patched", rel)
    else:
        print("no change", rel)
