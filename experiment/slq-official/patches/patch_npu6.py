"""补丁 6：Shapley 成本表去噪（负值钳到 0 + 位宽单调）。小校准集下 Shapley 估计会有负边际。"""
import pathlib

p = pathlib.Path("/opt/zh/slq-official/slq_search_dl.py")
s = p.read_text()

OLD = """    if isinstance(sens, ShapleySensitivityData):
        return getattr(sens, f'{metric}_shapley_per_level') or {}"""
NEW = """    if isinstance(sens, ShapleySensitivityData):
        table = getattr(sens, f'{metric}_shapley_per_level') or {}
        # 小校准集下 Shapley 边际可能为负（噪声）；成本物理上非负，且应随位宽单调不增。
        out = {}
        for _n, _costs in table.items():
            _bits = sorted(_costs)
            _vals = {b: max(0.0, float(_costs[b])) for b in _bits}
            for _i in range(len(_bits) - 2, -1, -1):
                _vals[_bits[_i]] = max(_vals[_bits[_i]], _vals[_bits[_i + 1]])
            out[_n] = _vals
        return out"""
assert OLD in s
p.write_text(s.replace(OLD, NEW))
print("patched layer_costs sanitization")
