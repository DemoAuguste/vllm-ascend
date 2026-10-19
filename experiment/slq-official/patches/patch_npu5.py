"""补丁 5：OR-Tools 在本平台 CreateSolver 段错误 → 用 scipy.optimize.milp（HiGHS）求解同一 MILP。"""
import pathlib

p = pathlib.Path("/opt/zh/slq-official/slq_search_dl.py")
s = p.read_text()

START = "    solver = pywraplp.Solver.CreateSolver('SCIP')"
END = """    return {n: b for i, g in enumerate(groups) for j, b in enumerate(available_bw)
            if x[i][j].solution_value() > 0.5 for n in g.layer_names}"""

i0 = s.index(START)
i1 = s.index(END) + len(END)

NEW = """    params = [sum(error_db.param_counts.get(n, 1) for n in g.layer_names) for g in groups]
    n_g, n_b = len(groups), len(available_bw)
    n_var = n_g * n_b
    import numpy as _np
    from scipy.optimize import milp, LinearConstraint, Bounds

    c = _np.zeros(n_var)
    for i, g in enumerate(groups):
        for j, b in enumerate(available_bw):
            c[i * n_b + j] = cost(g, b)

    A_eq = _np.zeros((n_g, n_var))
    for i in range(n_g):
        A_eq[i, i * n_b:(i + 1) * n_b] = 1.0

    A_ub = _np.zeros((1, n_var))
    for i in range(n_g):
        for j, b in enumerate(available_bw):
            A_ub[0, i * n_b + j] = get_actual_bitwidth(b, bitwidth_map) * params[i]

    A = _np.vstack([A_eq, A_ub])
    lb = _np.concatenate([_np.ones(n_g), [-_np.inf]])
    ub = _np.concatenate([_np.ones(n_g), [target_bitwidth * sum(params)]])
    res = milp(c=c, constraints=LinearConstraint(A, lb, ub),
               integrality=_np.ones(n_var), bounds=Bounds(0, 1),
               options={'time_limit': SOLVER_TIME_LIMIT_MS / 1000.0})
    if res.x is None:
        raise RuntimeError(f"ILP found no solution at average bitwidth {target_bitwidth:.3f} ({res.message})")
    if res.status != 0:
        logger.warning(f"ILP status {res.status}: {res.message}; using best feasible solution")
    x_arr = res.x.reshape(n_g, n_b)
    return {n: available_bw[j] for i, g in enumerate(groups) for j in range(n_b)
            if x_arr[i, j] > 0.5 for n in g.layer_names}"""

s = s[:i0] + NEW + s[i1:]
p.write_text(s)
print("patched solve_allocation -> scipy.milp")
