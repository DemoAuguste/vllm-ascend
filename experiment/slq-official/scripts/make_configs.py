"""Generate SLQ mixed-precision configs at fixed average-bitwidth budgets."""
import sys
from pathlib import Path

sys.path.insert(0, '/opt/zh/slq-official')

from slq_search_dl import (compute_avg_bitwidth, distribution_lines, extract_layer_groups,
                           load_sensitivity, predict_metrics_from_config, solve_allocation)
from src.sensitivity.database import ErrorDatabase, ShapleySensitivityData

LD = Path('/opt/zh/slq-official/data/slq_27b/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/4bit')
DB = LD / 'shapley_topk10_calib_27b_256k_2048_4x2048_8perms_bw2-4-6-8_grpb7761ae4_seed42.json'

error_db = ErrorDatabase.load(str(LD / 'error_db.json'))
sens = load_sensitivity(str(DB), ShapleySensitivityData)
groups = extract_layer_groups(sens)
bitwidths = [2, 4, 6, 8]
bw_map = {2: 2.156, 4: 4.156, 6: 6.156, 8: 8.156}

for budget in [3.4, 3.6, 3.8, 4.0]:
    cfg = solve_allocation(error_db, sens, budget, bitwidths, kl_weight=1.0, ear_weight=0.0,
                           bitwidth_map=bw_map, layer_groups=groups)
    avg = compute_avg_bitwidth(cfg, error_db.param_counts, bw_map)
    pred_kl = predict_metrics_from_config(cfg, error_db, sens)[0]
    out = Path('/opt/zh/slq-official') / f'config_tl27b_b{budget:.1f}.txt'
    with open(out, 'w') as f:
        f.write('# Mode: tl-budget\n')
        f.write('# Sensitivity method: shapley\n')
        f.write(f'# Available bitwidths: {bitwidths}\n')
        f.write(f'# Bitwidth map: {bw_map}\n')
        f.write(f'# Layer groups: {len(groups)}\n')
        f.write(f'# Target nominal bits: {budget:.2f}\n')
        f.write(f'# Average bitwidth: {avg:.4f}\n')
        f.write(f'# Predicted KL: {pred_kl:.6f}\n')
        f.write('# Bitwidth distribution:\n')
        for line in distribution_lines(cfg):
            f.write(f'#  {line}\n')
        f.write('#\n')
        f.writelines(f'{n}: {b}\n' for n, b in cfg.items())
    print(f'budget {budget}: avg {avg:.4f} bits, pred KL {pred_kl:.6f}, -> {out}', flush=True)
print('CONFIGS_DONE')
