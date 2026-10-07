"""Mutation check: introduce deliberate errors into the formulas and confirm the test suite catches each one.

Run from anywhere:  python scripts/mutation_check.py
Every line of output should start with CAUGHT. Files are restored after each mutation.
"""
import pathlib
import subprocess
import sys
root = pathlib.Path(__file__).resolve().parents[1]
M = [
 ('src/prodanalytics/experiment.py', 'se_log_rr = np.sqrt((1 - p_t) / x_t + (1 - p_c) / x_c)', 'se_log_rr = np.sqrt((1 - p_t) / n_t + (1 - p_c) / n_c)'),
 ('src/prodanalytics/experiment.py', 'expected_loss_treatment=float(np.maximum(s_c - s_t, 0).mean()),', 'expected_loss_treatment=float(np.maximum(s_t - s_c, 0).mean()),'),
 ('src/prodanalytics/experiment.py', 'expected_loss_control=float(np.maximum(s_t - s_c, 0).mean()),', 'expected_loss_control=float(np.maximum(s_c - s_t, 0).mean()),'),
 ('src/prodanalytics/experiment.py', 'var = p_baseline * (1 - p_baseline) + p_alt * (1 - p_alt)', 'var = 2 * p_baseline * (1 - p_baseline)'),
 ('src/prodanalytics/experiment.py', 'draws[i] = stat(t) - stat(c)', 'draws[i] = stat(c) - stat(t)'),
 ('src/prodanalytics/experiment.py', 'return float(stat(treatment) - stat(control)), (float(lo), float(hi)), draws', 'return float(stat(control) - stat(treatment)), (float(lo), float(hi)), draws'),
 ('src/prodanalytics/experiment.py', 'z = diff / se_pooled', 'z = diff / se_unpooled'),
 ('src/prodanalytics/experiment.py', 'running_max = max(running_max, (m - rank) * p[idx])', 'running_max = max(running_max, m * p[idx])'),
 ('src/prodanalytics/experiment.py', 'return float((z_a + z_b) * np.sqrt(2 * p_baseline * (1 - p_baseline) / n_per_group))', 'return float((z_a + z_b) * np.sqrt(p_baseline * (1 - p_baseline) / n_per_group))'),
 ('src/prodanalytics/clv.py', 'ratio = (a / (b + x_safe - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x)', 'ratio = (a / (b + x_safe)) * ((alpha + T) / (alpha + t_x)) ** (r + x)'),
 ('src/prodanalytics/clv.py', 'return (a + b - 1) / (a - 1) * (1 - (alpha / (alpha + t)) ** r * hyp)', 'return (a + b - 1) / (a - 1) * (1 - (alpha / (alpha + t)) ** (r + 1) * hyp)'),
 ('src/prodanalytics/clv.py', 'clv += (cum - prev) * spend * margin / (1 + monthly_discount) ** m', 'clv += cum * spend * margin / (1 + monthly_discount) ** m'),
 ('src/prodanalytics/clv.py', 'clv += (cum - prev) * spend * margin / (1 + monthly_discount) ** m', 'clv += (cum - prev) * spend * margin / (1 + monthly_discount) ** (m * 12)'),
 ('src/prodanalytics/clv.py', 'term2 = betaln(a + 1, b + x_safe - 1)', 'term2 = betaln(a, b + x_safe - 1)'),
 ('src/prodanalytics/clv.py', 'weight = (q - 1) / (p * x + q - 1)', 'weight = (q - 1) / (p * x + q)'),
 ('src/prodanalytics/clv.py', 'daily["is_repeat"] = daily[date_col] > first', 'daily["is_repeat"] = daily[date_col] >= first'),
 ('src/prodanalytics/peeking.py', 'significant.any(axis=1).mean()', 'significant[:, -1].mean()'),
]
missed = 0
for f, old, new in M:
    p = root / f
    orig = p.read_text()
    assert old in orig, old
    p.write_text(orig.replace(old, new, 1))
    try:
        r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x'], cwd=root, capture_output=True, text=True)
    finally:
        p.write_text(orig)   # always restore the original source, even if interrupted
    missed += r.returncode == 0
    print('CAUGHT' if r.returncode != 0 else 'MISSED', '|', f, '|', new[:70])
r = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=root, capture_output=True, text=True)
print(r.stdout.strip().splitlines()[-1])
sys.exit(1 if missed else 0)
