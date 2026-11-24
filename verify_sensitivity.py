"""CE 7405 Step 6 independent one-at-a-time sensitivity checks.

Run:
  conda run --no-capture-output -n ce7405 python verify_sensitivity.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from multistory_frame import generate_building
from semirigid_connections import analyze_connected_building, all_beam_ends, connection
from sensitivity_mapping import map_connection_sensitivity, sensitivity_table_rows
from sensitivity_plot import sensitivity_figure


def check(condition, message):
    assert condition, message
    print('PASS:', message)


m = generate_building()
keys = all_beam_ends(m)
check(len(keys) == 24, '4 story x 3 bay structure has 24 independently editable beam ends')

baseline = {(1, 1, 'left'): connection('pinned'),
            (2, 2, 'right'): connection('semi', alpha=3.0),
            (4, 3, 'left'): connection('semi', alpha=8.0)}
original = dict(baseline)
r0 = analyze_connected_building(m, baseline)
study = map_connection_sensitivity(m, baseline, connection('rigid'), baseline_result=r0)
check(baseline == original, 'mapping never changes input connection settings')
check(len(study.records) == 24, 'all 24 beam ends were visited')
check(len(study.successful) == 3, 'trial-to-rigid skips the 21 already-rigid ends')
check(len([r for r in study.records if r.status == 'unchanged']) == 21,
      'unchanged configurations are not assigned false sensitivity ranks')

for record in study.successful:
    modified = dict(baseline)
    modified.pop(record.key)
    r1 = analyze_connected_building(m, modified)
    D = r0.max_story_drift_ratio
    expected = (D-r1.max_story_drift_ratio)
    assert np.isclose(record.drift_reduction_pp, 100*expected, atol=1e-12)
    assert np.isclose(record.improvement_percent, 100*expected/D, atol=1e-10)
check(True, 'each computed influence matches independently re-solved trial')
check(all(study.successful[i].drift_reduction_pp >= study.successful[i+1].drift_reduction_pp
          for i in range(len(study.successful)-1)), 'ranking ordered by actual drift reduction')

study2 = map_connection_sensitivity(m, baseline, connection('semi', alpha=5.0),
                                    eligible_keys=[keys[0], keys[0], keys[1]])
check(len(study2.records) == 2, 'floor/end selection and duplicate candidate removal')
check(len(study2.successful) == 2, 'valid semi-rigid trials are solved independently')

zero = generate_building(n_stories=1, n_bays=1,
                         story_lateral_loads_kN=[0.0], beam_udl_kN_per_m=0.0)
zero_study = map_connection_sensitivity(zero, {}, connection('semi', alpha=5.0))
check(not zero_study.relative_score_available, 'zero baseline drift uses absolute effects, not division by zero')
check(all(r.improvement_percent is None for r in zero_study.successful),
      'no fictitious relative percentages for zero-drift case')

fig = sensitivity_figure(m, study)
fig.savefig("sensitivity_map_example.png", dpi=180)
plt.close(fig)
rows = sensitivity_table_rows(study)
check(len(rows) == 24 and sum(bool(x['Rank']) for x in rows) == 3,
      'ranking CSV distinguishes tested and unchanged beam ends')

all_rigid = map_connection_sensitivity(m, {}, connection('semi', alpha=5.0))
check(len(all_rigid.successful) == 24, 'default all-rigid frame maps all ends to finite-alpha alternatives')
print(f'BASELINE maximum drift = {100*r0.max_story_drift_ratio:.6f}%')
for trial in study.successful:
    print(f'  {trial.key}: relative drift reduction {trial.improvement_percent:+.5f}%; '
          f'candidate drift {100*trial.trial_drift:.6f}%')
print('ALL STEP 6 SENSITIVITY CHECKS PASSED')
