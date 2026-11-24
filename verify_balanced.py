"""Regression tests for Step 9: optional rigid and target-first semi-rigid search."""
import numpy as np
from multistory_frame import generate_building, equilibrium_residuals
from semirigid_connections import connection, all_beam_ends, analyze_connected_building
from greedy_optimization import optimize_connections


def check(ok, name):
    assert ok, name
    print('PASS:', name)

m = generate_building(n_stories=1, n_bays=1, story_lateral_loads_kN=[20],
                      beam_udl_kN_per_m=2)
ends = all_beam_ends(m)
start = {key: connection('pinned') for key in ends}
original = dict(start)

# Default user-friendly mode: finite, explicit alpha values only.
r = optimize_connections(
    m, start, target_drift_ratio=0.0, modification_budget=2,
    allow_semi=True, allow_rigid=False, candidate_alphas=(2, 5, 10, 20),
)
check(bool(r.steps), 'Semi-rigid-only search produces recommendations')
check(all(s.after.kind == 'semi' for s in r.steps),
      'Semi-rigid-only search never suggests rigid')
check(all(s.after.alpha <= 20 for s in r.steps),
      'Finite stiffness cap is respected')
check(r.final_drift < r.baseline_drift, 'Semi-rigid alternatives reduce drift')
check(start == original, 'Manual connection settings are unchanged')
check(np.max(np.abs(equilibrium_residuals(m, r.final_result))) < 1e-3,
      'Optimized configuration satisfies equilibrium')

# In mixed mode, use the weakest trial reaching the target rather than
# automatically choosing the strongest (or rigid) alternative.
meeting = optimize_connections(
    m, start, target_drift_ratio=0.0010, modification_budget=2,
    allow_semi=True, allow_rigid=True, candidate_alphas=(2, 5, 10, 20),
)
check(meeting.target_met, 'Target-first search meets attainable drift target')
check(len(meeting.steps) == 1 and meeting.steps[0].after.kind == 'semi'
      and meeting.steps[0].after.alpha == 2,
      'Mixed search prefers α=2 over stronger or rigid options when adequate')

# Original rigid-only option remains an explicit choice.
rigid = optimize_connections(
    m, start, target_drift_ratio=0.0, modification_budget=1,
    allow_semi=False, allow_rigid=True,
)
check(len(rigid.steps) == 1 and rigid.steps[0].after.kind == 'rigid',
      'Original rigid-only optimization still works')

# Input validation.
try:
    optimize_connections(m, start, allow_semi=True, allow_rigid=False,
                         candidate_alphas=[5, -1])
except ValueError:
    check(True, 'Invalid stiffness candidate is rejected')
else:
    raise AssertionError('Negative candidate alpha was accepted')

print('ALL STEP 9 BALANCED OPTIMIZATION CHECKS PASSED')
