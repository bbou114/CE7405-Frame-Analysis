"""CE 7405 Step 7: regression and exhaustive small-frame greedy checks.

Run: conda run --no-capture-output -n ce7405 python verify_greedy.py
"""
from itertools import product
import numpy as np

from multistory_frame import generate_building, equilibrium_residuals
from semirigid_connections import all_beam_ends, analyze_connected_building, connection
from greedy_optimization import optimize_connections, optimization_table_rows


def check(condition, label):
    assert condition, label
    print('PASS:', label)


# 1. A 4-story/3-bay model with a few initially flexible beam ends.
m = generate_building()
start = {
    (1, 1, 'left'): connection('pinned'),
    (2, 2, 'right'): connection('semi', alpha=3),
    (3, 1, 'right'): connection('semi', alpha=8),
    (4, 3, 'left'): connection('pinned'),
}
copy_start = dict(start)
base = analyze_connected_building(m, start)
search = optimize_connections(
    m, start, target_drift_ratio=0.0, modification_budget=3,
    allow_semi=True, upgrade_alpha=15, baseline_result=base,
)
check(start == copy_start, 'Greedy optimizer never mutates supplied manual settings')
check(len(search.steps) <= 3, 'At most three distinct beam ends changed')
check(len({step.key for step in search.steps}) == len(search.steps),
      'One change per beam end: modifications, not trials, use budget')
check(all(step.drift_after_ratio < step.drift_before_ratio
          for step in search.steps), 'Every accepted step strictly improves maximum drift')
check(all(step.before.kind != 'rigid' for step in search.steps),
      'The optimizer never downgrades originally rigid ends')
check(search.final_drift <= search.baseline_drift + 1e-12,
      'Recommended changes never increase maximum drift')
recalc = analyze_connected_building(m, search.recommended_settings)
check(np.isclose(recalc.max_story_drift_ratio, search.final_drift,
                 rtol=1e-9, atol=1e-12),
      'Final reported drift matches independently re-solved configuration')
check(np.max(np.abs(equilibrium_residuals(m, search.final_result))) < 1e-3,
      'Optimized frame satisfies force and moment equilibrium')
check(len(optimization_table_rows(search)) == len(search.steps),
      'Export sequence has exactly one row per accepted upgrade')

# 2. Optimization should be a true optional feature, including all-rigid case.
rigid = optimize_connections(m, {}, target_drift_ratio=0.0, modification_budget=4)
check(rigid.status == 'no_eligible_upgrades' and not rigid.steps,
      'All-rigid frame produces no fictitious stronger connection candidates')
already = optimize_connections(m, start, target_drift_ratio=base.max_story_drift_ratio,
                               modification_budget=4, baseline_result=base)
check(already.status == 'already_satisfied' and already.attempted_solves == 0,
      'Already-satisfied target stops immediately')
no_budget = optimize_connections(m, start, target_drift_ratio=0.0,
                                 modification_budget=0, baseline_result=base)
check(no_budget.status == 'budget_exhausted' and no_budget.attempted_solves == 0,
      'Zero change budget yields no upgrades')
restricted = optimize_connections(
    m, start, target_drift_ratio=0.0, modification_budget=2,
    eligible_keys=[(1, 1, 'left')], baseline_result=base,
)
check(all(step.key == (1, 1, 'left') for step in restricted.steps),
      'Eligible beam-end subset enforced')

# 3. Small-frame exhaustive enumeration as an independent B5 reference.
small = generate_building(n_stories=1, n_bays=2,
                          story_lateral_loads_kN=[18], beam_udl_kN_per_m=3)
ends = all_beam_ends(small)
pinned = {k: connection('pinned') for k in ends}
small_base = analyze_connected_building(small, pinned)
# Drift target is intentionally unreachable (0) so a two-change budget is used.
greedy = optimize_connections(small, pinned, target_drift_ratio=0.0,
                              modification_budget=2, allow_semi=True,
                              upgrade_alpha=12)
options = [None, connection('semi', alpha=12), connection('rigid')]
best_d = np.inf
best_combo = None
valid_count = 0
for selections in product(options, repeat=len(ends)):
    if sum(v is not None for v in selections) > 2:
        continue
    case = dict(pinned)
    for key, state in zip(ends, selections):
        if state is not None:
            if state.kind == 'rigid':
                case.pop(key)
            else:
                case[key] = state
    try:
        candidate = analyze_connected_building(small, case)
    except (ValueError, np.linalg.LinAlgError):
        continue
    valid_count += 1
    d = candidate.max_story_drift_ratio
    if d < best_d:
        best_d, best_combo = d, selections
# Verify the first greedy choice is the best single legal upgrade.
one_upgrade = optimize_connections(small, pinned, target_drift_ratio=0.0,
                                   modification_budget=1, allow_semi=True,
                                   upgrade_alpha=12)
manual_one_step_best = small_base.max_story_drift_ratio
for key in ends:
    for choice in (connection('semi', alpha=12), connection('rigid')):
        variant = dict(pinned)
        if choice.kind == 'rigid':
            variant.pop(key)
        else:
            variant[key] = choice
        manual_one_step_best = min(manual_one_step_best,
                                   analyze_connected_building(small, variant).max_story_drift_ratio)
check(np.isclose(one_upgrade.final_drift, manual_one_step_best, atol=1e-12),
      'Greedy first step chooses the best independently enumerated single upgrade')
check(valid_count > 10, 'B5 enumerates a nontrivial set of independent small-frame alternatives')
check(greedy.final_drift + 1e-12 >= best_d,
      'Exhaustive B5 result is at least as good as greedy search (not global guarantee)')
check(sum(greedy.recommended_settings.get(k, connection('rigid')) != pinned[k]
          for k in ends) <= 2, 'Greedy B5 recommendation obeys two-change budget')
check(all(step.drift_improvement_pp > 0 for step in greedy.steps),
      'B5 accepted upgrades have strictly positive drift benefit')

# 4. Input validation of user-defined constraints.
try:
    optimize_connections(m, start, modification_budget=-1)
except ValueError:
    pass
else:
    raise AssertionError('Negative budget should be rejected')
check(True, 'Invalid negative modification budget rejected')
try:
    optimize_connections(m, start, target_drift_ratio=-0.001)
except ValueError:
    pass
else:
    raise AssertionError('Negative target should be rejected')
check(True, 'Invalid negative drift target rejected')

print('Baseline 4-story drift:   ', f'{100*search.baseline_drift:.6f}%')
print('Greedy 4-story drift:    ', f'{100*search.final_drift:.6f}%')
print('4-story modifications:  ', len(search.steps))
print('4-story trial solves:   ', search.attempted_solves)
print('B5 exhaustive drift:    ', f'{100*best_d:.6f}%')
print('B5 greedy drift:        ', f'{100*greedy.final_drift:.6f}%')
print('B5 greedy gap (pp):     ', f'{100*(greedy.final_drift-best_d):.7f}')
print('ALL STEP 7 GREEDY CHECKS PASSED')
