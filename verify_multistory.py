"""Automated checks for the multi-story frame and uniform beam loads.

These are matrix-property, equilibrium, superposition, and closed-form
beam checks. A fully independent multi-story STAAD.Pro check is still
needed later for the proposal's Benchmark B4.
"""

import numpy as np

from main import analyze_frame, element_matrices
from multistory_frame import (
    analyze_building, equivalent_uniform_load_local, equilibrium_residuals,
    generate_building,
)


def check_close(actual, target, label, rtol=1e-8, atol=1e-5):
    np.testing.assert_allclose(actual, target, rtol=rtol, atol=atol,
                               err_msg=label)
    print(f"  PASS  {label}")


def verify_multistory():
    m = generate_building()
    r = analyze_building(m)
    print("CE 7405 | MULTI-STORY FRAME VERIFICATION")
    print("-" * 54)
    assert len(m.nodes) == 20
    assert len(m.elements) == 28
    assert len(m.local_y_uniform_loads) == 12
    assert r.stiffness.shape == (60, 60)
    assert len(m.fixed_dofs) == 12
    print("  PASS  4 stories / 3 bays: 20 nodes, 28 elements, 60 DOFs")
    check_close(r.stiffness, r.stiffness.T, "assembled stiffness symmetry")

    # Physical load-resultant checks, not solver comparisons.
    total_lateral_N = sum(m.story_lateral_loads_kN) * 1000.0
    total_beam_length_m = m.n_stories * sum(m.bay_widths)
    total_gravity_N = m.beam_udl_kN_per_m * 1000.0 * total_beam_length_m
    check_close(sum(r.reactions[0::3]), -total_lateral_N,
                "base shear balances all story loads")
    check_close(sum(r.reactions[1::3]), total_gravity_N,
                "vertical reactions balance total beam gravity loads")
    check_close(equilibrium_residuals(m, r), [0, 0, 0],
                "global force and moment equilibrium", atol=2e-4)

    free = sorted(set(range(3*len(m.nodes))) - set(m.fixed_dofs))
    check_close(r.reactions[free], np.zeros(len(free)),
                "zero force residual at free DOFs", atol=2e-4)
    assert np.all(r.story_drift_ratios > 0)
    assert np.all(np.isfinite(r.displacements))
    print("  PASS  finite displacements, positive story drift ratios")

    # Exactly linear system: lateral-only + gravity-only = combined response.
    base_inputs = dict(n_stories=4, n_bays=3)
    lateral = generate_building(**base_inputs, beam_udl_kN_per_m=0.0)
    gravity = generate_building(**base_inputs,
                                story_lateral_loads_kN=[0]*4)
    rl = analyze_building(lateral)
    rg = analyze_building(gravity)
    check_close(r.displacements, rl.displacements + rg.displacements,
                "linear superposition of gravity and lateral load cases",
                rtol=1e-8, atol=1e-11)

    # A fixed-fixed beam under UDL has a classical zero-displacement result:
    # local member end forces [0,wL/2,wL^2/12,0,wL/2,-wL^2/12].
    E, A, I, L, w = 200e9, 0.015, 1.2e-4, 6.0, 8_000.0
    f_eq = equivalent_uniform_load_local(-w, L)
    expected_fixed = np.array([0, w*L/2, w*L**2/12,
                               0, w*L/2, -w*L**2/12])
    check_close(-f_eq, expected_fixed,
                "fixed-fixed beam: textbook fixed-end forces")

    # Now solve a simply supported beam (left pin, right roller) with UDL.
    # Both end moments should vanish; reactions should be wL/2 each.
    nodes = [(0, 0), (L, 0)]
    elements = [(0, 1, E, A, I)]
    _, T, _, _ = element_matrices(*nodes[0], *nodes[1], E, A, I)
    fg = T.T @ f_eq
    nodal = {0: tuple(fg[:3]), 1: tuple(fg[3:])}
    U, reactions, end_forces_kd, K = analyze_frame(
        nodes, elements, nodal, [0, 1, 4]
    )
    end_forces_total = end_forces_kd[0] - f_eq
    check_close([reactions[1], reactions[4]], [w*L/2, w*L/2],
                "simply supported beam: wL/2 end reactions", atol=1e-6)
    check_close([end_forces_total[2], end_forces_total[5]], [0, 0],
                "simply supported beam: zero end moments", atol=1e-6)

    small = generate_building(
        n_stories=2, n_bays=1,
        bay_widths=[5.0], story_heights=[3.0, 3.2],
        story_lateral_loads_kN=[9.0, 12.0], beam_udl_kN_per_m=3.0,
        base_support="pinned",
    )
    rs = analyze_building(small)
    assert len(small.nodes) == 6 and len(small.elements) == 6
    assert np.all(np.isfinite(rs.displacements))
    check_close(equilibrium_residuals(small, rs), [0,0,0],
                "custom 2-story/1-bay frame and pinned bases", atol=1e-4)
    print("\nALL STEP 3 CHECKS PASSED")
    print("NOTE: Independent software comparison for the 4-story frame is still pending.")
    return m, r


if __name__ == "__main__":
    verify_multistory()
