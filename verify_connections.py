"""Step 4 regression checks, including textbook-informed pinned limit."""

import numpy as np

from main import local_stiffness
from multistory_frame import analyze_building, equilibrium_residuals, generate_building
from semirigid_connections import (
    all_beam_ends, analyze_connected_building, connection,
)


def verify_textbook_13_31():
    """Compare Eq. (13.31) to independent spring+beam DOF condensation.

    Textbook: McGuire/Gallagher/Ziemian, 2nd ed., p. 394, Eq. 13.31.
    Retained nodal bending DOFs: [v1, theta_joint1, v2, theta_joint2].
    Eliminated internal rotations: [theta_beam1, theta_beam2].
    """
    E, I, L = 200e9, 1.2e-4, 6.0
    a1, a2 = 2.5, 12.0
    kfull = local_stiffness(E, 0.015, I, L)
    kb = kfull[np.ix_([1,2,4,5], [1,2,4,5])]
    # Global DOF order [v1, phi1, v2, phi2, theta_b1, theta_b2].
    K6 = np.zeros((6,6))
    internal_map = [0, 4, 2, 5]
    K6[np.ix_(internal_map, internal_map)] += kb
    for alpha, phi, theta in [(a1,1,4), (a2,3,5)]:
        k = alpha*E*I/L
        K6[np.ix_([phi,theta],[phi,theta])] += k*np.array([[1,-1],[-1,1]])
    K_condensed = K6[:4,:4] - K6[:4,4:] @ np.linalg.solve(K6[4:,4:], K6[4:,:4])

    a = a1*a2/(a1*a2 + 4*a1 + 4*a2 + 12)
    z = 12/L**2 * (1 + 1/a1 + 1/a2)
    b1 = 6/L*(1 + 2/a1)
    b2 = 6/L*(1 + 2/a2)
    expected_13_31 = a*(E*I/L)*np.array([
        [ z,  b2, -z,  b1],
        [b2, 4*(1+3/a2), -b2, 2],
        [-z, -b2, z, -b1],
        [b1, 2, -b1, 4*(1+3/a1)],
    ])
    np.testing.assert_allclose(K_condensed, expected_13_31, rtol=1e-12, atol=1e-6)
    print('PASS: flexible-ended beam stiffness matches textbook Eq. (13.31)')


def verify():
    verify_textbook_13_31()
    m = generate_building()
    rigid_reference = analyze_building(m)
    rigid = analyze_connected_building(m)
    np.testing.assert_allclose(rigid.displacements, rigid_reference.displacements,
                               rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(rigid.reactions, rigid_reference.reactions,
                               rtol=1e-8, atol=1e-4)
    for f_new, f_old in zip(rigid.member_end_forces,
                            rigid_reference.member_end_forces):
        np.testing.assert_allclose(f_new, f_old, rtol=1e-8, atol=1e-5)
    assert rigid.number_of_extra_rotations == 0
    print('PASS: rigid baseline exactly matches previous Step 3 solver')

    specific = {(2, 2, 'left'): connection('pinned'),
                (2, 2, 'right'): connection('semi', alpha=10.0)}
    result = analyze_connected_building(m, specific)
    assert result.number_of_extra_rotations == 2
    np.testing.assert_allclose(result.stiffness, result.stiffness.T, atol=1e-6)
    np.testing.assert_allclose(equilibrium_residuals(m, result), [0, 0, 0], atol=1e-3)
    assert abs(result.spring_responses[(2, 2, 'left')].beam_end_moment_Nm) < 1e-4
    for r in result.spring_responses.values():
        assert abs(r.beam_end_balance_residual_Nm) < 1e-4, r
    print('PASS: pinned beam end has zero moment; finite spring balances moment')
    print('PASS: stiffness symmetric and global force/moment equilibrium')

    # alpha is normalized to the respective beam EI/L (not column stiffness).
    spring = result.spring_responses[(2, 2, 'right')]
    expected_k = 10 * 200e9 * 1.2e-4 / 6
    np.testing.assert_allclose(spring.k_theta_Nm_per_rad, expected_k)
    print('PASS: alpha = k_theta * L / (E*I) scaling')

    ends = all_beam_ends(m)
    pinned = analyze_connected_building(m, {k: connection('pinned') for k in ends})
    semi5 = analyze_connected_building(m, {k: connection('semi', alpha=5) for k in ends})
    assert pinned.number_of_extra_rotations == 24
    assert semi5.number_of_extra_rotations == 24
    assert rigid.max_story_drift_ratio < semi5.max_story_drift_ratio
    assert semi5.max_story_drift_ratio < pinned.max_story_drift_ratio
    print('PASS: expected drift ordering for this regular frame: rigid < semi < pinned')

    # Verify spring moment equilibrium at EVERY semi-rigid beam end.
    for response in semi5.spring_responses.values():
        assert abs(response.beam_end_balance_residual_Nm) < 1e-3
    np.testing.assert_allclose(equilibrium_residuals(m, semi5), [0,0,0], atol=1e-3)
    print('PASS: all 24 semi-rigid spring moments and building equilibrium')

    # High but finite stiffness should APPROACH rigid behavior; use this only
    # as a regression, not to model the actual rigid setting.
    near_rigid = analyze_connected_building(
        m, {k: connection('semi', alpha=1e5) for k in ends})
    np.testing.assert_allclose(near_rigid.story_drift_ratios,
                               rigid.story_drift_ratios, rtol=5e-4, atol=1e-7)
    print('PASS: large-alpha limit approaches exact rigid connection response')

    # Changing just one allowed setting has no effect on unrelated element
    # indexing; this small case catches floor/bay/end lookup mistakes.
    smaller = generate_building(n_stories=2, n_bays=1,
                                story_lateral_loads_kN=[10,15],
                                beam_udl_kN_per_m=5)
    rr = analyze_connected_building(smaller,
                                   {(1,1,'left'): connection('pinned')})
    assert rr.number_of_extra_rotations == 1
    assert len(rr.member_end_forces) == len(smaller.elements)
    np.testing.assert_allclose(equilibrium_residuals(smaller, rr), [0,0,0], atol=1e-3)
    print('PASS: adjustable 2-story / 1-bay frame')
    print('\nALL STEP 4 CHECKS PASSED')


if __name__ == '__main__':
    verify()
