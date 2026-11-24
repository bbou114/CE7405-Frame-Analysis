"""CE 7405 | Benchmark B2: one-story, one-bay, fixed-base portal frame.

Uses analyze_frame() from the existing working main.py (no edits required).

Geometry (m):

    node 2 ---------------- node 3       +X -->
      |           6 m           |         +Y up
      |                         |
     3 m                       3 m
      |                         |
    node 0 (fixed)           node 1 (fixed)

Two horizontal joint loads of P/2 act to the right at nodes 2 and 3.

Independent comparison: classical portal-frame slope-deflection solution
with axial deformations neglected (rigid axial idealization). The Python
finite-element model retains realistic axial flexibility, so a small
nonzero difference is expected. This is an approximate, not exact,
independent reference for the finite-A numerical model.

Units throughout: N, m, Pa, radians.
"""

import numpy as np
from main import analyze_frame


def slope_deflection_reference(E, I_col, I_beam, H, B, P):
    """Independent analytical sway solution for a symmetric fixed-base portal.

    Assumptions: identical columns; rigid beam-column joints; lateral
    joint loads P/2 at each roof corner; negligible member axial strains;
    no distributed loads; small deflections.

    Joint moment equilibrium gives equal top rotations theta:
        theta = -(6 E I_col/H^2) * Delta / (4 E I_col/H + 6 E I_beam/B)

    Lateral equilibrium gives P = k_sway * Delta:
        k_sway = 24 E I_col/H^3
                 - 72 (E I_col)^2 / (H^4 * (4 E I_col/H + 6 E I_beam/B))
    """
    EI_c = E * I_col
    EI_b = E * I_beam
    rotational_stiffness = 4 * EI_c / H + 6 * EI_b / B
    lateral_stiffness = (
        24 * EI_c / H**3
        - 72 * EI_c**2 / (H**4 * rotational_stiffness)
    )
    delta = P / lateral_stiffness
    theta = -(6 * EI_c / H**2) * delta / rotational_stiffness
    base_moment = 6 * EI_c * delta / H**2 + 2 * EI_c * theta / H
    column_top_moment = 6 * EI_c * delta / H**2 + 4 * EI_c * theta / H
    return delta, theta, base_moment, column_top_moment


def run_portal_verification():
    # Geometry and member properties; change these values to test a new frame.
    H = 3.0           # column height (m)
    B = 6.0           # beam span (m)
    E = 200e9         # Young's modulus (Pa)
    A = 0.02          # area for all members (m^2)
    I_col = 8e-5      # column moment of inertia (m^4)
    I_beam = 1.2e-4   # beam moment of inertia (m^4)
    P = 20_000.0      # TOTAL lateral story load (N)

    # Node ordering: left base, right base, left roof, right roof.
    nodes = [(0, 0), (B, 0), (0, H), (B, H)]
    elements = [
        (0, 2, E, A, I_col),  # left column
        (1, 3, E, A, I_col),  # right column
        (2, 3, E, A, I_beam),  # roof beam
    ]
    nodal_loads = {
        2: (P / 2, 0.0, 0.0),
        3: (P / 2, 0.0, 0.0),
    }
    fixed_dofs = [0, 1, 2, 3, 4, 5]  # all 3 DOFs at both bases

    U, reactions, member_forces, K = analyze_frame(
        nodes, elements, nodal_loads, fixed_dofs
    )

    (delta_ref, theta_ref, base_m_ref, top_m_ref) = slope_deflection_reference(
        E, I_col, I_beam, H, B, P
    )

    # Displacement vector order per node is [ux, uy, theta].
    ux_left = U[3 * 2]
    ux_right = U[3 * 3]
    drift = max(abs(ux_left), abs(ux_right)) / H
    base_m_left, base_m_right = reactions[2], reactions[5]
    col_left, col_right, beam = member_forces
    # Local member-end force order: [N_i, V_i, M_i, N_j, V_j, M_j]

    delta_error = 100 * abs(ux_left - delta_ref) / abs(delta_ref)
    base_m_error = 100 * abs(base_m_left - base_m_ref) / abs(base_m_ref)

    print("=" * 68)
    print("CE 7405 - B2: FIXED-BASE PORTAL FRAME VERIFICATION")
    print("=" * 68)
    print(f"Geometry: {H:.1f} m tall x {B:.1f} m wide | 4 nodes | 3 elements")
    print(f"Lateral force: {P / 1000:.1f} kN total ({P / 2000:.1f} kN per roof joint)")
    print("Both bases fixed; beam-column connections rigid")
    print()
    print("NUMERICAL FRAME RESPONSE (FULL AXIAL + FLEXURAL STIFFNESS)")
    for node in (2, 3):
        ux, uy, theta = U[3 * node:3 * node + 3]
        print(f"  Roof node {node}: ux = {ux * 1000: .6f} mm, "
              f"uy = {uy * 1000: .6f} mm, theta = {theta: .8f} rad")
    print(f"  Maximum interstory drift ratio: {drift * 100:.6f}%")
    print()
    print("INDEPENDENT SLOPE-DEFLECTION REFERENCE (AXIAL-RIGID APPROX.)")
    print(f"  Roof sway:             {delta_ref * 1000:.6f} mm")
    print(f"  Roof joint rotation:   {theta_ref:.8f} rad")
    print(f"  Each base end moment:  {base_m_ref / 1000:.6f} kN*m")
    print()
    print("COMPARISON")
    print(f"  Roof sway numerical:   {ux_left * 1000:.6f} mm")
    print(f"  Roof sway error:       {delta_error:.4f}% (target <= 1%)")
    print(f"  Base moment numerical: {base_m_left / 1000:.6f} kN*m")
    print(f"  Base moment error:     {base_m_error:.4f}% (target <= 1%)")
    print()
    print("SUPPORT REACTIONS")
    for node in (0, 1):
        Rx, Ry, M = reactions[3 * node:3 * node + 3]
        print(f"  Base node {node}: Rx = {Rx / 1000: .6f} kN, "
              f"Ry = {Ry / 1000: .6f} kN, Mz = {M / 1000: .6f} kN*m")

    # Independent physical checks (not just comparing with the same solver).
    tol_force = max(1e-5, P * 1e-9)
    tol_moment = max(1e-5, P * H * 1e-9)
    assert len(nodes) == 4 and len(elements) == 3
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T, rtol=1e-12, atol=1e-7), "Global K is not symmetric."
    assert delta_error < 1.0, "Roof sway differs >1% from slope-deflection reference."
    assert base_m_error < 1.0, "Base moment differs >1% from reference."
    assert np.isclose(ux_left, ux_right, rtol=1e-9), "Roof sway is not symmetric."
    assert np.isclose(U[8], U[11], rtol=1e-9), "Roof rotations are not symmetric."
    assert abs(reactions[0] + reactions[3] + P) < tol_force, "Horizontal equilibrium fails."
    assert abs(reactions[1] + reactions[4]) < tol_force, "Vertical equilibrium fails."
    assert abs(reactions[2] + reactions[5] + B * reactions[4] - P * H) < tol_moment, (
        "Global moment equilibrium about left base fails."
    )
    # At each rigid roof joint, member-end moments must sum to zero.
    assert abs(col_left[5] + beam[2]) < tol_moment, "Left joint moment equilibrium fails."
    assert abs(col_right[5] + beam[5]) < tol_moment, "Right joint moment equilibrium fails."
    assert np.allclose(reactions[6:], 0, atol=tol_force), "Free DOF residual is nonzero."

    print()
    print("PASS: B2 portal frame verification and equilibrium checks")
    return U, reactions, member_forces, K


if __name__ == "__main__":
    run_portal_verification()
