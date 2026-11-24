"""Run this file to see your four-story, three-bay CE 7405 frame results.

Run from the project directory (in VS Code PowerShell):
& "C:\\Users\\User\\miniconda3\\Scripts\\conda.exe" run --no-capture-output -n ce7405 python .\\run_multistory.py
"""

import numpy as np

from multistory_frame import (
    generate_building, analyze_building, equilibrium_residuals,
)

# ------------------- CHANGE YOUR BUILDING INPUTS HERE ------------------
N_STORIES = 4
N_BAYS = 3
BAY_WIDTHS_M = [6.0, 6.0, 6.0]              # each horizontal bay
STORY_HEIGHTS_M = [3.5, 3.5, 3.5, 3.5]       # bottom to top

E = 200e9               # Pa (steel illustrative value)
COLUMN_A = 0.020        # m^2
COLUMN_I = 2.0e-4       # m^4
BEAM_A = 0.015          # m^2
BEAM_I = 1.2e-4         # m^4

STORY_LATERAL_KN = [10.0, 15.0, 20.0, 25.0]  # total per floor, bottom to top
BEAM_GRAVITY_KN_M = 8.0  # downward uniformly distributed, every beam
BASE_SUPPORT = "fixed"  # or "pinned"
# -----------------------------------------------------------------------


def main():
    model = generate_building(
        n_stories=N_STORIES,
        n_bays=N_BAYS,
        bay_widths=BAY_WIDTHS_M,
        story_heights=STORY_HEIGHTS_M,
        E=E, column_A=COLUMN_A, column_I=COLUMN_I,
        beam_A=BEAM_A, beam_I=BEAM_I,
        story_lateral_loads_kN=STORY_LATERAL_KN,
        beam_udl_kN_per_m=BEAM_GRAVITY_KN_M,
        base_support=BASE_SUPPORT,
    )
    result = analyze_building(model)
    n_col = model.n_stories * (model.n_bays + 1)
    n_beam = model.n_stories * model.n_bays
    print("=" * 72)
    print("CE 7405 | MULTI-STORY RIGID FRAME: BASELINE ANALYSIS")
    print("=" * 72)
    print(f"Stories x bays: {model.n_stories} x {model.n_bays}")
    print(f"Nodes: {len(model.nodes)} | Columns: {n_col} | "
          f"Beams: {n_beam} | Total elements: {len(model.elements)}")
    print(f"Total DOFs: {len(result.displacements)} | "
          f"Restrained DOFs: {len(model.fixed_dofs)}")
    print(f"Total lateral load: {sum(model.story_lateral_loads_kN):.3f} kN")
    print(f"Uniform gravity on each beam: {model.beam_udl_kN_per_m:.3f} kN/m")
    print()
    print("STORY RESPONSE (no rigid-diaphragm constraint; max across columns)")
    print("Story  Elevation(m)  Mean floor Ux(mm)  Max story drift(mm)  Drift ratio(%)")
    elev = 0.0
    for j, (h, ratio) in enumerate(zip(model.story_heights,
                                        result.story_drift_ratios), start=1):
        elev += h
        drift_mm = ratio * h * 1000.0
        ux_mm = result.mean_floor_displacements[j] * 1000.0
        print(f"  {j:>2}      {elev:>6.2f}        {ux_mm:>11.5f}"
              f"          {drift_mm:>10.5f}       {ratio*100:>9.5f}")
    print()
    print(f"Maximum absolute roof displacement: "
          f"{result.roof_max_displacement * 1000:.5f} mm")
    print(f"Maximum story drift ratio: "
          f"{result.max_story_drift_ratio*100:.5f}% "
          f"(story {result.worst_story})")
    print()
    print("BASE REACTIONS (+X right, +Y up, +Mz CCW)")
    for col in range(model.n_bays + 1):
        node = model.node_id(0, col)
        rx, ry, mz = result.reactions[3*node:3*node+3]
        print(f"Base {col+1}: Rx={rx/1000:>10.4f} kN  "
              f"Ry={ry/1000:>10.4f} kN  Mz={mz/1000:>11.4f} kN*m")
    residual = equilibrium_residuals(model, result)
    print("\nGLOBAL EQUILIBRIUM RESIDUALS")
    print(f"Sum Fx = {residual[0]:.3e} N | "
          f"Sum Fy = {residual[1]:.3e} N | "
          f"Sum Mz = {residual[2]:.3e} N*m")
    assert np.linalg.norm(residual[:2], np.inf) < 1e-3
    assert abs(residual[2]) < 1e-2
    print("PASS: force and moment equilibrium")
    print("\nNEXT: Run verify_multistory.py for additional independent checks.")
    return model, result


if __name__ == "__main__":
    model, result = main()
