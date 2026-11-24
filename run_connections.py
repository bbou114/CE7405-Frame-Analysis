"""CE 7405 Step 4: compare rigid, pinned and semi-rigid beam ends.

Run in VS Code PowerShell from the project folder:
& "C:\\Users\\User\\miniconda3\\Scripts\\conda.exe" run --no-capture-output -n ce7405 python .\\run_connections.py
"""

from multistory_frame import generate_building, equilibrium_residuals
from semirigid_connections import (
    all_beam_ends, analyze_connected_building, connection,
)


# EDIT THESE: floor and bay are numbered starting from 1.
# (floor, bay, 'left' or 'right'): connection('semi', alpha=...)
MANUAL_CONNECTIONS = {
    (2, 2, 'left'): connection('semi', alpha=5.0),
    (2, 2, 'right'): connection('pinned'),
    (3, 1, 'right'): connection('semi', alpha=15.0),
}


def show_row(label, result):
    print(f"{label:<24s} "
          f"{result.roof_max_displacement*1000:>11.5f} "
          f"{result.max_story_drift_ratio*100:>15.6f} "
          f"{result.worst_story:>8d} "
          f"{result.number_of_extra_rotations:>13d}")


def main():
    model = generate_building(n_stories=4, n_bays=3)
    ends = all_beam_ends(model)
    all_pinned = {key: connection('pinned') for key in ends}
    all_semi = {key: connection('semi', alpha=5.0) for key in ends}

    rigid = analyze_connected_building(model)
    semi = analyze_connected_building(model, all_semi)
    pinned = analyze_connected_building(model, all_pinned)
    manual = analyze_connected_building(model, MANUAL_CONNECTIONS)

    print('='*83)
    print('CE 7405 | STEP 4: SEMI-RIGID BEAM-COLUMN CONNECTIONS')
    print('='*83)
    print(f'{"Configuration":<24s} {"Roof Ux (mm)":>11s} '
          f'{"Max drift (%)":>15s} {"Story":>8s} {"Extra rot DOF":>13s}')
    print('-'*83)
    for name, result in [('All rigid', rigid), ('All semi, alpha=5', semi),
                         ('All pinned', pinned), ('Manual selection', manual)]:
        show_row(name, result)

    print('\nMANUAL BEAM-END CONNECTIONS')
    for key in sorted(manual.spring_responses):
        r = manual.spring_responses[key]
        print(f"  {str(key):<19s} {r.kind:<6s}  alpha={r.alpha:>5.1f}  "
              f"k={r.k_theta_Nm_per_rad/1000:>10.2f} kN*m/rad  "
              f"M_beam={r.beam_end_moment_Nm/1000:>10.5f} kN*m  "
              f"relative rotation={r.relative_rotation_rad:>+.6e} rad")
    print('\nSTORY DRIFT RATIOS (%)')
    print(f'{"Story":>5} {"Rigid":>13} {"Semi":>13} {"Pinned":>13} {"Manual":>13}')
    for s in range(model.n_stories):
        print(f'{s+1:>5} {rigid.story_drift_ratios[s]*100:>13.6f} '
              f'{semi.story_drift_ratios[s]*100:>13.6f} '
              f'{pinned.story_drift_ratios[s]*100:>13.6f} '
              f'{manual.story_drift_ratios[s]*100:>13.6f}')
    for label, res in [('rigid',rigid), ('semi',semi), ('pinned',pinned), ('manual',manual)]:
        balance = equilibrium_residuals(model, res)
        assert max(abs(balance)) < 1e-3, (label, balance)
        assert all(abs(r.beam_end_balance_residual_Nm) < 1e-3
                   for r in res.spring_responses.values()), label
    print('\nPASS: global equilibrium and beam-end spring moment balance')
    print('NOTE: This linear-elastic model is for study and preliminary comparison, not design approval.')


if __name__ == '__main__':
    main()
