# CE 7405 — Step 8: Friendlier Streamlit GUI

This update changes the **interface only**. The Step 7 numerical frame solver, semi-rigid beam-column connections, sensitivity mapping, and greedy optimization are **unchanged**.

## Install without losing your working version

1. Stop Streamlit with `Ctrl+C` in the VS Code terminal.
2. In `C:\Users\User\CE7405_Frame_Analysis`, rename your existing `app.py` to `app_step7_backup.py`.
3. Copy the **new** `app.py` from this update into the same folder.
4. Keep `main.py`, `multistory_frame.py`, `semirigid_connections.py`, `gui_plots.py`, `sensitivity_mapping.py`, `sensitivity_plot.py`, and `greedy_optimization.py` unchanged.
5. Run the command below in the VS Code **PowerShell terminal**:

```powershell
& "C:\Users\User\miniconda3\Scripts\conda.exe" run --no-capture-output -n ce7405 python -m streamlit run app.py
```

The app normally opens at http://localhost:8501. You don't need to install any new packages if the Step 7 GUI already runs.

## A simpler five-step workflow

1. **Build frame** — Set the number of stories and bays in the left sidebar. Edit individual bay widths, story heights, and lateral loads in two compact tables. A frame preview shows the model before analysis.
2. **Connections** — Choose a floor, bay, and beam end. The preview circles the selected end in purple; choose rigid, semi-rigid (stiffness slider or physical spring input), or pinned. Apply the change. A quick-presets panel lets you compare all-rigid, all-semi-rigid, and all-pinned models.
3. **Analyze** — Click **Run / Refresh Analysis**. Read the headline drift and roof-displacement results, frame deformation, and per-story drift plot. Detailed force, support-reaction, and spring-check tables are in a collapsed section.
4. **Sensitivity** *(optional)* — Try one connection change at a time, map its effect on drift, and optionally apply a trial manually.
5. **Optimize** *(optional)* — Set a drift target and maximum number of connections to change, evaluate greedy upgrades, inspect the comparison and approval table, and apply only checked recommendations.

The program displays a **Changes not analyzed yet** warning when the geometry, loads, supports, or manual connection settings differ from the last solved model. Mapping, optimization, and applying their recommendations are disabled until the frame is recalculated.

**Important:** Replacing `app.py` restarts/reloads the GUI. Re-enter unsaved interactive inputs after restart; this version does not yet save a project file automatically. Keep your working backup file until you've tested the new GUI.

## Quick validation

Set **All semi-rigid (α = 5)** under **Connections**, go to **Analyze**, and click **Run / Refresh Analysis**. Then run **Sensitivity** and **Optimize** to verify both work from the analyzed semi-rigid model.

Existing regression tests (unchanged from Step 7) can be run in the project folder:

```powershell
& "C:\Users\User\miniconda3\Scripts\conda.exe" run --no-capture-output -n ce7405 python .\verify_greedy.py
```

## Scope and limitations

- Units: kN, m, GPa, kN·m/rad in the GUI; the solver uses N, m, Pa, and radians.
- Drift limits are user-supplied screening targets **not** code-prescribed limits.
- Semi-rigid connections use linear rotational springs; pinned ends transmit no moment; rigid ends enforce equal rotations.
- Optimization is a greedy heuristic, **not** a guaranteed global optimum or a strength/code-compliant design.
- The baseline for **Analyze** is the all-rigid version of the current building; optimization starts from the **most recently analyzed manual** configuration.

**Verification:** Existing Step 4, Step 5 helper, Step 6, and Step 7 numerical tests all passed with the new app file. A local control-flow smoke test exercised initial launch, analysis, sensitivity, manual preset, greedy recommendation, application, and reanalysis. A full browser rendering test on your Windows installation is still recommended.
