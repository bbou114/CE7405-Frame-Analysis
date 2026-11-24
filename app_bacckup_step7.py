"""CE 7405: interactive frame analysis with optional mapping and greedy optimization.

Run from the project folder:
  conda run --no-capture-output -n ce7405 python -m streamlit run app.py

Depends on the unchanged Step 4 modules: main.py, multistory_frame.py,
semirigid_connections.py. Units in inputs are explicitly stated; the solver
uses N, m, Pa and radians. Not a structural design/code compliance tool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from multistory_frame import generate_building, equilibrium_residuals
from semirigid_connections import (
    all_beam_ends, analyze_connected_building, connection,
)
from sensitivity_mapping import map_connection_sensitivity, sensitivity_table_rows
from sensitivity_plot import sensitivity_figure
from greedy_optimization import optimize_connections, optimization_table_rows
from gui_plots import (
    frame_figure, drift_figure, moment_figure,
    member_force_rows, reaction_rows, csv_text,
)

st.set_page_config(page_title="CE 7405 | Multi-Story Frame Analysis",
                   page_icon="🏗️", layout="wide")
st.markdown("""
<style>
.main h1 { color: #461D70; }
div[data-testid="stMetric"] { border: 1px solid #DDD7E6; border-radius: .6rem; padding: .65rem; }
</style>
""", unsafe_allow_html=True)

st.title("Multi-Story Frame Analysis & Connection Optimization")
st.caption("CE 7405 • Linear-elastic 2D frame analysis • Manual editor + optional sensitivity mapping and optimization")

# Persistent manually applied modifications. Unspecified beam ends are rigid.
if "connections" not in st.session_state:
    st.session_state.connections = {}
if "editor_revision" not in st.session_state:
    st.session_state.editor_revision = 0

st.sidebar.header("Frame configuration")
ns = st.sidebar.number_input("Number of stories", min_value=1, max_value=5,
                             value=4, step=1, help="Floors above the base")
nb = st.sidebar.number_input("Number of bays", min_value=1, max_value=4,
                             value=3, step=1)
E_GPa = st.sidebar.number_input("Elastic modulus E (GPa)", min_value=1.0,
                                value=200.0, step=10.0)
with st.sidebar.expander("Member cross-sections", expanded=True):
    column_A = st.number_input("Column area A (m²)", min_value=.000001,
                               value=.020, format="%.5f")
    column_I = st.number_input("Column inertia I (m⁴)", min_value=.000000001,
                               value=.00020, format="%.8f")
    beam_A = st.number_input("Beam area A (m²)", min_value=.000001,
                             value=.015, format="%.5f")
    beam_I = st.number_input("Beam inertia I (m⁴)", min_value=.000000001,
                             value=.00012, format="%.8f")
base_support = st.sidebar.selectbox("Base support type", ["fixed", "pinned"], index=0)
beam_udl = st.sidebar.number_input("Beam gravity load (kN/m ↓)", min_value=0.0,
                                   value=8.0, step=.5)
st.sidebar.caption("Global X points right; global Y points up. Story lateral forces are shared equally among floor joints.")

geometry_tab, editor_tab, results_tab, mapping_tab, optimization_tab = st.tabs([
    "1 · Geometry & loads", "2 · Manual connections",
    "3 · Analysis results", "4 · Sensitivity map", "5 · Greedy optimization"
])

# These tables allow different widths, heights and lateral loads on each bay/story.
with geometry_tab:
    st.subheader("Edit bay widths and story loads")
    gleft, gright = st.columns(2)
    with gleft:
        st.markdown("**Bay dimensions**")
        widths_df = pd.DataFrame({
            "Bay": [f"Bay {i+1}" for i in range(nb)],
            "Width (m)": [6.0]*nb,
        })
        width_edit = st.data_editor(
            widths_df, hide_index=True, use_container_width=True,
            num_rows="fixed", key=f"width_table_{nb}",
            disabled=["Bay"],
            column_config={"Width (m)": st.column_config.NumberColumn(
                "Width (m)", min_value=.1, step=.25, format="%.2f")})
    with gright:
        st.markdown("**Story height and lateral loading**")
        levels_df = pd.DataFrame({
            "Story": [f"Story {i+1}" for i in range(ns)],
            "Height (m)": [3.5]*ns,
            "Lateral force (kN →)": [10.0+5.0*i for i in range(ns)],
        })
        story_edit = st.data_editor(
            levels_df, hide_index=True, use_container_width=True,
            num_rows="fixed", key=f"story_table_{ns}", disabled=["Story"],
            column_config={
                "Height (m)": st.column_config.NumberColumn(
                    "Height (m)", min_value=.1, step=.25, format="%.2f"),
                "Lateral force (kN →)": st.column_config.NumberColumn(
                    "Lateral force (kN →)", step=5.0, format="%.2f"),
            })
    st.info("Editing these tables changes the next analysis model. Click **Run Analysis** in the results tab to recalculate.")

# Validate the editable table before connecting it to the structural solver.
model = None
try:
    bay_widths = [float(x) for x in width_edit["Width (m)"].tolist()]
    story_heights = [float(x) for x in story_edit["Height (m)"].tolist()]
    lateral_loads = [float(x) for x in story_edit["Lateral force (kN →)"].tolist()]
    if not all(np.isfinite(x) and x > 0 for x in bay_widths+story_heights):
        raise ValueError("Bay widths and story heights must be positive finite numbers.")
    if not all(np.isfinite(x) for x in lateral_loads):
        raise ValueError("Story loads must be finite numbers.")
    model = generate_building(
        n_stories=int(ns), n_bays=int(nb),
        bay_widths=bay_widths, story_heights=story_heights,
        E=E_GPa*1e9, column_A=column_A, column_I=column_I,
        beam_A=beam_A, beam_I=beam_I,
        story_lateral_loads_kN=lateral_loads,
        beam_udl_kN_per_m=beam_udl, base_support=base_support,
    )
except (ValueError, TypeError, KeyError) as exc:
    st.error(f"Invalid frame input: {exc}")

with editor_tab:
    st.subheader("Select and edit a beam-end connection")
    st.write("Each **beam end** has an independent rotational connection to its column joint. "
             "A rigid connection shares the same rotation; a pinned end transfers zero moment; "
             "a semi-rigid end transfers moment through a rotational spring.")
    if model is not None:
        # Drop selections that no longer exist if the user reduces story/bay counts.
        valid = set(all_beam_ends(model))
        active = {k:v for k,v in st.session_state.connections.items() if k in valid}
        if len(active) != len(st.session_state.connections):
            st.session_state.connections = active
        e1, e2, e3 = st.columns(3)
        floor = e1.selectbox("Floor", list(range(1,ns+1)), index=0)
        bay = e2.selectbox("Bay", list(range(1,nb+1)), index=0)
        end = e3.selectbox("Beam end", ["left", "right"])
        key = (floor,bay,end)
        applied = active.get(key, connection("rigid"))
        st.caption(f"Currently applied to {key}: **{applied.kind}**" +
                   (f", α = {applied.alpha:g}" if applied.kind == "semi" else ""))

        suffix = f"{floor}_{bay}_{end}_{st.session_state.editor_revision}"
        selected_kind = st.selectbox("Proposed connection type",
                                     ["rigid", "semi", "pinned"],
                                     index=["rigid", "semi", "pinned"].index(applied.kind),
                                     key=f"kind_{suffix}")
        alpha = 0.0
        if selected_kind == "semi":
            input_mode = st.radio("Stiffness entry", ["Normalized α", "Physical kθ"],
                                  horizontal=True, key=f"mode_{suffix}")
            beam_L = float(model.bay_widths[bay-1])
            beam_EI = (E_GPa*1e9)*beam_I
            if input_mode == "Normalized α":
                alpha = st.number_input("α = kθ L / (EI)", min_value=.001,
                                        value=float(applied.alpha if applied.kind == "semi" else 5.0),
                                        max_value=1e6, step=1.0, format="%.3f",
                                        key=f"alpha_{suffix}")
                k_knm = alpha * beam_EI / beam_L / 1000.0
                st.caption(f"Equivalent rotational stiffness: **{k_knm:,.2f} kN·m/rad**")
            else:
                initial_k = ((applied.alpha if applied.kind == "semi" else 5.0)
                             * beam_EI / beam_L / 1000.0)
                k_knm = st.number_input("kθ (kN·m/rad)", min_value=.001,
                                        value=float(initial_k), step=100.0,
                                        format="%.3f", key=f"k_{suffix}")
                alpha = k_knm*1000*beam_L/beam_EI
                st.caption(f"Equivalent normalized stiffness: **α = {alpha:.3f}**")
            st.caption("For α-based entries, physical kθ changes if you change E, I, or beam length.")

        if st.button("Apply selected connection", type="primary"):
            if selected_kind == "rigid":
                st.session_state.connections.pop(key, None)
            else:
                st.session_state.connections[key] = connection(selected_kind, alpha=float(alpha))
            st.session_state.editor_revision += 1
            st.success("Connection applied. Select **Run Analysis** to update the response.")
            st.rerun()

        b1,b2,b3 = st.columns(3)
        if b1.button("All rigid (reset)"):
            st.session_state.connections = {}
            st.session_state.editor_revision += 1
            st.rerun()
        if b2.button("All semi-rigid, α = 5"):
            st.session_state.connections = {k: connection("semi",alpha=5.0)
                                            for k in all_beam_ends(model)}
            st.session_state.editor_revision += 1
            st.rerun()
        if b3.button("All pinned"):
            st.session_state.connections = {k: connection("pinned")
                                            for k in all_beam_ends(model)}
            st.session_state.editor_revision += 1
            st.rerun()

        active = st.session_state.connections
        st.markdown(f"**Applied nonrigid beam ends: {len(active)} of {len(valid)}**")
        if active:
            table = [{"Floor": f, "Bay": b, "Beam end": end,
                      "Type": s.kind, "α": s.alpha if s.kind == "semi" else "—"}
                     for (f,b,end),s in sorted(active.items())]
            st.dataframe(table, hide_index=True, use_container_width=True)
        else:
            st.caption("All beam ends are currently rigid.")

with results_tab:
    st.subheader("Run manual structural analysis")
    st.caption("Manual analysis works independently of sensitivity mapping and greedy optimization.")
    drift_limit_percent = st.number_input("User-defined drift screening limit (%)",
                                          min_value=.001, value=.50, step=.05, format="%.3f",
                                          help="Informational comparison only; not a building-code limit.")
    if st.button("▶ Run Analysis", type="primary", disabled=model is None):
        try:
            chosen = analyze_connected_building(model, dict(st.session_state.connections))
            rigid = analyze_connected_building(model)
            balance = equilibrium_residuals(model, chosen)
            if np.max(np.abs(balance)) > 1e-3:
                raise ValueError(f"Global equilibrium check failed: {balance}")
            if not np.all(np.isfinite(chosen.displacements)):
                raise ValueError("Nonfinite displacement result.")
            # Store *the analyzed model* rather than the editable GUI's current draft.
            st.session_state.analysis = (model, chosen, rigid,
                                         dict(st.session_state.connections))
            st.session_state.pop("sensitivity", None)  # Previous rankings become stale
            st.session_state.pop("optimization", None)  # Previous recommendations become stale
        except (ValueError, np.linalg.LinAlgError) as exc:
            st.error(f"Model could not be analyzed: {exc}")
    if "analysis" not in st.session_state:
        st.info("Configure your frame and connections, then click **Run Analysis** to view results.")
    else:
        used_model, chosen, rigid, used_settings = st.session_state.analysis
        st.warning("Showing the **last run**. If you changed inputs or applied connections since then, click **Run Analysis** again.")
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Maximum roof |Ux|",f"{chosen.roof_max_displacement*1e3:.3f} mm",
                  delta=f"{(chosen.roof_max_displacement-rigid.roof_max_displacement)*1000:+.3f} mm vs rigid", delta_color="off")
        m2.metric("Maximum drift",f"{chosen.max_story_drift_ratio*100:.4f}%",
                  delta=f"{(chosen.max_story_drift_ratio-rigid.max_story_drift_ratio)*100:+.4f} pp vs rigid", delta_color="off")
        m3.metric("Worst story",str(chosen.worst_story))
        m4.metric("Nonrigid beam ends",str(chosen.number_of_extra_rotations))
        if chosen.max_story_drift_ratio*100 > drift_limit_percent:
            st.warning("Maximum drift exceeds your **user-defined screening limit**. This is not a code compliance determination.")
        else:
            st.success("Maximum drift is below your user-defined screening limit (not a code compliance check).")
        p1,p2 = st.columns([1.25,1])
        with p1:
            fig = frame_figure(used_model, chosen, used_settings)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        with p2:
            fig = drift_figure(chosen, rigid, target_percent=drift_limit_percent)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

        st.markdown("#### Story-by-story response")
        drifts = [{"Story": f, "Rigid drift (%)":rigid.story_drift_ratios[f-1]*100,
                   "Selected drift (%)":chosen.story_drift_ratios[f-1]*100,
                   "Mean floor Ux (mm)": chosen.mean_floor_displacements[f]*1000}
                  for f in range(1,used_model.n_stories+1)]
        st.dataframe(drifts, hide_index=True, use_container_width=True)
        st.download_button("Download drift CSV", csv_text(drifts), "CE7405_story_drift.csv", "text/csv")

        tab_forces, tab_reactions, tab_springs = st.tabs([
            "Member forces & moments", "Support reactions", "Connection spring checks"
        ])
        with tab_forces:
            forces = member_force_rows(used_model, chosen)
            st.dataframe(forces, hide_index=True, use_container_width=True)
            st.download_button("Download member forces CSV",csv_text(forces),
                               "CE7405_member_forces.csv","text/csv")
            selected_member = st.selectbox("Plot internal bending moment for member",
                                           list(range(1,len(forces)+1)),
                                           format_func=lambda n: f"Member {n}: {used_model.element_types[n-1]}")
            fig = moment_figure(used_model, chosen, selected_member-1)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
            st.caption("Moment diagram uses local element sign convention and includes fixed-end force corrections for uniform loads.")
        with tab_reactions:
            reactions = reaction_rows(used_model, chosen)
            st.dataframe(reactions, hide_index=True, use_container_width=True)
            st.download_button("Download reactions CSV",csv_text(reactions),
                               "CE7405_reactions.csv","text/csv")
            residual = equilibrium_residuals(used_model, chosen)
            st.caption(f"Global equilibrium residual: Fx={residual[0]:.3e} N, "
                       f"Fy={residual[1]:.3e} N, Mz={residual[2]:.3e} N·m")
        with tab_springs:
            responses=[]
            for key,r in sorted(chosen.spring_responses.items()):
                responses.append({"Floor": key[0],"Bay":key[1],"End":key[2],
                    "Type":r.kind,"α":r.alpha,"kθ (kN·m/rad)":r.k_theta_Nm_per_rad/1000,
                    "Relative rotation (rad)":r.relative_rotation_rad,
                    "Beam end moment (kN·m)":r.beam_end_moment_Nm/1000,
                    "Moment balance error (kN·m)":r.beam_end_balance_residual_Nm/1000})
            if responses:
                st.dataframe(responses, hide_index=True, use_container_width=True)
                st.download_button("Download connection results CSV",csv_text(responses),
                                   "CE7405_connection_results.csv","text/csv")
            else:
                st.caption("No flexible connections; all beam ends are rigid.")


with mapping_tab:
    st.subheader("Connection Sensitivity Mapping")
    st.write("Independently change **one beam end at a time**, rerun the structural "
             "solver, and rank the effect on **maximum interstory drift**. "
             "Positive sensitivity means the change **reduces** drift. "
             "Negative sensitivity means it **increases** drift. "
             "No changes are applied automatically.")
    if "analysis" not in st.session_state:
        st.info("First click **Run Analysis** in the Analysis results tab. "
                "The sensitivity map starts from that selected/manual configuration.")
    else:
        used_model, selected_result, _, used_settings = st.session_state.analysis
        st.warning("Mapping uses your **last analyzed** frame, loads, and connections. "
                   "If you edited them since then, run the analysis again before mapping.")
        st.metric("Selected-model baseline maximum drift",
                  f"{100*selected_result.max_story_drift_ratio:.4f}%")

        col1, col2, col3 = st.columns(3)
        trial_type = col1.selectbox("One-at-a-time trial connection",
                                    ["semi", "rigid", "pinned"], index=0,
                                    help="Start with semi from the all-rigid baseline; "
                                         "use rigid to examine upgrades from nonrigid settings.")
        trial_alpha = 5.0
        if trial_type == "semi":
            trial_alpha = col2.number_input("Trial stiffness α = kθL/(EI)",
                                            min_value=.001, max_value=1e6,
                                            value=5.0, step=1.0, format="%.3f")
        area = col3.selectbox("Connections to examine", ["All floors", "One floor"])
        keys_to_evaluate = all_beam_ends(used_model)
        if area == "One floor":
            selected_floor = st.selectbox("Selected floor", list(range(1, used_model.n_stories+1)))
            keys_to_evaluate = [key for key in keys_to_evaluate if key[0] == selected_floor]

        if st.button("Map Connections", type="primary"):
            try:
                with st.spinner(f"Testing up to {len(keys_to_evaluate)} connections independently..."):
                    new_study = map_connection_sensitivity(
                        model=used_model,
                        baseline_settings=used_settings,
                        trial_setting=connection(trial_type, alpha=float(trial_alpha)),
                        eligible_keys=keys_to_evaluate,
                        baseline_result=selected_result,
                    )
                    st.session_state.sensitivity = new_study
            except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
                st.error(f"Sensitivity mapping could not finish: {exc}")

        if "sensitivity" in st.session_state:
            study = st.session_state.sensitivity
            st.caption(f"Last mapped trial state: **{study.trial_setting.kind}**"
                       + (f" (α = {study.trial_setting.alpha:g})" if study.trial_setting.kind == "semi" else "")
                       + ". Click Map Connections to apply new trial controls.")
            valid_trials = study.successful
            skipped = sum(r.status == "unchanged" for r in study.records)
            invalid = sum(r.status == "invalid" for r in study.records)
            st.caption(f"{len(valid_trials)} successful trial analyses; "
                       f"{skipped} unchanged; {invalid} invalid/unstable. "
                       "Each trial changes exactly one beam end relative to the analyzed baseline.")
            if valid_trials:
                best = valid_trials[0]
                if study.relative_score_available:
                    value = f"{best.improvement_percent:+.3f}%"
                    metric_label = "Best relative reduction in maximum drift"
                else:
                    value = f"{best.drift_reduction_pp:+.5f} pp"
                    metric_label = "Best absolute reduction in drift"
                a, b = st.columns(2)
                a.metric(metric_label, value)
                b.metric("Most influential tested end",
                         f"Floor {best.key[0]}, Bay {best.key[1]}, {best.key[2]}")
                if best.drift_reduction_pp <= 0:
                    st.info("None of the tested changes reduced maximum drift. "
                            "The ranked map still shows which ones increased it most or least.")
            else:
                st.info("No changed, stable configurations were available for this selection.")

            fig = sensitivity_figure(used_model, study)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
            st.caption("Blue = improved maximum story drift; red = worsened drift; "
                       "gray = already in trial state or invalid. "
                       "Two markers per beam identify its left and right connection separately.")
            rows = sensitivity_table_rows(study)
            st.dataframe(rows, hide_index=True, use_container_width=True)
            st.download_button("Download sensitivity ranking CSV", csv_text(rows),
                               "CE7405_sensitivity_mapping.csv", "text/csv")
            if valid_trials:
                st.markdown("#### Optionally apply one mapped change")
                trial_key = st.selectbox(
                    "Select a connection to change manually",
                    [r.key for r in valid_trials],
                    format_func=lambda k: f"Floor {k[0]}, Bay {k[1]}, {k[2]} end",
                )
                if st.button("Apply selected trial to manual editor"):
                    updated = dict(st.session_state.connections)
                    if study.trial_setting.kind == "rigid":
                        updated.pop(trial_key, None)
                    else:
                        updated[trial_key] = study.trial_setting
                    st.session_state.connections = updated
                    st.session_state.editor_revision += 1
                    st.success("Applied to your current manual editor. "
                               "Click **Run Analysis** to calculate the updated response. "
                               "If other edits are pending, your resulting drift may differ "
                               "from the one-change sensitivity trial.")
                    st.rerun()
            st.caption("This feature ranks single-change effects only; it does NOT search "
                       "connection combinations or guarantee an optimal design. "
                       "Trial rankings are specific to the selected load case and frame.")



def _model_signature(m):
    """Detect edits to geometry, sections, supports and loads since last analysis."""
    if m is None:
        return None
    return (
        tuple(m.nodes), tuple(m.elements), tuple(m.fixed_dofs),
        tuple(sorted(m.nodal_loads.items())),
        tuple(sorted(m.local_y_uniform_loads.items())),
        m.n_stories, m.n_bays,
    )


with optimization_tab:
    st.subheader("Optional greedy connection optimization")
    st.write(
        "Starting from your **last analyzed manual configuration**, evaluate allowed "
        "beam-end **upgrades** and select the one with the greatest positive decrease "
        "in maximum interstory drift. Recalculate after each choice. "
        "The search stops at your target, modification budget, or when no beneficial upgrade remains. "
        "**Nothing is applied** until you explicitly accept recommendations."
    )
    if "analysis" not in st.session_state:
        st.info("First click **Run Analysis**. The manual configuration becomes the optimization baseline.")
    else:
        used_model, selected_result, _, used_settings = st.session_state.analysis
        model_current = _model_signature(model) == _model_signature(used_model)
        connections_current = dict(st.session_state.connections) == used_settings
        baseline_current = model_current and connections_current
        if not baseline_current:
            st.warning("Your editable model differs from the last analyzed baseline. "
                       "Click **Run Analysis** before optimizing or applying any recommendations.")
        d_base = 100 * selected_result.max_story_drift_ratio
        st.metric("Current analyzed maximum story drift", f"{d_base:.5f}%")
        if not used_settings:
            st.info("The analyzed baseline is **all rigid**. No stronger beam-end connection "
                    "is available in this model. For a meaningful upgrade search, first set "
                    "some beam ends to pinned or semi-rigid in the manual editor and rerun analysis.")

        left, center, right = st.columns(3)
        drift_target = left.number_input(
            "Target maximum drift (%)", min_value=0.0,
            value=0.15, step=0.025, format="%.3f", key="opt_target_percent",
            help="User-selected screening target, NOT a code-prescribed limit.")
        opt_budget = center.number_input(
            "Maximum beam ends to change", min_value=0,
            max_value=len(all_beam_ends(used_model)), value=min(4, len(all_beam_ends(used_model))),
            step=1, key="opt_budget")
        options = right.selectbox("Allowed connection upgrades", [
            "Rigid only", "Semi-rigid or rigid"], key="opt_connection_choices")
        allow_semi = options == "Semi-rigid or rigid"
        if allow_semi:
            opt_alpha = st.number_input(
                "Semi-rigid upgrade stiffness α = kθ L/(EI)",
                min_value=0.001, max_value=1e6, value=15.0,
                step=1.0, format="%.3f", key="opt_alpha")
        else:
            opt_alpha = 15.0
        scope = st.selectbox("Connections eligible for upgrades", [
            "All floors", "One floor"], key="opt_scope")
        eligible = all_beam_ends(used_model)
        if scope == "One floor":
            which_floor = st.selectbox("Eligible floor", range(1, used_model.n_stories+1),
                                       key="opt_selected_floor")
            eligible = [k for k in eligible if k[0] == which_floor]

        st.caption("Only stiffness-increasing changes are considered. Each end can be "
                   "upgraded at most once, and uses one budget unit. "
                   "The result is a greedy heuristic, not a global minimum or an engineering design.")
        if st.button("Optimize Connections", type="primary", disabled=not baseline_current):
            try:
                with st.spinner("Reanalyzing candidate upgrades at each iteration..."):
                    outcome = optimize_connections(
                        used_model, used_settings,
                        target_drift_ratio=float(drift_target)/100.0,
                        modification_budget=int(opt_budget),
                        allow_semi=allow_semi,
                        upgrade_alpha=float(opt_alpha),
                        eligible_keys=eligible,
                        baseline_result=selected_result,
                    )
                    st.session_state.optimization = outcome
                    st.session_state.optimization_revision = (
                        st.session_state.get("optimization_revision", 0) + 1)
            except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
                st.error(f"Optimization could not finish: {exc}")

        if "optimization" in st.session_state:
            opt = st.session_state.optimization
            status_labels = {
                "already_satisfied": "Baseline already meets the drift target — no changes needed.",
                "target_satisfied": "A recommended sequence met the user-entered drift target.",
                "budget_exhausted": "The modification budget was reached before the target.",
                "no_positive_improvement": "No remaining trial produced a meaningful reduction in maximum drift.",
                "no_eligible_upgrades": "No eligible connection upgrade is available.",
            }
            st.caption(f"Last optimization run: target {100*opt.target_ratio:.3f}%, "
                       f"budget {opt.modification_budget} beam ends; "
                       f"{'semi-rigid and rigid' if opt.allow_semi else 'rigid only'} upgrades. "
                       "Change controls and click Optimize Connections to recalculate.")
            st.write("**Search outcome:** " + status_labels.get(opt.status, opt.status))
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Baseline max drift", f"{100*opt.baseline_drift:.5f}%")
            r2.metric("Recommended max drift", f"{100*opt.final_drift:.5f}%",
                      delta=f"{100*(opt.final_drift-opt.baseline_drift):+.5f} pp",
                      delta_color="off")
            r3.metric("Recommended changes", f"{len(opt.steps)} / {opt.modification_budget}")
            r4.metric("Trial analyses", str(opt.attempted_solves))
            if opt.target_met:
                st.success("Target satisfied for the modeled load case (not a code check).")
            else:
                st.warning("Drift target not met. Consider a different baseline, larger budget, "
                           "or another analysis/design approach; no global optimum is guaranteed.")
            st.caption(f"{opt.invalid_solves} invalid/unstable candidate solves skipped. "
                       "No trial changes were applied to your manual editor.")

            if opt.steps:
                rows = optimization_table_rows(opt)
                st.markdown("#### Greedy upgrade sequence")
                st.dataframe(rows, hide_index=True, use_container_width=True)
                st.download_button("Download upgrade sequence CSV", csv_text(rows),
                                   "CE7405_greedy_upgrade_sequence.csv", "text/csv")
                with st.expander("Show baseline vs recommended response plots", expanded=True):
                    a, b = st.columns([1.15, 1])
                    with a:
                        fig = frame_figure(used_model, opt.final_result, opt.recommended_settings)
                        st.pyplot(fig, clear_figure=True)
                        plt.close(fig)
                    with b:
                        fig = drift_figure(opt.final_result, opt.baseline_result,
                                           target_percent=100*opt.target_ratio)
                        st.pyplot(fig, clear_figure=True)
                        plt.close(fig)
                    st.caption("Drift plot compares the recommended configuration against "
                               "the last analyzed manual baseline (not necessarily all rigid).")

                st.markdown("#### Review upgrades before applying")
                st.write("Uncheck any connection changes you do **not** want. "
                         "After applying your selection, run Manual Analysis again to "
                         "verify the actual response; partial selections can have different drifts.")
                apply_df = pd.DataFrame([
                    {"Apply?": True, "Step": step.number, "Floor": step.key[0],
                     "Bay": step.key[1], "End": step.key[2],
                     "New type": step.after.kind,
                     "New α": step.after.alpha if step.after.kind == "semi" else None}
                    for step in opt.steps
                ])
                choices = st.data_editor(
                    apply_df, hide_index=True, use_container_width=True,
                    disabled=["Step", "Floor", "Bay", "End", "New type", "New α"],
                    num_rows="fixed",
                    key=f"opt_accept_{st.session_state.get('optimization_revision', 0)}")
                if st.button("Apply checked changes to manual editor",
                             disabled=not baseline_current):
                    selected = [s for s, use in zip(opt.steps, choices["Apply?"].tolist()) if use]
                    if not selected:
                        st.warning("Select at least one proposed change to apply.")
                    else:
                        updated = dict(used_settings)
                        for step in selected:
                            if step.after.kind == "rigid":
                                updated.pop(step.key, None)
                            else:
                                updated[step.key] = step.after
                        st.session_state.connections = updated
                        st.session_state.editor_revision += 1
                        st.session_state.pop("optimization", None)
                        st.session_state.pop("sensitivity", None)
                        st.rerun()
            else:
                st.info("No upgrades were recommended. Your manual settings remain unchanged.")


st.divider()
st.caption("CE 7405 educational model • First-order linear elasticity; no P–Δ, strength/capacity, connection design, or code checks. "
           "Verify independently before any engineering use.")
