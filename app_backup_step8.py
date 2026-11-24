"""CE 7405 | streamlined, user-friendly Streamlit interface (Step 8).

Uses the UNCHANGED structural model, semi-rigid solver, sensitivity mapper,
and greedy optimizer from Step 7.  Inputs are kN, m, GPa, and kN*m/rad;
computations inside the solver are N, m, Pa, and radians.

Launch: conda run --no-capture-output -n ce7405 python -m streamlit run app.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

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

st.set_page_config(
    page_title="CE 7405 | Frame Explorer", page_icon="🏗️", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
  .block-container { max-width: 1440px; padding-top: 1.7rem; }
  h1, h2 { letter-spacing: -.025em; }
  div[data-testid="stMetric"] {
    border: 1px solid #d8d9e0; border-radius: 12px;
    padding: 12px 15px; background: transparent;
  }
  div[data-testid="stMetricLabel"] { font-size: .88rem; }
  div[data-testid="stTabs"] button { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

if "connections" not in st.session_state:
    st.session_state.connections = {}
if "editor_revision" not in st.session_state:
    st.session_state.editor_revision = 0


def conn_label(kind: str) -> str:
    return {"rigid": "Rigid", "semi": "Semi-rigid", "pinned": "Pinned"}.get(kind, kind)


def end_label(key):
    f, b, e = key
    return f"Floor {f} · Bay {b} · {e.title()} end"


def model_signature(m):
    if m is None:
        return None
    return (
        tuple(m.nodes), tuple(m.elements), tuple(m.fixed_dofs),
        tuple(sorted(m.nodal_loads.items())),
        tuple(sorted(m.local_y_uniform_loads.items())),
        m.n_stories, m.n_bays, m.base_support,
    )


def edited_since_analysis(m, current_connections):
    prior = st.session_state.get("analysis")
    if prior is None:
        return False
    old_model, _, _, old_settings = prior
    return (model_signature(m) != model_signature(old_model)
            or dict(current_connections) != old_settings)


def apply_manual_change(key, setting):
    updated = dict(st.session_state.connections)
    if setting.kind == "rigid":
        updated.pop(key, None)
    else:
        updated[key] = setting
    st.session_state.connections = updated
    st.session_state.editor_revision += 1
    st.session_state.pop("sensitivity", None)
    st.session_state.pop("optimization", None)


def show_figure(fig):
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)


def frame_preview(m, settings, selected=None):
    """Simple undeformed plan/elevation with a marker at the selected beam end."""
    xy = np.asarray(m.nodes, dtype=float)
    span_x = max(float(np.ptp(xy[:, 0])), 1.0)
    span_y = max(float(np.ptp(xy[:, 1])), 1.0)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for idx, (i, j, *_unused) in enumerate(m.elements):
        ax.plot(xy[[i, j], 0], xy[[i, j], 1],
                color="#7d8796" if m.element_types[idx] == "column" else "#465376",
                linewidth=2.0 if m.element_types[idx] == "column" else 2.7,
                zorder=2)
    for (floor, bay, end), setting in settings.items():
        if setting.kind == "rigid":
            continue
        left = xy[m.node_id(floor, bay - 1)]
        right = xy[m.node_id(floor, bay)]
        p = left + (.13 if end == "left" else .87) * (right - left)
        ax.scatter(p[0], p[1], s=70,
                   c="#cf9218" if setting.kind == "semi" else "#bd4e54",
                   marker="o", edgecolor="white", linewidth=1, zorder=5)
    if selected is not None:
        floor, bay, end = selected
        left = xy[m.node_id(floor, bay - 1)]
        right = xy[m.node_id(floor, bay)]
        p = left + (.13 if end == "left" else .87) * (right - left)
        ax.scatter(p[0], p[1], s=245, facecolors="none", edgecolors="#6a3696",
                   linewidth=2.4, zorder=7)
        ax.annotate("Selected", p, xytext=(10, 14), textcoords="offset points",
                    color="#6a3696", fontsize=9, weight="bold")
    for i in range(m.n_stories + 1):
        y = xy[m.node_id(i, 0), 1]
        ax.text(-.035 * span_x, y, "Base" if i == 0 else f"Floor {i}",
                fontsize=9, ha="right", va="center", color="#434b57")
    for i in range(m.n_bays):
        xleft = xy[m.node_id(0, i), 0]
        xright = xy[m.node_id(0, i + 1), 0]
        ax.text((xleft + xright) / 2, span_y * 1.08, f"Bay {i+1}",
                fontsize=9, ha="center", color="#434b57")
    for i in range(m.n_bays + 1):
        p = xy[m.node_id(0, i)]
        ax.scatter(p[0], p[1], marker="^", s=65, color="#44505f", zorder=6)
    ax.scatter([], [], color="#cf9218", s=55, label="Semi-rigid")
    ax.scatter([], [], color="#bd4e54", s=55, label="Pinned")
    ax.set_xlim(-.18 * span_x, 1.06 * span_x)
    ax.set_ylim(-.08 * span_y, 1.16 * span_y)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.legend(loc="lower center", ncol=2, frameon=False, fontsize=9)
    return fig


# --- Top heading and lightweight guide ---
st.title("Frame Explorer")
st.caption("CE 7405 · Multi-story frame analysis with semi-rigid beam-column connections")
st.markdown("**Your workflow:**  ① Build frame  →  ② Edit connections  →  ③ Run analysis  →  ④ Explore sensitivity  →  ⑤ Optimize *(optional)*")

# --- Sidebar: high-frequency inputs first; uncommon inputs folded away ---
with st.sidebar:
    st.header("Building settings")
    st.caption("Edit the basic frame here. Enter individual dimensions and floor loads under **Build frame**.")
    ns = int(st.number_input("Stories", min_value=1, max_value=5, value=4, step=1,
                             help="Number of floors above the ground"))
    nb = int(st.number_input("Bays", min_value=1, max_value=4, value=3, step=1))
    base_support = st.selectbox("Column base support", ["fixed", "pinned"],
                                format_func=lambda x: x.title(),
                                help="This is the COLUMN BASE support, not a beam-column connection.")
    beam_udl = st.number_input("Gravity load on beams (kN/m ↓)", min_value=0.0,
                               value=8.0, step=0.5)
    with st.expander("Advanced member properties", expanded=False):
        st.caption("Elastic, prismatic members. Values are used for every beam or column of the same type.")
        E_GPa = st.number_input("Elastic modulus E (GPa)", min_value=1.0, value=200.0, step=5.0)
        column_A = st.number_input("Column area (m²)", min_value=1e-6, value=0.020, format="%.5f")
        column_I = st.number_input("Column inertia (m⁴)", min_value=1e-9, value=0.00020, format="%.8f")
        beam_A = st.number_input("Beam area (m²)", min_value=1e-6, value=0.015, format="%.5f")
        beam_I = st.number_input("Beam inertia (m⁴)", min_value=1e-9, value=0.00012, format="%.8f")
    st.caption("Linear-elastic 2D frame model · Units converted to N, m, Pa for the solver")

# Streamlit preserves table edits independently for each story/bay count.
widths_df = pd.DataFrame({
    "Bay": [f"Bay {i+1}" for i in range(nb)],
    "Width (m)": [6.0] * nb,
})
stories_df = pd.DataFrame({
    "Story": [f"Story {i+1}" for i in range(ns)],
    "Height (m)": [3.5] * ns,
    "Lateral force (kN →)": [10.0 + 5.0 * i for i in range(ns)],
})

build_tab, connections_tab, analysis_tab, sensitivity_tab, optimization_tab = st.tabs([
    "① Build frame", "② Connections", "③ Analyze", "④ Sensitivity", "⑤ Optimize",
])

with build_tab:
    st.subheader("Build your frame")
    st.write("Change bay widths and floor heights or loads. You can start with the supplied 4-story, 3-bay example.")
    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown("**Bay widths**")
        width_edit = st.data_editor(
            widths_df, key=f"width_editor_{nb}", hide_index=True,
            num_rows="fixed", use_container_width=True, disabled=["Bay"],
            column_config={"Width (m)": st.column_config.NumberColumn(
                "Width (m)", min_value=0.1, step=0.25, format="%.2f")},
        )
    with rcol:
        st.markdown("**Floor heights and total lateral forces**")
        story_edit = st.data_editor(
            stories_df, key=f"story_editor_{ns}", hide_index=True,
            num_rows="fixed", use_container_width=True, disabled=["Story"],
            column_config={
                "Height (m)": st.column_config.NumberColumn(
                    "Height (m)", min_value=0.1, step=0.25, format="%.2f"),
                "Lateral force (kN →)": st.column_config.NumberColumn(
                    "Lateral force (kN →)", step=5.0, format="%.2f"),
            },
        )
    st.caption("Each story's horizontal force is distributed equally among its floor joints. Gravity loading is uniform on each floor beam.")

# Assemble the editable frame once; the same model is used by all tabs.
model = None
try:
    bay_widths = [float(v) for v in width_edit["Width (m)"]]
    story_heights = [float(v) for v in story_edit["Height (m)"]]
    story_loads = [float(v) for v in story_edit["Lateral force (kN →)"]]
    if not all(np.isfinite(v) and v > 0 for v in bay_widths + story_heights):
        raise ValueError("Bay widths and story heights must be positive and finite.")
    if not all(np.isfinite(v) for v in story_loads):
        raise ValueError("Story lateral loads must be finite.")
    model = generate_building(
        n_stories=ns, n_bays=nb, bay_widths=bay_widths,
        story_heights=story_heights, story_lateral_loads_kN=story_loads,
        E=E_GPa * 1e9, beam_A=beam_A, beam_I=beam_I,
        column_A=column_A, column_I=column_I,
        beam_udl_kN_per_m=beam_udl, base_support=base_support,
    )
except (ValueError, TypeError, KeyError) as exc:
    st.error(f"Check your frame inputs: {exc}")

# Canonicalize the manual settings if frame dimensions shrink.
if model is not None:
    allowed_keys = set(all_beam_ends(model))
    st.session_state.connections = {
        k: v for k, v in st.session_state.connections.items() if k in allowed_keys
    }
manual_settings = dict(st.session_state.connections)
stale = edited_since_analysis(model, manual_settings)

with build_tab:
    if model is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Nodes", str(len(model.nodes)))
        c2.metric("Frame members", str(len(model.elements)))
        c3.metric("Available beam ends", str(len(all_beam_ends(model))))
        show_figure(frame_preview(model, manual_settings))
        st.info("Your geometry is ready. Go to **② Connections** to edit beam ends, or **③ Analyze** to run the all-rigid default.")

# --- A visual, reversible connection editor ---
with connections_tab:
    st.subheader("Edit beam-column connections")
    st.caption("Select one beam end on a floor, choose the restraint, then apply it. Unchanged beam ends are rigid.")
    if model is not None:
        c1, c2 = st.columns([1.1, 1], gap="large")
        with c1:
            floor_col, bay_col, end_col = st.columns(3)
            floor = floor_col.selectbox("Floor", list(range(1, ns + 1)), key="connection_floor")
            bay = bay_col.selectbox("Bay", list(range(1, nb + 1)), key="connection_bay")
            end = end_col.selectbox("Beam end", ["left", "right"],
                                    format_func=str.title, key="connection_end")
            selected_key = (floor, bay, end)
            applied = manual_settings.get(selected_key, connection("rigid"))
            st.markdown(f"**Selected:** {end_label(selected_key)}")
            st.caption("Currently: " + conn_label(applied.kind) +
                       (f" · α = {applied.alpha:g}" if applied.kind == "semi" else ""))
            rev = st.session_state.editor_revision
            suffix = f"{rev}_{floor}_{bay}_{end}"
            type_options = ["Rigid", "Semi-rigid", "Pinned"]
            selected_type = st.radio(
                "Connection type", type_options, horizontal=True,
                index=type_options.index(conn_label(applied.kind)),
                key=f"type_{suffix}",
                help="Rigid: same beam/column rotation. Pinned: no moment transfer. Semi-rigid: rotational spring.",
            )
            chosen_alpha = 0.0
            if selected_type == "Semi-rigid":
                entry = st.radio("Stiffness input", ["Simple slider (α)", "Enter physical kθ"],
                                 horizontal=True, key=f"entry_{suffix}")
                EI = (E_GPa * 1e9) * beam_I
                beam_length = float(model.bay_widths[bay - 1])
                if entry == "Simple slider (α)":
                    default_alpha = applied.alpha if applied.kind == "semi" else 5.0
                    chosen_alpha = st.slider(
                        "Rotational stiffness α", min_value=0.5, max_value=100.0,
                        value=float(max(0.5, min(default_alpha, 100.0))), step=0.5,
                        key=f"alpha_slider_{suffix}",
                    )
                    st.caption(f"Physical spring stiffness: **{chosen_alpha * EI / beam_length / 1000:,.1f} kN·m/rad**")
                    st.caption("A larger α means a stiffer semi-rigid connection. For other values, use the physical input.")
                else:
                    k_default = (applied.alpha if applied.kind == "semi" else 5.0) * EI / beam_length / 1000
                    k_value = st.number_input(
                        "Rotational stiffness kθ (kN·m/rad)", min_value=0.001,
                        value=float(k_default), step=100.0, format="%.3f",
                        key=f"k_input_{suffix}",
                    )
                    chosen_alpha = float(k_value * 1000 * beam_length / EI)
                    st.caption(f"Equivalent normalized stiffness: **α = {chosen_alpha:.3f}**")
            setting_kind = {"Rigid": "rigid", "Semi-rigid": "semi", "Pinned": "pinned"}[selected_type]
            if st.button("Apply connection change", type="primary", key="apply_connection", use_container_width=True):
                apply_manual_change(selected_key, connection(setting_kind, alpha=chosen_alpha))
                st.rerun()
        with c2:
            show_figure(frame_preview(model, manual_settings, selected_key))
            st.caption("Purple outline = selected beam end · Gold = semi-rigid · Red = pinned")

        with st.expander("Quick presets for a comparison study", expanded=False):
            st.caption("These presets change **all beam ends**. They do not run optimization.")
            p1, p2, p3 = st.columns(3)
            if p1.button("Make all rigid", use_container_width=True):
                st.session_state.connections = {}
                st.session_state.editor_revision += 1
                st.session_state.pop("sensitivity", None)
                st.session_state.pop("optimization", None)
                st.rerun()
            if p2.button("All semi-rigid (α = 5)", use_container_width=True):
                st.session_state.connections = {
                    k: connection("semi", alpha=5.0) for k in all_beam_ends(model)
                }
                st.session_state.editor_revision += 1
                st.session_state.pop("sensitivity", None)
                st.session_state.pop("optimization", None)
                st.rerun()
            if p3.button("All pinned", use_container_width=True):
                st.session_state.connections = {
                    k: connection("pinned") for k in all_beam_ends(model)
                }
                st.session_state.editor_revision += 1
                st.session_state.pop("sensitivity", None)
                st.session_state.pop("optimization", None)
                st.rerun()
        st.markdown(f"**Manual edits:** {len(manual_settings)} non-rigid ends out of {len(all_beam_ends(model))}")
        if manual_settings:
            st.dataframe([{"Floor": k[0], "Bay": k[1], "End": k[2].title(),
                           "Restraint": conn_label(s.kind),
                           "α": round(s.alpha, 3) if s.kind == "semi" else None}
                          for k, s in sorted(manual_settings.items())],
                         use_container_width=True, hide_index=True)
        else:
            st.caption("All beam ends are rigid. You can analyze immediately or change one above.")
        st.info("After applying changes, go to **③ Analyze** and click **Run / Refresh Analysis**.")

# --- Results: the primary destination; never confuse draft with analyzed baseline ---
with analysis_tab:
    st.subheader("Analyze your frame")
    st.caption("This runs your manual configuration only. Sensitivity mapping and optimization are not required.")
    limit_percent = st.number_input(
        "Optional drift screening limit (%)", min_value=0.001, value=0.50,
        step=0.05, format="%.3f", key="analysis_screening_limit",
        help="Comparison threshold selected by you; not a code-prescribed limit.",
    )
    if st.button("▶ Run / Refresh Analysis", type="primary", disabled=model is None,
                 use_container_width=True, key="run_analysis"):
        try:
            with st.spinner("Solving the frame and checking equilibrium..."):
                calculated = analyze_connected_building(model, manual_settings)
                rigid_result = analyze_connected_building(model)
                residual = equilibrium_residuals(model, calculated)
                if not np.all(np.isfinite(calculated.displacements)):
                    raise ValueError("Displacements are not finite.")
                if np.max(np.abs(residual)) > 1e-3:
                    raise ValueError(f"Global equilibrium residual too large: {residual}")
                st.session_state.analysis = (model, calculated, rigid_result, dict(manual_settings))
                st.session_state.pop("sensitivity", None)
                st.session_state.pop("optimization", None)
            st.success("Analysis complete. The results below match your current frame settings.")
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            st.error(f"The model could not be analyzed: {exc}")
    # A new successful run clears any previously pending changes in this rerun.
    stale = edited_since_analysis(model, manual_settings)
    previous = st.session_state.get("analysis")
    if previous is None:
        st.info("Start here: click **Run / Refresh Analysis** to analyze your frame. All beam ends are rigid unless you've edited them.")
    else:
        analyzed_model, result, rigid_result, analyzed_settings = previous
        if stale:
            st.warning("**Changes not analyzed yet.** The figures below are from your previous run. Click **Run / Refresh Analysis** to update them before mapping or optimizing.")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Maximum interstory drift", f"{100*result.max_story_drift_ratio:.4f}%")
        r2.metric("Roof horizontal displacement", f"{1000*result.roof_max_displacement:.2f} mm")
        r3.metric("Story with greatest drift", str(result.worst_story))
        r4.metric("Non-rigid beam ends", f"{len(analyzed_settings)}")
        if 100 * result.max_story_drift_ratio > limit_percent:
            st.warning("The analyzed drift exceeds your **screening limit**. This is not a building-code determination.")
        else:
            st.success("The analyzed drift is below your screening limit (not a building-code check).")
        left, right = st.columns([1.15, 1], gap="medium")
        with left:
            st.markdown("**Frame deformation**")
            show_figure(frame_figure(analyzed_model, result, analyzed_settings))
            st.caption("Exaggerated deformation for visibility.")
        with right:
            st.markdown("**Story drift vs. all-rigid baseline**")
            show_figure(drift_figure(result, rigid_result, target_percent=limit_percent))
        drift_rows = [{
            "Story": story,
            "Your drift (%)": float(result.story_drift_ratios[story-1] * 100),
            "All-rigid drift (%)": float(rigid_result.story_drift_ratios[story-1] * 100),
            "Mean lateral Ux (mm)": float(result.mean_floor_displacements[story] * 1000),
        } for story in range(1, analyzed_model.n_stories + 1)]
        st.dataframe(drift_rows, hide_index=True, use_container_width=True)
        st.download_button("Download story drift (CSV)", csv_text(drift_rows),
                           "CE7405_story_drift.csv", "text/csv")
        with st.expander("Detailed engineering results: forces, moments, reactions, springs"):
            d1, d2, d3 = st.tabs(["Member forces", "Support reactions", "Connection springs"])
            with d1:
                forces = member_force_rows(analyzed_model, result)
                st.dataframe(forces, hide_index=True, use_container_width=True)
                st.download_button("Download member forces (CSV)", csv_text(forces),
                                   "CE7405_member_forces.csv", "text/csv")
                member_id = st.selectbox(
                    "Show bending moment diagram for member", range(1, len(forces)+1),
                    format_func=lambda i: f"Member {i}: {analyzed_model.element_types[i-1].title()}",
                )
                show_figure(moment_figure(analyzed_model, result, member_id-1))
                st.caption("Moments use local axes and include beam fixed-end force corrections.")
            with d2:
                rx_rows = reaction_rows(analyzed_model, result)
                st.dataframe(rx_rows, hide_index=True, use_container_width=True)
                st.download_button("Download reactions (CSV)", csv_text(rx_rows),
                                   "CE7405_reactions.csv", "text/csv")
                balance = equilibrium_residuals(analyzed_model, result)
                st.caption(f"Equilibrium residual: Fx={balance[0]:.2e} N · "
                           f"Fy={balance[1]:.2e} N · Mz={balance[2]:.2e} N·m")
            with d3:
                springs = []
                for key, response in sorted(result.spring_responses.items()):
                    springs.append({
                        "Floor": key[0], "Bay": key[1], "End": key[2],
                        "Type": conn_label(response.kind), "α": response.alpha,
                        "kθ (kN·m/rad)": response.k_theta_Nm_per_rad/1000,
                        "Relative rotation (rad)": response.relative_rotation_rad,
                        "Beam-end moment (kN·m)": response.beam_end_moment_Nm/1000,
                        "Moment-balance residual (kN·m)": response.beam_end_balance_residual_Nm/1000,
                    })
                if springs:
                    st.dataframe(springs, hide_index=True, use_container_width=True)
                    st.download_button("Download spring checks (CSV)", csv_text(springs),
                                       "CE7405_connection_springs.csv", "text/csv")
                else:
                    st.caption("All beam ends in this analysis are rigid.")

# --- Optional sensitivity, operating on the last CURRENT analyzed model ---
with sensitivity_tab:
    st.subheader("Which connections matter most?")
    st.write("**Optional sensitivity mapping:** change one beam end at a time and rank the effect on maximum story drift. Your manual model is not changed.")
    previous = st.session_state.get("analysis")
    if previous is None:
        st.info("Run an analysis in **③ Analyze** first.")
    else:
        analyzed_model, result, _rigid, analyzed_settings = previous
        if stale:
            st.warning("Your manual frame has changed. Run the analysis again before starting or applying a sensitivity study.")
        s1, s2, s3 = st.columns(3)
        trial_choice = s1.selectbox("Test a connection type", ["Semi-rigid", "Rigid", "Pinned"],
                                    key="map_trial_type")
        trial_kind = {"Semi-rigid": "semi", "Rigid": "rigid", "Pinned": "pinned"}[trial_choice]
        trial_alpha = 5.0
        if trial_kind == "semi":
            trial_alpha = s2.number_input("Trial stiffness α", min_value=0.001,
                                          value=5.0, step=1.0, format="%.3f", key="map_alpha")
        scope = s3.selectbox("Test which floors?", ["All floors", "One floor"], key="map_scope")
        trial_keys = all_beam_ends(analyzed_model)
        if scope == "One floor":
            chosen_floor = st.selectbox("Floor to test", range(1, analyzed_model.n_stories+1),
                                        key="map_floor")
            trial_keys = [k for k in trial_keys if k[0] == chosen_floor]
        if st.button("Map connections", type="primary", disabled=stale,
                     key="run_mapping"):
            try:
                with st.spinner(f"Testing up to {len(trial_keys)} beam ends..."):
                    st.session_state.sensitivity = map_connection_sensitivity(
                        analyzed_model, baseline_settings=analyzed_settings,
                        trial_setting=connection(trial_kind, alpha=float(trial_alpha)),
                        eligible_keys=trial_keys, baseline_result=result,
                    )
            except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
                st.error(f"Sensitivity mapping failed: {exc}")
        study = st.session_state.get("sensitivity")
        if study is not None:
            st.caption("Showing the most recent mapping run. Change the controls and click **Map connections** to recalculate.")
            successful = study.successful
            if successful:
                best = successful[0]
                a, b = st.columns(2)
                a.metric("Largest drift reduction from a single trial", f"{best.drift_reduction_pp:+.5f} pp")
                b.metric("Highest-ranked beam end", end_label(best.key))
                if best.drift_reduction_pp <= 0:
                    st.info("None of the tested changes reduces maximum drift. Negative values mean drift increased.")
            else:
                st.info("There were no stable, changed candidate configurations for this selection.")
            show_figure(sensitivity_figure(analyzed_model, study))
            st.caption("Blue = drift decreases · Red = drift increases · Gray = unchanged or unstable trial")
            map_rows = sensitivity_table_rows(study)
            st.dataframe(map_rows, hide_index=True, use_container_width=True)
            st.download_button("Download ranking (CSV)", csv_text(map_rows),
                               "CE7405_sensitivity.csv", "text/csv")
            if successful:
                st.markdown("**Try one of these changes yourself (optional)**")
                chosen_key = st.selectbox("Connection", [r.key for r in successful],
                                          format_func=end_label, key="apply_map_key")
                if st.button("Apply this one change to manual settings", disabled=stale,
                             key="apply_map"):
                    apply_manual_change(chosen_key, study.trial_setting)
                    st.rerun()
            st.caption("Ranking describes one-at-a-time changes for this frame and loading; it is not an optimization or code check.")

# --- Greedy, but never auto-apply changes ---
with optimization_tab:
    st.subheader("Find helpful connection upgrades")
    st.write("**Optional greedy optimization:** recommend connection stiffening to reduce maximum interstory drift. Review every proposed upgrade before applying.")
    previous = st.session_state.get("analysis")
    if previous is None:
        st.info("Run an analysis in **③ Analyze** first. For a useful upgrade study, start with some pinned or semi-rigid beam ends.")
    else:
        analyzed_model, result, _rigid, analyzed_settings = previous
        if stale:
            st.warning("You have un-analyzed edits. Run the analysis again before optimizing or applying recommendations.")
        if not analyzed_settings:
            st.info("Your last analysis is **all rigid**. No stronger beam-end connections are available. Try **All semi-rigid (α = 5)** under **② Connections**, then rerun analysis.")
        o1, o2 = st.columns(2)
        with o1:
            target_percent = st.number_input("Maximum drift target (%)", min_value=0.0,
                                             value=0.15, step=0.025, format="%.3f",
                                             key="opt_target",
                                             help="Your screening target, not a code-prescribed drift limit.")
            budget = int(st.number_input("How many beam ends may change?", min_value=0,
                                         max_value=len(all_beam_ends(analyzed_model)),
                                         value=min(4, len(all_beam_ends(analyzed_model))),
                                         step=1, key="opt_budget"))
        with o2:
            upgrade_mode = st.selectbox("Allowed upgrades", ["Rigid only", "Semi-rigid or rigid"],
                                        key="opt_mode")
            upgrade_alpha = 15.0
            if upgrade_mode == "Semi-rigid or rigid":
                upgrade_alpha = st.number_input("Semi-rigid candidate α", min_value=0.001,
                                                value=15.0, step=1.0, format="%.3f",
                                                key="opt_alpha")
            floor_scope = st.selectbox("Eligible floors", ["All floors", "One floor"],
                                       key="opt_scope")
        keys = all_beam_ends(analyzed_model)
        if floor_scope == "One floor":
            opt_floor = st.selectbox("Eligible floor", range(1, analyzed_model.n_stories + 1),
                                     key="opt_floor")
            keys = [k for k in keys if k[0] == opt_floor]
        st.caption("One beam end changed = one budget unit. Only upgrades are considered, never downgrades. Greedy search does not guarantee a global optimum.")
        if st.button("Find recommended upgrades", type="primary", disabled=stale,
                     key="run_optimization"):
            try:
                with st.spinner("Evaluating connection candidates..."):
                    st.session_state.optimization = optimize_connections(
                        analyzed_model, analyzed_settings,
                        target_drift_ratio=float(target_percent)/100.0,
                        modification_budget=budget,
                        allow_semi=(upgrade_mode == "Semi-rigid or rigid"),
                        upgrade_alpha=float(upgrade_alpha),
                        eligible_keys=keys, baseline_result=result,
                    )
                    st.session_state.opt_revision = st.session_state.get("opt_revision", 0) + 1
            except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
                st.error(f"Optimization failed: {exc}")
        optimum = st.session_state.get("optimization")
        if optimum is not None:
            messages = {
                "already_satisfied": "Your analyzed frame already meets the selected target.",
                "target_satisfied": "The recommended sequence reaches the target.",
                "budget_exhausted": "The allowed number of upgrades was used.",
                "no_positive_improvement": "No further single upgrade reduced drift.",
                "no_eligible_upgrades": "No eligible connection can be stiffened.",
            }
            st.write("**Search result:** " + messages.get(optimum.status, optimum.status))
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Starting drift", f"{100*optimum.baseline_drift:.4f}%")
            p2.metric("Recommended drift", f"{100*optimum.final_drift:.4f}%")
            p3.metric("Upgrades", f"{len(optimum.steps)} / {optimum.modification_budget}")
            p4.metric("Trial solves", str(optimum.attempted_solves))
            if optimum.target_met:
                st.success("Target reached for this modeled load case (not a code or strength check).")
            else:
                st.warning("Target not reached under the selected upgrades and budget.")
            if optimum.steps:
                rows = optimization_table_rows(optimum)
                st.dataframe(rows, hide_index=True, use_container_width=True)
                st.download_button("Download upgrade sequence (CSV)", csv_text(rows),
                                   "CE7405_greedy_upgrades.csv", "text/csv")
                with st.expander("Compare suggested and starting frames", expanded=True):
                    left, right = st.columns([1.1, 1])
                    with left:
                        show_figure(frame_figure(analyzed_model, optimum.final_result,
                                                 optimum.recommended_settings))
                    with right:
                        # The legacy helper's legend names are all-rigid/selected;
                        # customize axis labels to describe this actual comparison.
                        fig = drift_figure(optimum.final_result, optimum.baseline_result,
                                           target_percent=100*optimum.target_ratio)
                        for axis in fig.axes:
                            handles, labels = axis.get_legend_handles_labels()
                            labels = ["Starting model" if x == "All rigid" else
                                      "Recommended model" if x == "Selected model" else x
                                      for x in labels]
                            if handles:
                                axis.legend(handles, labels, fontsize=8)
                        show_figure(fig)
                st.markdown("#### Review changes before applying")
                st.caption("Uncheck any recommendation you want to skip. The drift from a partial selection needs to be recalculated with **③ Analyze**.")
                approval_rows = [{
                    "Apply?": True, "Step": s.number, "Floor": s.key[0],
                    "Bay": s.key[1], "End": s.key[2],
                    "Recommended type": conn_label(s.after.kind),
                    "α": s.after.alpha if s.after.kind == "semi" else None,
                } for s in optimum.steps]
                approvals = st.data_editor(
                    pd.DataFrame(approval_rows), use_container_width=True, hide_index=True,
                    disabled=["Step", "Floor", "Bay", "End", "Recommended type", "α"],
                    num_rows="fixed",
                    key=f"approval_{st.session_state.get('opt_revision', 0)}",
                )
                if st.button("Apply checked recommendations", disabled=stale,
                             key="apply_recommendations"):
                    picked = [step for step, accepted in zip(
                        optimum.steps, approvals["Apply?"].tolist()) if accepted]
                    if not picked:
                        st.warning("Choose at least one recommendation before applying.")
                    else:
                        new_settings = dict(analyzed_settings)
                        for step in picked:
                            if step.after.kind == "rigid":
                                new_settings.pop(step.key, None)
                            else:
                                new_settings[step.key] = step.after
                        st.session_state.connections = new_settings
                        st.session_state.editor_revision += 1
                        st.session_state.pop("sensitivity", None)
                        st.session_state.pop("optimization", None)
                        st.rerun()
            else:
                st.info("No changes recommended; your manual model remains unchanged.")

st.divider()
st.caption("Educational tool · Linear elastic, first-order 2D analysis. No P–Δ, connection strength, full cost optimization, or structural code compliance. Independent verification required.")
