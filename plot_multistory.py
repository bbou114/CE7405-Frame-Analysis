"""Optional plots for Step 3. Run using:

conda run --no-capture-output -n ce7405 python plot_multistory.py

Requires matplotlib, but the solver and verification scripts need only NumPy.
Creates frame_deformation.png and story_drift.png next to this file.
"""

from pathlib import Path

import numpy as np

from multistory_frame import generate_building, analyze_building


def deflected_member_points(model, result, element_index, scale, samples=26):
    """Interpolate Euler-Bernoulli axial/beam displacement from nodal DOFs."""
    i, j, E, A, I = model.elements[element_index]
    xi, yi = model.nodes[i]
    xj, yj = model.nodes[j]
    L = np.hypot(xj-xi, yj-yi)
    c, s = (xj-xi)/L, (yj-yi)/L
    T = np.array([
        [c, s, 0, 0, 0, 0], [-s, c, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0], [0, 0, 0, c, s, 0],
        [0, 0, 0, -s, c, 0], [0, 0, 0, 0, 0, 1],
    ])
    indices = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
    u1, v1, th1, u2, v2, th2 = T @ result.displacements[indices]
    t = np.linspace(0.0, 1.0, samples)
    # Linear axial interpolation, cubic Hermite flexural interpolation.
    du = (1-t)*u1 + t*u2
    dv = ((1-3*t*t+2*t*t*t)*v1
          + L*(t-2*t*t+t*t*t)*th1
          + (3*t*t-2*t*t*t)*v2
          + L*(-t*t+t*t*t)*th2)
    x_ref = xi + t*(xj-xi)
    y_ref = yi + t*(yj-yi)
    x_def = x_ref + scale*(c*du-s*dv)
    y_def = y_ref + scale*(s*du+c*dv)
    return x_ref, y_ref, x_def, y_def


def save_plots(model, result, output_dir=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is needed only for plotting. "
                           "Install it using 'conda install -n ce7405 matplotlib'.") from exc

    output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    # Amplify deformation so that the change is visible on an 18 m-wide frame.
    max_trans = max(np.hypot(result.displacements[0::3],
                             result.displacements[1::3]))
    scale = (0.12 * max(model.story_heights) / max_trans) if max_trans else 1.0

    fig, ax = plt.subplots(figsize=(9, 7))
    # Reuse Matplotlib's default two colors across every frame member.
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_original, color_deformed = default_colors[0], default_colors[1]
    for e in range(len(model.elements)):
        x0, y0, xd, yd = deflected_member_points(model, result, e, scale)
        ax.plot(x0, y0, linestyle="--", linewidth=1.1,
                color=color_original, alpha=0.65,
                label="Original geometry" if e == 0 else None)
        ax.plot(xd, yd, linewidth=1.9, color=color_deformed,
                label=f"Deformed x{scale:.1f}" if e == 0 else None)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Global X (m)")
    ax.set_ylabel("Global Y (m)")
    ax.set_title("CE 7405: Multi-story frame - original vs. deflected")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path_deform = output_dir / "frame_deformation.png"
    fig.savefig(path_deform, dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    levels = np.arange(1, model.n_stories + 1)
    values = 100.0 * result.story_drift_ratios
    ax.barh(levels, values, height=0.55)
    ax.set_yticks(levels)
    ax.set_ylabel("Story number (ground to roof)")
    ax.set_xlabel("Maximum interstory drift ratio (%)")
    ax.set_title("CE 7405: Interstory drift by floor")
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlim(0, max(values)*1.22)
    for floor, value in zip(levels, values):
        ax.text(value + max(values)*0.012, floor, f"{value:.4f}%",
                va="center", fontsize=10)
    fig.tight_layout()
    path_drift = output_dir / "story_drift.png"
    fig.savefig(path_drift, dpi=190)
    plt.close(fig)

    print(f"Saved: {path_deform}")
    print(f"Saved: {path_drift}")
    return path_deform, path_drift


if __name__ == "__main__":
    m = generate_building()
    r = analyze_building(m)
    save_plots(m, r)
