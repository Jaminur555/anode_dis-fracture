"""
Results pipeline: regenerates every figure reported in the paper and prints
every number to the console (headline numbers live in the README). Outputs:
  - results/figures/*.png
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from anode_dis import (GRAPHITE, SILICON, AnodeMaterials, DiffusionSolver,
    elastic_stress,plastic_core_shell_stress,
    griffith_fracture_stress,peak_hoop_stress,
    find_critical_radius,)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(HERE, "results", "figures")
os.makedirs(FIGD, exist_ok=True)


def log(msg=""):
    print(msg)


def savefig(fig, name):
    path = os.path.join(FIGD, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  [fig] {os.path.relpath(path, HERE)}")


# ---- publication figure style --------------------------------------------
# Palette: validated CVD-safe categorical pair + ordinal ramp (dataviz method).
INK    = "#0b0b0b"  # primary text
INK2   = "#52514e"  # secondary text (ticks, value labels)
MUTED  = "#898781"  # reference lines, annotations
GRID   = "#e1e0d9"  # hairline grid
AXIS   = "#c3c2b7"  # axis spines
BLUE   = "#2a78d6"  # categorical slot 1
ORANGE = "#eb6834"  # categorical slot 2

SOC_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # light -> dark

plt.rcParams.update(
    {
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
        "lines.linewidth": 1.3, "lines.solid_capstyle": "round",
        "axes.labelsize": 8, "axes.titlesize": 8.5, "axes.titlepad": 5.0,
        "axes.linewidth": 0.6, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.fontsize": 7, "legend.frameon": False, "legend.handlelength": 1.6,
        "figure.dpi": 150, "savefig.dpi": 300,
    }
)


def ref_label(ax, x, y, text, ha="right", va="bottom"):
    """Small muted annotation for a reference line (kept out of the legend)."""
    ax.text(x, y, text, ha=ha, va=va, fontsize=6.5, color=MUTED)

C_RATES = (0.5, 1.0, 2.0, 3.0, 5.0)
SOC_REF = 0.5

# ======================================================================
log("=" * 72)
log(" 1. CONCENTRATION PROFILES  (Fig. 1)  -- 1C, SOC = 0.1/0.25/0.5/0.75/1.0")
log("=" * 72)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

for i, (ax, mat) in enumerate(zip(axes, (GRAPHITE, SILICON))):
    solver = DiffusionSolver(mat, N=200)
    t, hist = solver.run(c_rate=1.0, n_steps=2000)
    for soc, color in zip((0.10, 0.25, 0.50, 0.75, 1.0), SOC_RAMP):
        c = solver.profile_at_soc(t, hist, soc)
        ax.plot(
            solver.r_c / mat.R, c / mat.c_max, color=color, lw=1.2,
            label=f"SOC = {soc:.2f}",
        )
    ax.set_xlabel(r"$r/R$")
    ax.set_xlim(0.0, 1.0)
    ax.set_title( f"({chr(97 + i)}) {mat.name.capitalize()} ($R$ = {mat.R * 1e6:g} µm, 1C)", loc="left")
    ax.set_ylim(-0.03, None)

axes[0].set_ylabel(r"$c/c_{\max}$")
axes[1].set_ylim(-0.03, 2.6)  # reserved band so the legend clears the curves
axes[1].legend(loc="upper left", handlelength=1.4, labelspacing=0.3)
savefig(fig, "fig1_concentration_profiles.png")

# headline numbers for text
for mat in (GRAPHITE, SILICON):
    solver = DiffusionSolver(mat, N=200)
    t, hist = solver.run(c_rate=1.0, n_steps=2000)
    c25 = solver.profile_at_soc(t, hist, 0.25)
    log(
        f"  {mat.name:9s} SOC=0.25: c_surf/c_max = {c25[-1] / mat.c_max:.3f}, "
        f"c_center/c_max = {c25[0] / mat.c_max:.3f}, surf/center = "
        f"{(c25[-1] / c25[0]) if c25[0] > 0 else float('inf'):.2f}"
    )

# ======================================================================
log()
log("=" * 72)
log(" 2. STRESS PROFILES AT 1C, SOC=0.5  (Fig. 2)")
log("=" * 72)
fig = plt.figure(figsize=(7.0, 4.6))
gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.55)
axes = [
    fig.add_subplot(gs[0, 0:2]),
    fig.add_subplot(gs[0, 2:4]),
    fig.add_subplot(gs[1, 1:3]),  # (c) centered on the second row
]

# graphite elastic
sv_g     = DiffusionSolver(GRAPHITE, N=400)
t_g, h_g = sv_g.run(1.0, 2000)

c_g        = sv_g.profile_at_soc(t_g, h_g, SOC_REF)
sr_g, st_g = elastic_stress(GRAPHITE, sv_g.r_c, c_g)

axes[0].plot(sv_g.r_c / GRAPHITE.R, st_g / 1e6, color=BLUE, label=r"$\sigma_\theta$")
axes[0].plot(sv_g.r_c / GRAPHITE.R, sr_g / 1e6, color=ORANGE, label=r"$\sigma_r$")
axes[0].set_title(f"(a) Graphite, elastic (peak {np.max(st_g) / 1e6:.1f} MPa)", loc="left")

log(
    f"  Graphite : peak tensile hoop = {np.max(st_g) / 1e6:.2f} MPa at r/R = "
    f"{sv_g.r_c[np.argmax(st_g)] / GRAPHITE.R:.4f}; surf hoop = {st_g[-1] / 1e6:.2f} MPa"
)

# silicon raw elastic (diagnostic)
sv_s     = DiffusionSolver(SILICON, N=100)
t_s, h_s = sv_s.run(1.0, 1500)

c_s        = sv_s.profile_at_soc(t_s, h_s, SOC_REF)
sr_s, st_s = elastic_stress(SILICON, sv_s.r_c, c_s)

axes[1].plot(sv_s.r_c / SILICON.R, st_s / 1e9, color=BLUE, label=r"$\sigma_\theta$")
axes[1].plot(sv_s.r_c / SILICON.R, sr_s / 1e9, color=ORANGE, label=r"$\sigma_r$")
axes[1].axhline(SILICON.E / 1e9, ls=(0, (4, 3)), c=MUTED, lw=0.7)
axes[1].axhline(-SILICON.E / 1e9, ls=(0, (4, 3)), c=MUTED, lw=0.7)

ref_label(axes[1], 0.02, 94, r"$+E_{Si}$", ha="left")
ref_label(axes[1], 0.02, -94, r"$-E_{Si}$", ha="left", va="top")
axes[1].set_title(f"(b) Si, raw elastic (peak {np.max(np.abs(st_s)) / 1e9:.0f} GPa)", loc="left")

log(
    f"  Si raw   : peak |hoop| = {np.max(np.abs(st_s)) / 1e9:.2f} GPa (E = {SILICON.E / 1e9:.0f} GPa); "
    f"center = {st_s[0] / 1e9:.2f}, surf = {st_s[-1] / 1e9:.2f} GPa"
)

# silicon corrected core-shell
res = plastic_core_shell_stress(SILICON, sv_s.r_c, c_s)
axes[2].plot(sv_s.r_c / SILICON.R, res["sigma_theta"] / 1e9, color=BLUE)
axes[2].plot(sv_s.r_c / SILICON.R, res["sigma_r"] / 1e9, color=ORANGE)
axes[2].axvline(res["r_p"] / SILICON.R, ls=(0, (1, 2)), c=MUTED, lw=0.8)
axes[2].annotate(
    r"$r_p$", xy=(res["r_p"] / SILICON.R, 0.90),
    xycoords=("data", "axes fraction"), color=MUTED, fontsize=7,
)

axes[2].axhline(SILICON.sigma_Y / 1e9, ls=(0, (4, 3)), c=MUTED, lw=0.7)
axes[2].axhline(-SILICON.sigma_Y / 1e9, ls=(0, (4, 3)), c=MUTED, lw=0.7)

ref_label(axes[2], 0.99, 1.55, r"$+\sigma_Y$")
ref_label(axes[2], 0.02, -1.58, r"$-\sigma_Y$", ha="left", va="top")

axes[2].set_title(f"(c) Si, core-shell ($r_p/R$ = {res['r_p'] / SILICON.R:.2f})", loc="left")
log(
    f"  Si corr  : r_p/R = {res['r_p'] / SILICON.R:.3f}, center hoop = "
    f"{np.max(res['sigma_theta']) / 1e9:.3f} GPa, surf hoop = {res['sigma_theta'][-1] / 1e9:.3f} GPa"
)

for ax in axes:
    ax.set_xlabel(r"$r/R$")
    ax.set_xlim(0.0, 1.0)

axes[0].set_ylabel("stress (MPa)")
axes[1].set_ylabel("stress (GPa)")
axes[2].set_ylabel("stress (GPa)")

lo, hi = axes[0].get_ylim()
axes[0].set_ylim(lo * 1.1, hi * 1.5)  # reserved band for the legend
axes[0].legend(loc="upper right")
savefig(fig, "fig2_stress_profiles.png")

# ======================================================================
log()
log("=" * 72)
log(" 3. C-RATE SWEEP AT SOC=0.5  (Table 2 / Fig. 3)")
log("=" * 72)
rows = []
for cr in C_RATES:
    sv   = DiffusionSolver(GRAPHITE, N=400)
    t, h = sv.run(cr, 2000)

    c     = sv.profile_at_soc(t, h, SOC_REF)
    _, st = elastic_stress(GRAPHITE, sv.r_c, c)

    svs    = DiffusionSolver(SILICON, N=100)
    ts, hs = svs.run(cr, 1500)

    cs = svs.profile_at_soc(ts, hs, SOC_REF)
    r  = plastic_core_shell_stress(SILICON, svs.r_c, cs)

    rows.append(
        (
            cr,
            np.max(st) / 1e6,
            np.max(r["sigma_theta"]) / 1e9,
            r["sigma_theta"][-1] / 1e9,
            r["r_p"] / SILICON.R,
            r["yielded"],
        )
    )
log(
    "  C-rate | graphite peak (MPa) | Si center (GPa) | Si surf (GPa) | r_p/R | yielded"
)

for cr, g, sctr, ssrf, rp, y in rows:
    log(f"   {cr:>4}C | {g:>19.2f} | {sctr:>15.3f} | {ssrf:>12.3f} | {rp:>5.3f} | {y}")

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
crs = [r[0] for r in rows]

axes[0].plot(
    crs, [r[1] for r in rows], "o-", color=BLUE, ms=3.5,
    markeredgecolor="white", markeredgewidth=0.5,
)

axes[0].set_xlabel("C-rate")
axes[0].set_ylabel(r"peak $\sigma_\theta$ (MPa)")
axes[0].set_title("(a) Graphite (elastic)", loc="left")
axes[1].plot(
    crs, [r[2] for r in rows], "o-", color=BLUE, ms=3.5,
    markeredgecolor="white", markeredgewidth=0.5, label="center (tensile)",
)

axes[1].plot(
    crs, [r[3] for r in rows], "s--", color=ORANGE, ms=3.5, lw=1.1,
    markeredgecolor="white", markeredgewidth=0.5, label="surface",
)

axes[1].axhline(-SILICON.sigma_Y / 1e9, ls=(0, (4, 3)), c=MUTED, lw=0.7)
axes[1].set_ylim(-1.85, 5.0)  # reserved band for the legend

ref_label(axes[1], 0.55, -1.56, r"$-\sigma_Y$", ha="left", va="top")

axes[1].set_xlabel("C-rate")
axes[1].set_ylabel(r"$\sigma_\theta$ (GPa)")
axes[1].set_title("(b) Silicon (core-shell)", loc="left")
axes[1].legend(loc="upper right")
savefig(fig, "fig3_crate_sweep.png")

# ======================================================================
log()
log("=" * 72)
log(" 4. GRIFFITH SCREENING / CRITICAL RADIUS  (Fig. 4)")
log("=" * 72)


def g_curve(mat, N, n_steps=800):
    Rs = np.geomspace(*((1e-6, 50e-6) if mat is GRAPHITE else (50e-9, 3e-6)), 60)
    pk = np.array(
        [peak_hoop_stress(mat, R, 1.0, SOC_REF, N=N, n_steps=n_steps) for R in Rs]
    )
    return Rs, pk


fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
for i, (ax, mat, N) in enumerate(((axes[0], GRAPHITE, 400), (axes[1], SILICON, 100))):
    scale = 1e6 if mat is GRAPHITE else 1e9
    unit  = "µm" if mat is GRAPHITE else "nm"

    lo, hi = (1e-6, 50e-6) if mat is GRAPHITE else (50e-9, 3e-6)
    Rs, pk = g_curve(mat, N)

    sf = np.array([griffith_fracture_stress(mat.K_IC, R, 0.1) for R in Rs])
    ax.plot(Rs * scale, pk / scale, color=BLUE, label=r"peak $\sigma_\theta$")
    ax.plot(
        Rs * scale, sf / scale, ls=(0, (4, 3)), lw=1.1,
        color=ORANGE, label=r"$\sigma_f(R)$",
    )
    ax.fill_between(
        Rs * scale, pk / scale, sf / scale, where=pk > sf,
        interpolate=True, color="#e34948", alpha=0.07, lw=0,
    )

    Rc = find_critical_radius(mat, 1.0, SOC_REF, R_bracket=(lo, hi))
    ax.axvline(Rc * scale, ls=(0, (1, 2)), c=MUTED, lw=0.8)
    ax.annotate(
        r"$R_{crit}$", xy=(Rc * scale, 0.93),
        xycoords=("data", "axes fraction"), ha="center",
        color=MUTED, fontsize=7,
    )

    ax.text(
        0.97, 0.30, "fracture", transform=ax.transAxes, ha="right",
        color=MUTED, fontsize=6.5, style="italic",
    )

    ax.set_xlabel(f"$R$ ({unit})")
    ax.set_ylabel("stress (MPa)" if mat is GRAPHITE else "stress (GPa)")
    ax.set_title(
        f"({chr(97 + i)}) {mat.name.capitalize()}: "
        f"$R_{{crit}}$ = {Rc * scale:.1f} {unit}", loc="left",
    )

    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.28)  # reserved band for the legend
    ax.legend(loc="upper right")
    log(
        f"  {mat.name:9s} R_crit = {Rc * 1e9:.1f} nm ({Rc * 1e6:.2f} um) at 1C, SOC=0.5"
    )
savefig(fig, "fig4_critical_radius.png")

# silicon peak-stress curve shape diagnostics (elastic -> plastic transition)
lo, hi = 50e-9, 3e-6

Rs = np.geomspace(lo, hi, 60)
pk = np.array(
    [peak_hoop_stress(SILICON, R, 1.0, SOC_REF, N=100, n_steps=800) for R in Rs]
)
yld = []
for R in Rs:
    m = AnodeMaterials(**{**SILICON.__dict__, "R": R})
    sv = DiffusionSolver(m, N=100)
    t, h = sv.run(1.0, 800)
    cc = sv.profile_at_soc(t, h, SOC_REF)
    yld.append(plastic_core_shell_stress(m, sv.r_c, cc)["yielded"])
i0 = next((i for i, y in enumerate(yld) if y), None)
if i0 is not None:
    log(
        f"  Si first-yield radius in scan: between {Rs[max(i0 - 1, 0)] * 1e9:.0f} and "
        f"{Rs[i0] * 1e9:.0f} nm"
    )

# ======================================================================
log()
log("=" * 72)
log(" 5. CONVERGENCE STUDY  (Fig. 5)")
log("=" * 72)
# Si uncapped surface peak vs N and n_steps
Ns = (25, 50, 100, 200, 400)
pN = []
for N in Ns:
    sv = DiffusionSolver(SILICON, N=N)
    t, h = sv.run(1.0, 1500)
    c = sv.profile_at_soc(t, h, SOC_REF)
    _, st = elastic_stress(SILICON, sv.r_c, c)
    pN.append(np.max(np.abs(st)) / 1e9)
log(
    "  Si uncapped peak |sigma_theta| vs N: "
    + ", ".join(f"{N}:{v:.2f}" for N, v in zip(Ns, pN))
)
steps = (250, 500, 1000, 2000, 4000)
pT = []
for ns in steps:
    sv = DiffusionSolver(SILICON, N=100)
    t, h = sv.run(1.0, ns)
    c = sv.profile_at_soc(t, h, SOC_REF)
    _, st = elastic_stress(SILICON, sv.r_c, c)
    pT.append(np.max(np.abs(st)) / 1e9)
log(
    "  Si uncapped peak vs n_steps: "
    + ", ".join(f"{ns}:{v:.2f}" for ns, v in zip(steps, pT))
)
# graphite center peak vs N
gN = []
for N in Ns:
    sv = DiffusionSolver(GRAPHITE, N=N)
    t, h = sv.run(1.0, 2000)
    c = sv.profile_at_soc(t, h, SOC_REF)
    _, st = elastic_stress(GRAPHITE, sv.r_c, c)
    gN.append(np.max(st) / 1e6)
log("  Graphite center peak vs N: " + ", ".join(f"{N}:{v:.2f}" for N, v in zip(Ns, gN)))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
# normalized deviation from the finest-grid / finest-step converged value
axes[0].plot(
    Ns, np.abs(np.array(pN) / pN[-1] - 1), "o-", color=ORANGE, ms=3.5,
    markeredgecolor="white", markeredgewidth=0.5, label="silicon",
)
axes[0].plot(
    Ns, np.abs(np.array(gN) / gN[-1] - 1), "s-", color=BLUE, ms=3.5,
    markeredgecolor="white", markeredgewidth=0.5, label="graphite",
)
axes[0].set_yscale("log")
axes[0].set_xlabel("N (radial cells)")
axes[0].set_ylabel(r"$|\mathrm{peak/converged} - 1|$")
axes[0].axhline(0.01, ls=(0, (1, 2)), c=MUTED, lw=0.7)

ref_label(axes[0], 26, 0.0105, "1%", ha="left")

axes[0].set_title("(a) radial refinement", loc="left")
axes[0].legend(loc="upper right")  # sits in the empty band above the curves

ax2 = axes[1]
ax2.plot(
    steps, np.abs(np.array(pT) / pT[-1] - 1), "o-", color=ORANGE, ms=3.5,
    markeredgecolor="white", markeredgewidth=0.5,
)
ax2.set_yscale("log")
ax2.set_xlabel("time steps")
ax2.set_ylabel(r"$|\mathrm{peak/converged} - 1|$")
ax2.axhline(0.01, ls=(0, (1, 2)), c=MUTED, lw=0.7)
ref_label(ax2, 260, 0.0105, "1%", ha="left")
ax2.set_title("(b) temporal refinement (Si)", loc="left")
savefig(fig, "fig5_convergence.png")

# ======================================================================
log()
log("=" * 70)
log(" 6. SENSITIVITY OF R_crit (Si)  (Fig. 6)")
log("=" * 70)
BR = (50e-9, 3e-6)
base = find_critical_radius(SILICON, 1.0, SOC_REF, R_bracket=BR)
log(f"  baseline R_crit = {base * 1e9:.0f} nm")
sens = []
for tag, kw in (
    ("-20% E", {"E": 0.8 * 90e9}),
    ("+20% E", {"E": 1.2 * 90e9}),
    ("-20% omega", {"omega": 0.8 * 1.08e-5}),
    ("+20% omega", {"omega": 1.2 * 1.08e-5}),
    ("-20% sigma_Y", {"sigma_Y": 0.8 * 1.5e9}),
    ("+20% sigma_Y", {"sigma_Y": 1.2 * 1.5e9}),
    ("-20% K_IC", {"K_IC": 0.8 * 0.7e6}),
    ("+20% K_IC", {"K_IC": 1.2 * 0.7e6}),
    ("-20% D", {"D": 0.8 * 1e-16}),
    ("+20% D", {"D": 1.2 * 1e-16}),
):
    m = AnodeMaterials(**{**SILICON.__dict__, **kw})
    try:
        Rc = find_critical_radius(m, 1.0, SOC_REF, R_bracket=BR)
        sens.append((tag, Rc * 1e9))
        log(f"   {tag:>12s} -> {Rc * 1e9:6.0f} nm")
    except RuntimeError as e:
        sens.append((tag, np.nan))
        log(f"   {tag:>12s} -> no crossover ({str(e)[:60]}...)")
a_sens = []
for af in (0.05, 0.1, 0.2):
    Rc = find_critical_radius(SILICON, 1.0, SOC_REF, a_frac=af, R_bracket=BR)
    a_sens.append((f"a={af}R", Rc * 1e9))
    log(f"   {'a=' + str(af) + 'R':>12s} -> {Rc * 1e9:6.0f} nm")

fig, ax = plt.subplots(figsize=(7.0, 2.9))
labels  = [s[0] for s in sens] + [a[0] for a in a_sens]
vals    = [s[1] for s in sens] + [a[1] for a in a_sens]
xticks  = list(range(len(sens))) + [len(sens) + 0.7 + k for k in range(len(a_sens))]
colors  = [ORANGE if "a=" in lbl else BLUE for lbl in labels]
ax.bar(xticks, vals, width=0.62, color=colors, edgecolor="white", linewidth=0.6)
for x, v in zip(xticks, vals):
    if np.isfinite(v):
        ax.text(x, v + 8, f"{v:.0f}", ha="center", fontsize=6.5, color=INK2)
ax.axhline(base * 1e9, ls=(0, (4, 3)), c=MUTED, lw=0.8)
ax.set_xticks(xticks)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
ax.set_ylabel(r"$R_{crit}$ (nm)")
ax.set_ylim(0, max(vals) * 1.30)  # reserved band for the legend
ax.legend(
    handles=[
        Patch(fc=BLUE, label="material parameter ±20%"),
        Patch(fc=ORANGE, label=r"flaw depth $a/R$"),
        Line2D([0], [0], color=MUTED, lw=0.8, ls=(0, (4, 3)),
               label=f"baseline ({base * 1e9:.0f} nm)"),
    ],
    loc="upper left", ncol=3,
)
savefig(fig, "fig6_sensitivity.png")

# ======================================================================
log()
log("Done. Figures written to results/figures/")
