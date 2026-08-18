"""
Automated Verification and Regression Suite.

Two tiers of checks:
  A. Analytic / internal-consistency tests (must hold exactly, up to discretization):
     these are regression-proof -- any change to the stress algebra (e.g. a wrong
     prefactor in the Tresca yield equation) breaks them.
  B. Frozen-result benchmarks: values produced by the verified code, kept with
     tolerances so silent behavioral drift is caught. These are NOT literature
     validations by themselves; literature anchors are cited in the paper text.
"""

import numpy as np
from scipy.optimize import brentq

from anode_dis import (GRAPHITE, SILICON, AnodeMaterials, DiffusionSolver,
    elastic_stress, cumulative_I, I_of_r, c_of_r,
    plastic_core_shell_stress, check_equilibrium_residual,
    find_critical_radius,)


PASS, FAIL = [], []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    (PASS if cond else FAIL).append(name)
    print(f" [{status}] {name}" + (f"  ({detail})" if detail else ""))


def si_case(N=100, c_rate=1.0, soc=0.5, n_steps=1500):
    solver = DiffusionSolver(SILICON, N=N)
    t, hist = solver.run(c_rate=c_rate, n_steps=n_steps)
    c = solver.profile_at_soc(t, hist, soc_target=soc)
    return solver, c


def sphere_flux_series(r, t, R, D, j_n, n_modes=300):
    """
    Analytic solution of  c_t = (D/r^2)(r^2 c_r)_r,  c(r,0) = 0, symmetry at r = 0
    -D c_r(R) = j_n (constant inward flux), :
    c = (j_n R/D) [ 3*tau + (rho^2/2 - 3/10)
                    - sum_n 2/(mu_n^2 cos mu_n) * sin(mu_n rho)/(mu_n rho)
                          * exp(-mu_n^2 tau) ],
    tau = D t / R^2, rho = r/R, mu_n = positive roots of tan(mu) = mu.
    """

    def f(m):
        return np.tan(m) - m

    mus = np.array(
        [
            brentq(f, n * np.pi + 1e-9, n * np.pi + np.pi / 2 - 1e-9)
            for n in range(1, n_modes + 1)
        ]
    )

    tau, rho = D * t / R**2, r / R
    shape = np.sinc(np.outer(mus, rho) / np.pi)  # sin(mu r)/(mu r)
    amp = 2.0 / (mus**2 * np.cos(mus))
    series = (amp[:, None] * shape * np.exp(-(mus[:, None] ** 2) * tau)).sum(axis=0)
    return (j_n * R / D) * (3.0 * tau + 0.5 * rho**2 - 0.3 - series)


# ======================================================================
# A1. Christensen-Newman closed-form: manufactured solutions
# ======================================================================


def test_cn_manufactured():
    print("==========================================================")
    print(" A1: Christensen-Newman vs analytic manufactured solutions")
    print("==========================================================")

    m = AnodeMaterials(
        name="test",
        R=1e-6,
        D=1e-14,
        E=15e9,
        nu=0.3,
        omega=3.17e-6,
        c_max=26390.0,
        K_IC=0.5e6,
    )
    N = 400
    r_c = (np.arange(N) + 0.5) / N * m.R
    pref = 2 * m.E * m.omega / (3 * (1 - m.nu))

    # (a) uniform concentration -> exactly stress-free
    c0 = 1e4
    sr, st = elastic_stress(m, r_c, np.full(N, c0))

    check(
        "uniform c: sigma = 0",
        max(np.max(np.abs(sr)), np.max(np.abs(st))) < 1e-3,
        f"max|sigma| = {max(np.max(np.abs(sr)), np.max(np.abs(st))):.2e} Pa",
    )

    # (b) linear profile c(r) = c1 * r/R -> closed form
    c1 = 2e4
    c_lin = c1 * r_c / m.R
    sr, st = elastic_stress(m, r_c, c_lin)
    sr_ex = pref * (c1 / 4 - c1 * r_c / (4 * m.R))
    st_ex = 0.5 * pref * (2 * c1 / 4 + c1 * r_c / (4 * m.R) - c_lin)
    scale = np.max(np.abs(st_ex))

    check(
        "linear c: sigma_r matches analytic",
        np.max(np.abs(sr - sr_ex)) / scale < 1e-3,
        f"rel err = {np.max(np.abs(sr - sr_ex)) / scale:.2e}",
    )
    check(
        "linear c: sigma_theta matches analytic",
        np.max(np.abs(st - st_ex)) / scale < 1e-3,
        f"rel err = {np.max(np.abs(st - st_ex)) / scale:.2e}",
    )

    # (c) traction-free surface: sigma_r(R) = 0 exactly by construction
    #     (I_r = I_tot at r = R). At the last cell center (r = R - dr/2) the
    #     residual is O(dr): verify it shrinks with grid refinement.

    sr_n100, _ = elastic_stress(
        m,
        (np.arange(100) + 0.5) / 100 * m.R,
        c_lin[:100] * 0 + c1 * ((np.arange(100) + 0.5) / 100),
    )
    sr_n400, _ = elastic_stress(m, r_c, c_lin)

    check(
        "surface sigma_r residual is O(dr) (shrinks 4x under refinement)",
        abs(sr_n400[-1]) < abs(sr_n100[-1]) / 3,
        f"|sigma_r(R-dr/2)|: N=100 {abs(sr_n100[-1]):.1e} -> N=400 {abs(sr_n400[-1]):.1e} Pa",
    )


# ======================================================================
#     A2. Diffusion solver: exact mass balance under galvanostatic flux
# ======================================================================


def test_mass_conservation():
    print("==========================================================")
    print(" A2: Diffusion solver mass conservation (galvanostatic)")
    print("==========================================================")

    for mat, cr in ((GRAPHITE, 1.0), (SILICON, 1.0), (SILICON, 5.0)):
        s = DiffusionSolver(mat, N=100)
        t, hist = s.run(c_rate=cr, n_steps=1500)
        j_n = mat.c_max * mat.R * cr / (3 * 3600)
        soc_exact = j_n * 3 * t / (mat.c_max * mat.R)
        err = np.max(np.abs(s.soc_history(hist) - soc_exact))

        check(
            f"{mat.name} {cr}C: SOC = exact mass balance",
            err < 1e-10,
            f"max err = {err:.2e}",
        )


# ======================================================================
#       A2b. Diffusion solver vs analytic series (radial SHAPE check)
# ======================================================================


def test_diffusion_series():
    print("==========================================================")
    print(" A2b: profile shape vs analytic series (constant flux)")
    print("==========================================================")

    # (a) absolute agreement in the production configuration
    for mat in (GRAPHITE, SILICON):
        s = DiffusionSolver(mat, N=400)
        t, hist = s.run(c_rate=1.0, n_steps=1500)
        j_n = mat.c_max * mat.R / (3 * 3600)  # 1C flux
        c_ex = sphere_flux_series(s.r_c, t[-1], mat.R, mat.D, j_n)
        err = np.linalg.norm(hist[-1] - c_ex) / np.linalg.norm(c_ex)

        check(
            f"{mat.name} 1C: profile matches analytic series",
            err < 5e-3,
            f"rel L2 = {err:.2e}",
        )

    # (b) spatial convergence order on graphite (smooth profile): expect ~O(dr^2)
    errs = []
    for N in (50, 200):
        s = DiffusionSolver(GRAPHITE, N=N)
        t, hist = s.run(c_rate=1.0, n_steps=1500)
        j_n = GRAPHITE.c_max * GRAPHITE.R / (3 * 3600)
        c_ex = sphere_flux_series(s.r_c, t[-1], GRAPHITE.R, GRAPHITE.D, j_n)

        errs.append(np.linalg.norm(hist[-1] - c_ex) / np.linalg.norm(c_ex))

    p = np.log(errs[0] / errs[1]) / np.log(4.0)
    check(
        "graphite: error under 4x refinement",
        errs[1] < errs[0] / (4**1.5),
        f"err {errs[0]:.2e} (N=50) -> {errs[1]:.2e} (N=200), order ~ {p:.2f}",
    )


# ======================================================================
# A3. Tresca core-shell: internal consistency (regression-proof tier)
# ======================================================================


def test_tresca_consistency():
    print("==========================================================")
    print(" A3: Elastic-plastic core-shell internal consistency")
    print("==========================================================")

    solver, c = si_case()
    res = plastic_core_shell_stress(SILICON, solver.r_c, c)
    pref = 2 * SILICON.E * SILICON.omega / (3 * (1 - SILICON.nu))

    r_nodes, I_nodes = cumulative_I(solver.r_c, c, SILICON.R)

    check("silicon yields at 1C, SOC=0.5", res["yielded"])

    # (a) yield continuity at r_p: shell side is s*sigma_Y by construction;
    #     the core side must equal it too -- this is the test that catches a
    #     wrong prefactor in the yield equation (e.g. missing factor of 2).

    r_p, s = res["r_p"], res["s"]
    I_rp = I_of_r(r_p, r_nodes, I_nodes)
    c_rp = c_of_r(r_p, solver.r_c, c, SILICON.R)

    tresca_at_rp = 0.5 * pref * (3 * I_rp / r_p**3 - c_rp)

    check(
        "Tresca condition exactly met at r_p",
        abs(tresca_at_rp - s * SILICON.sigma_Y) / SILICON.sigma_Y < 1e-6,
        f"(st-sr)(r_p) = {tresca_at_rp / 1e9:.4f} GPa vs s*sigma_Y = "
        f"{s * SILICON.sigma_Y / 1e9:.4f} GPa",
    )

    # (b) shell stresses satisfy sigma_theta - sigma_r = s*sigma_Y everywhere
    shell = solver.r_c >= r_p
    d = res["sigma_theta"][shell] - res["sigma_r"][shell]

    check(
        "shell: sigma_theta - sigma_r = s*sigma_Y",
        np.max(np.abs(d - s * SILICON.sigma_Y)) / SILICON.sigma_Y < 1e-12,
    )

    # (c) shell hoop approaches s*sigma_Y at the surface: exactly at r = R,
    #     traction-free + Tresca gives sigma_theta(R) = s*sigma_Y; at the last
    #     cell center (r = R - dr/2) the shell formula deviates by O(dr/R).

    check(
        "surface hoop -> s*sigma_Y (last cell, O(dr/R))",
        abs(res["sigma_theta"][-1] - s * SILICON.sigma_Y) / SILICON.sigma_Y < 0.03,
        f"sigma_theta(r_last) = {res['sigma_theta'][-1] / 1e9:.4f} GPa vs "
        f"s*sigma_Y = {s * SILICON.sigma_Y / 1e9:.4f} GPa",
    )
    # (d) radial stress continuity across r_p
    p_bt = -2 * s * SILICON.sigma_Y * np.log(SILICON.R / r_p)
    check(
        "sigma_r continuous at r_p (core uses shell traction)",
        abs(
            res["sigma_r"][~shell][-1]
            - (
                pref
                * (
                    I_of_r(r_p, r_nodes, I_nodes) / r_p**3
                    - I_of_r(solver.r_c[~shell][-1], r_nodes, I_nodes)
                    / solver.r_c[~shell][-1] ** 3
                )
                + p_bt
            )
        )
        < 1e6,
        "residual < 1 MPa",
    )

    # (e) discrete equilibrium residual small away from interface/axis
    eq = check_equilibrium_residual(
        SILICON, solver.r_c, res["sigma_r"], res["sigma_theta"], r_p=r_p
    )
    check("spherical equilibrium residual < 1%", eq < 0.01, f"residual = {eq:.2e}")


# ======================================================================
# B. Frozen-result benchmarks (verified code, current parameters)
# ======================================================================


def benchmarks():
    print("==========================================================")
    print(" B: Frozen benchmarks (graphite 1C / silicon 1C, SOC=0.5)")
    print("==========================================================")

    # Graphite linear-elastic peak (grid-converged at N=400)
    solver = DiffusionSolver(GRAPHITE, N=400)
    t, hist = solver.run(c_rate=1.0, n_steps=2000)

    c = solver.profile_at_soc(t, hist, soc_target=0.5)
    sr, st = elastic_stress(GRAPHITE, solver.r_c, c)
    peak = np.max(st) / 1e6

    print(f" Graphite peak tensile hoop   : {peak:.2f} MPa")
    print(f" Graphite surface sigma_r     : {sr[-1] / 1e6:.4f} MPa")

    # traction-free at r=R exactly by construction; last cell center is O(dr) away
    check(
        "graphite surface traction-free (last cell < 0.1 MPa)",
        abs(sr[-1]) / 1e6 < 0.1,
        f"{sr[-1] / 1e6:.4f} MPa",
    )
    check("graphite peak hoop ~ 18.2 MPa", 17.5 <= peak <= 19.0, f"{peak:.2f} MPa")
    check("graphite peak at center (tensile core)", st[0] > 0 and st[-1] < 0)

    # Silicon uncapped diagnostic (small-strain breakdown)
    solver, c = si_case()
    _, st_unc = elastic_stress(SILICON, solver.r_c, c)
    gpa = np.max(np.abs(st_unc)) / 1e9

    print(
        f" Si uncapped peak magnitude  : {gpa:.2f} GPa (E = {SILICON.E / 1e9:.0f} GPa)"
    )
    check("Si uncapped stress exceeds E (breakdown diagnostic)", gpa > SILICON.E / 1e9)

    # Silicon corrected core-shell
    res = plastic_core_shell_stress(SILICON, solver.r_c, c)
    rp, ctr, srf = (
        res["r_p"] / SILICON.R,
        np.max(res["sigma_theta"]) / 1e9,
        res["sigma_theta"][-1] / 1e9,
    )

    print(
        f" Si corrected: r_p/R = {rp:.4f}, center = {ctr:.3f} GPa, surface = {srf:.3f} GPa"
    )

    check("Si r_p/R ~ 0.52", 0.51 <= rp <= 0.53, f"{rp:.4f}")
    check(
        "Si center hoop ~ 2.42 GPa (> sigma_Y)", 2.35 <= ctr <= 2.49, f"{ctr:.3f} GPa"
    )
    check(
        "Si surface hoop pinned near -sigma_Y", abs(srf + 1.5) < 0.02, f"{srf:.3f} GPa"
    )

    # Critical radii
    Rg = find_critical_radius(GRAPHITE, C_rate=1.0, soc=0.5, R_bracket=(1e-6, 50e-6))
    Rs = find_critical_radius(SILICON, C_rate=1.0, soc=0.5, R_bracket=(50e-9, 3e-6))

    print(f" Graphite R_crit = {Rg * 1e6:.2f} um | Si R_crit = {Rs * 1e9:.0f} nm")

    check(
        "graphite R_crit ~ 27.5 um (safe zone > 15 um)",
        26.0 <= Rg * 1e6 <= 29.0,
        f"{Rg * 1e6:.2f} um",
    )
    check(
        "Si R_crit ~ 331 nm (near exp. 150-300 nm)",
        310 <= Rs * 1e9 <= 350,
        f"{Rs * 1e9:.0f} nm",
    )


if __name__ == "__main__":
    print("\nRunning Verification & Regression Suite...\n")
    test_cn_manufactured()
    test_mass_conservation()
    test_diffusion_series()
    test_tresca_consistency()
    benchmarks()
    print("\n==========================================================")
    print(f" RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print(" Failed checks:")
        for f in FAIL:
            print(f"   - {f}")
    print("==========================================================")
    raise SystemExit(0 if not FAIL else 1)
