"""
Griffith Fracture Criterion & Critical Particle Radius Solver.
"""

from dataclasses import replace
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from .materials import AnodeMaterials
from .diffusion import DiffusionSolver
from .christensen_newman import elastic_stress
from .tresca import plastic_core_shell_stress


def griffith_fracture_stress(K_IC: float, R: float, a_frac: float = 0.1):
    """
    Computes Griffith surface fracture stress for pre-existing flaw depth a = a_frac * R.
    Formula: sigma_f = K_IC / sqrt(a_frac * pi * R)
    """
    return K_IC / np.sqrt(a_frac * np.pi * R)


def peak_hoop_stress(
    mat: AnodeMaterials,
    R_override: float,
    C_rate: float,
    soc: float,
    N: int = 100,
    n_steps: int = 1500,
    use_plastic: bool = True,
) -> float:
    """
    Runs the diffusion solver for a particle of radius R_override at a given C_rate, extracts
    concentration profile at target SOC, computes stress, and return peak tensile hoop stress
    """
    # Create temporary material instance with modified radius R_override
    m = replace(mat, R=R_override)

    # Run diffusion solver
    solver = DiffusionSolver(m, N=N)
    t, hist = solver.run(C_rate, n_steps=n_steps)
    c = solver.profile_at_soc(t, hist, soc)

    # Evaluate stress field (plastic core-shell if applicable, else linear elastic)
    if m.sigma_Y is not None and use_plastic:
        res = plastic_core_shell_stress(m, solver.r_c, c)
        return float(np.max(res["sigma_theta"]))
    else:
        _, st = elastic_stress(m, solver.r_c, c)
        return float(np.max(st))


def find_critical_radius(
    mat: AnodeMaterials,
    C_rate: float,
    soc: float,
    a_frac: float = 0.1,
    R_bracket=(50e-9, 50e-6),
    N: Optional[int] = None,
    n_steps: int = 800,
) -> float:
    """
    Finds critical particle radius R_crit where peak tensile stress equals fracture stress.
    Returns R_crit in meters.

    N defaults to 400 for linear-elastic materials (their peak hoop stress sits at the
    particle center and is grid-sensitive) and 100 for elastic-plastic ones (surface-
    dominated peak, converged at N=100; see the convergence study in the paper).
    """
    if N is None:
        N = 100 if mat.sigma_Y is not None else 400

    def g(R):
        s_peak = peak_hoop_stress(mat, R, C_rate, soc, N=N, n_steps=n_steps)
        s_frac = griffith_fracture_stress(mat.K_IC, R, a_frac)
        return s_peak - s_frac

    lo_init, hi_init = R_bracket
    Rs = np.geomspace(lo_init, hi_init, 100)
    gs = np.array([g(R) for R in Rs])

    # Find indices where sign changes
    sign_changes = np.where(np.diff(np.sign(gs)) != 0)[0]

    if len(sign_changes) == 0:
        raise RuntimeError(
            f"No Griffith crossover found for {mat.name} in R = "
            f"[{lo_init * 1e9:.0f} nm, {hi_init * 1e6:.1f} um]. "
            f"min|g| = {np.min(np.abs(gs)):.3e} Pa at R = {Rs[np.argmin(np.abs(gs))]:.3e} m. "
            f"Widen R_bracket or revisit parameters."
        )

    # Target the FIRST crossover in the specified bracket
    idx = sign_changes[0]
    lo, hi = Rs[idx], Rs[idx + 1]

    return float(brentq(g, lo, hi, xtol=1e-9))
