"""
Elastic-plastic (Tresca) core-shell stress correction
"""

from typing import Optional

import numpy as np
from scipy.optimize import brentq

from .materials import AnodeMaterials
from .christensen_newman import cumulative_I, elastic_stress, I_of_r, c_of_r


def _elastic_result(sr_trial, st_trial, s, R):
    """Fallback: particle stays fully elastic (no yield) -- return the trial field."""
    return {
        "r_p": R,
        "sigma_r": sr_trial,
        "sigma_theta": st_trial,
        "s": s,
        "yielded": False,
    }


def plastic_core_shell_stress(mat: AnodeMaterials, r_c, c):
    """Computes elastic-plastic core-shell stress field satisfying Tresca Yield Condition"""
    R = mat.R

    if mat.sigma_Y is None:
        raise ValueError(
            f"{mat.name}: plastic_core_shell_stress requires a yield stress (sigma_Y)"
        )
    sigma_Y = mat.sigma_Y
    pref = 2.0 * mat.E * mat.omega / (3.0 * (1.0 - mat.nu))

    r_nodes, I_nodes = cumulative_I(r_c, c, R)

    # trial elastic solution to get sign of (sigma_theta - sigma_r) at surface
    sr_trial, st_trial = elastic_stress(mat, r_c, c)
    surf_diff = st_trial[-1] - sr_trial[-1]

    s = np.sign(surf_diff)
    if s == 0:
        s = 1.0

    def yield_equation(rp_val: float) -> float:
        rp_safe = max(rp_val, 1e-30)
        I_rp = I_of_r(rp_safe, r_nodes, I_nodes)
        c_rp = c_of_r(rp_safe, r_c, c, R)
        # sigma_theta - sigma_r = [E*Omega/(3(1-nu))] * (3 I_r / r^3 - c(r)) = 0.5 * pref * (...)
        diff = 0.5 * pref * (3.0 * I_rp / (rp_safe**3) - c_rp)
        return diff - s * sigma_Y

    # Check if surface yields at all
    if abs(surf_diff) <= sigma_Y:
        return _elastic_result(sr_trial, st_trial, s, R)

    r_min = 1e-6 * R
    r_max = R * (1.0 - 1e-9)

    f_min = yield_equation(r_min)
    f_max = yield_equation(r_max)

    if np.sign(f_min) == np.sign(f_max):
        return _elastic_result(sr_trial, st_trial, s, R)

    # solve for r_p cleanly using Brent's root-finding method
    try:
        r_p = float(brentq(yield_equation, r_min, r_max, xtol=1e-6 * R))
    except ValueError:
        return _elastic_result(sr_trial, st_trial, s, R)

    # Assemble Core-Shell Stress Field
    sigma_r = np.zeros_like(r_c)
    sigma_theta = np.zeros_like(r_c)

    shell_mask = r_c >= r_p
    r_shell = np.maximum(r_c[shell_mask], 1e-30)

    # Plastic Shell Stresses
    sigma_r[shell_mask] = -2.0 * s * sigma_Y * np.log(R / r_shell)
    sigma_theta[shell_mask] = sigma_r[shell_mask] + s * sigma_Y

    # Elastic Core Stresses
    core_mask = ~shell_mask
    p_bt = -2.0 * s * sigma_Y * np.log(R / r_p)  # Internal traction p = sigma(r_p)

    if np.any(core_mask):
        r_core = r_c[core_mask]
        I_core = I_of_r(r_core, r_nodes, I_nodes)
        I_rp = I_of_r(r_p, r_nodes, I_nodes)
        c_core = c_of_r(r_core, r_c, c, R)
        r_core_safe = np.maximum(r_core, 1e-30)

        sigma_r[core_mask] = pref * (I_rp / (r_p**3) - I_core / (r_core_safe**3)) + p_bt
        sigma_theta[core_mask] = (
            0.5 * pref * (2.0 * I_rp / (r_p**3) + I_core / (r_core_safe**3) - c_core)
            + p_bt
        )

    return {
        "r_p": r_p,
        "s": s,
        "sigma_r": sigma_r,
        "sigma_theta": sigma_theta,
        "yielded": True,
    }


def check_equilibrium_residual(
    mat: AnodeMaterials,
    r_c: np.ndarray,
    sigma_r: np.ndarray,
    sigma_theta: np.ndarray,
    r_p: Optional[float] = None,
) -> float:
    """Calculate the RMS residual error of spherical stress equilibrium.

    Cells adjacent to the elastic-plastic interface r_p (where d(sigma_r)/dr is
    discontinuous by construction) and the first cell at the symmetry axis are
    excluded; np.gradient's one-sided stencils there measure the discretization,
    not the physics.
    """
    dsr_dr = np.gradient(sigma_r, r_c)
    rhs = 2.0 * (sigma_theta - sigma_r) / np.maximum(r_c, 1e-30)
    diff = dsr_dr - rhs

    mask = np.ones_like(diff, dtype=bool)
    mask[0] = False  # axis cell: 1/r singular
    mask[-1] = False  # surface cell: one-sided stencil
    if r_p is not None:
        mask[np.searchsorted(r_c, r_p)] = False  # first cell inside the shell

    # Compute absolute RMS error [Pa/m]
    abs_rms_err = np.sqrt(np.mean(diff[mask] ** 2))

    # Characteristic stress gradient scale: sigma_max / R
    sigma_max = max(np.max(np.abs(sigma_theta)), 1e-30)
    characteristic_scale = sigma_max / mat.R

    return float(abs_rms_err / characteristic_scale)
