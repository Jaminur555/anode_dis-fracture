"""Linear-elastic Christensen-Newman Stress Solution"""

import numpy as np
from .materials import AnodeMaterials


def cumulative_I_trapz(r_c, c, R):
    """Computes I(r) = integral of (c(r') * r'^2)dr from 0 -> r

    Inputs:
    -------
        r_c : 1D array of cell center radii [m]
        c   : 1D array of cell concentrations [mol/m^3]
        R   : Total Particle radius [m]

    Returns:
    --------
        r_nodes: Array of radial positions including boundary points 0 and R
        I_nodes: Cumulative Integral values corresponding to r_nodes
    """
    N = len(r_c)
    dr = r_c[1] - r_c[0] if N > 1 else R
    r_f = np.arange(N + 1) * dr
    r_f[-1] = R

    shell_I = c * (r_f[1:] ** 3 - r_f[:-1] ** 3) / 3.0
    I_left = np.concatenate(([0.0], np.cumsum(shell_I)[:-1]))
    I_center = I_left + c * (r_c**3 - r_f[:-1] ** 3) / 3.0

    I_tot = I_left[-1] + shell_I[-1]

    r_nodes = np.concatenate(([0.0], r_c, [R]))
    I_nodes = np.concatenate(([0.0], I_center, [I_tot]))
    return r_nodes, I_nodes


def cumulative_I(r_c: np.ndarray, c: np.ndarray, R: float):
    """Cumulative concentration integral I(r) on the finite-volume grid."""

    return cumulative_I_trapz(r_c, c, R)


def elastic_stress(mat: AnodeMaterials, r_c: np.ndarray, c: np.ndarray):
    """Computes closed-form linear elastic radial (sigma_r) and tangential (sigma_theta) stresses."""

    R = mat.R
    r_nodes, I_nodes = cumulative_I(r_c, c, R)
    I_tot = I_nodes[-1]
    I_r = np.interp(r_c, r_nodes, I_nodes)

    pref = 2.0 * mat.E * mat.omega / (3.0 * (1.0 - mat.nu))
    r_safe = np.maximum(r_c, 1e-30)

    sigma_r = pref * (I_tot / R**3 - I_r / np.maximum(r_c, 1e-30) ** 3)
    sigma_theta = 0.5 * pref * (2.0 * I_tot / (R**3) + I_r / (r_safe**3) - c)

    return sigma_r, sigma_theta


def I_of_r(r, r_nodes: np.ndarray, I_nodes: np.ndarray) -> float:
    """Interpolates cumulative integral I(r) at radial position r."""

    return np.interp(r, r_nodes, I_nodes)


def c_of_r(r, r_c: np.ndarray, c: np.ndarray, R: float):
    """Interpolates concentration c(r) at radial position r."""

    r_ext = np.concatenate(([0.0], r_c, [R]))
    c_ext = np.concatenate(([c[0]], c, [c[-1]]))
    return np.interp(r, r_ext, c_ext)
