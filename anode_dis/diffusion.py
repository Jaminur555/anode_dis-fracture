"""
Implicit Finite-Volume Diffusion Solver For Spherical Electrode Particles.
"""

from typing import Optional
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .materials import AnodeMaterials


class DiffusionSolver:
    def __init__(self, mat: AnodeMaterials, N: int = 100):
        self.mat = mat
        self.N = N
        self.dr = mat.R / N

        # Grid geometry: Cell centers r_c and cell faces r_f
        self.r_c = (np.arange(N) + 0.5) * self.dr  # cell centers
        self.r_f = np.arange(N + 1) * self.dr  # cell faces (0, R)

        # Concentric shell volume for each cell
        self.vol = (4.0 / 3.0) * np.pi * (self.r_f[1:] ** 3 - self.r_f[:-1] ** 3)

    def build_matrix(self, dt: float):
        """
        Builds Backward-Euler system matrix M such that M @ c_new = rhs.
        """
        N, D, dr = self.N, self.mat.D, self.dr

        M = np.zeros((N, N))

        # Internal face areas:
        face_r = self.r_f[1:N]
        face_area = 4.0 * np.pi * face_r**2

        cond = D * face_area / dr

        # Accumulate flux conduction terms across internal cell boundaries
        for i in range(N):
            M[i, i] += self.vol[i] / dt
        for i in range(N - 1):
            g = cond[i]

            M[i, i] += g
            M[i, i + 1] -= g
            M[i + 1, i + 1] += g
            M[i + 1, i] -= g
        return M

    def run(self, c_rate: float, n_steps: int = 1500, t_total: Optional[float] = None):
        """
        Advance the concentration field under galvanostatic surface flux.
        Returns:
        --------
            t    : 1D array of time stamps[s]
            hist : 2D array of concentration profiles of shape (n_steps+1, N).
        """
        mat = self.mat

        # Galvanostatic Neumann flux at particle surface:
        j_n = mat.c_max * mat.R * c_rate / (3.0 * 3600.0)

        if t_total is None:
            t_total = 3600.0 / c_rate
        dt = t_total / n_steps

        # build and LU-factorize matrix M once for performance
        M = self.build_matrix(dt)
        lu, piv = lu_factor(M)

        c = np.zeros(self.N)
        hist = np.zeros((n_steps + 1, self.N))
        hist[0] = c
        t = np.zeros(n_steps + 1)

        boundary_flux = j_n * 4.0 * np.pi * mat.R**2

        for k in range(n_steps):
            rhs = (self.vol / dt) * c
            rhs[-1] += boundary_flux  # Apply Neumann surface flux to cell N-1

            c = lu_solve((lu, piv), rhs)
            hist[k + 1] = c
            t[k + 1] = t[k] + dt

        return t, hist

    def soc_history(self, hist):
        """
        Calculate Volume-averaged state of charge (SOC in [0, 1]) at each saved time step
        """
        total_vol = (4.0 / 3.0) * np.pi * self.mat.R**3
        avg_c = (hist @ self.vol) / total_vol
        return avg_c / self.mat.c_max

    def profile_at_soc(self, t, hist, soc_target):
        """
        Interpolate radial concentration profile at target SOC (e.g. SOC = 0.5).
        """
        soc = self.soc_history(hist)
        idx = np.clip(np.searchsorted(soc, soc_target), 1, len(soc) - 1)

        s0, s1 = soc[idx - 1], soc[idx]
        w = 0.0 if s1 == s0 else (soc_target - s0) / (s1 - s0)
        return (1 - w) * hist[idx - 1] + w * hist[idx]
