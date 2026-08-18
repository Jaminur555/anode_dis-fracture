"""
anode_dis - diffusion-induced stress (DIS) fracture framework for spherical Li-ion anode
particles: graphite(linear-elastic) and silicon (elastic-plastic).

Pipeline: material -> diffusion -> christensen_newman -> tresca-> fracture
"""

__version__ = "1.0.0"

from .materials import AnodeMaterials, GRAPHITE, SILICON, eigen_strain
from .diffusion import DiffusionSolver
from .christensen_newman import elastic_stress, cumulative_I, I_of_r, c_of_r
from .tresca import plastic_core_shell_stress, check_equilibrium_residual
from .fracture import griffith_fracture_stress, peak_hoop_stress, find_critical_radius

__all__ = [
    "AnodeMaterials",
    "GRAPHITE",
    "SILICON",
    "eigen_strain",
    "DiffusionSolver",
    "elastic_stress",
    "cumulative_I",
    "I_of_r",
    "c_of_r",
    "plastic_core_shell_stress",
    "check_equilibrium_residual",
    "griffith_fracture_stress",
    "peak_hoop_stress",
    "find_critical_radius",
]
