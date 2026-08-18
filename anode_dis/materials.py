"""
Dataclass and baseline material properties for Lithium-ion anode particles.
All values are strictly in base SI units.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AnodeMaterials:
    name: str
    R: float  # Particle radius, m
    D: float  # Li diffusivity, m^2/s
    E: float  # Young's modulus, Pa
    nu: float  # Poisson's ratio
    omega: float  # Partial Molar Volume, m^3/mol
    c_max: float  # max Li concentration, mol/ m^3
    K_IC: float  # fracture toughness, Pa.sqrt(m)
    sigma_Y: Optional[float] = (
        None  # Yield (flow) stress, Pa; None => linear-elastic only
    )


# GRAPHITE (Linear-Elastic)
GRAPHITE = AnodeMaterials(
    name="graphite",
    R=8.0e-6,  # 8 um baseline radius
    D=3.9e-14,  # High diffusivity -> flat concentration profile
    E=15.0e9,  # 15 GPa Young's modulus
    nu=0.30,  # Poisson's ratio
    omega=3.17e-6,  # Partial molar volume (m^3/mol)
    c_max=26390.0,  # Max concentration corresponding to LiC6
    K_IC=0.5e6,  # 0.5 MPa*sqrt(m) = 0.5e6 Pa*sqrt(m)
    sigma_Y=None,  # No yield stress defined -> pure linear-elastic
)

#  SILICON (Elastic-Plastic)
SILICON = AnodeMaterials(
    name="silicon",
    R=3.0e-6,  # 3 um baseline radius
    D=1.0e-16,  # Low diffusivity -> steep concentration gradients
    E=90.0e9,  # 90 GPa Young's modulus
    nu=0.22,  # Poisson's ratio
    omega=1.08e-5,  # Partial molar volume (m^3/mol)
    c_max=278000.0,  # Max concentration corresponding to Li3.75Si
    K_IC=0.7e6,  # 0.7 MPa*sqrt(m) = 0.7e6 Pa*sqrt(m)
    sigma_Y=1.5e9,  # Flow stress plateau = 1.5 GPa (Sethuraman et al. 2010)
)


def eigen_strain(mat: AnodeMaterials) -> float:
    """
    Computes total chemical volumetric eigenstrain (Omega * c_max).
    """
    return mat.omega * mat.c_max
