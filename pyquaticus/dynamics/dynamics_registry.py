from pyquaticus.dynamics.dynamics import *

"""
Registry to map string representations of dynamics to their actual classes.
To add your own dynamics, create a new class that inherits from Dynamics
and implement _move_agents(), then add it to the registry here

"""

dynamics_registry = {
    "heron": Heron,
    "large_usv": LargeUSV,
    "si": SingleIntegrator,
    "di": DoubleIntegrator,
    "drone": Drone,
    "fixed_wing": FixedWing,
}
