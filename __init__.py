# Finch: Evolutionary Algorithm Framework
# Version: 3.5.0

# Core components
from .generic import GenePool, Individual, Layer, Environment, Competition

# Utility functions
from .universal import ARRAY_MANAGER, use_cupy

# Selectors
from .selectors import Select, TournamentSelection, RandomSelection, RankBasedSelection

# Rate-related utilities
from .rates import Rate, make_switcher, make_callable

# Layers
from .layers.float_arrays import (
    FloatPool,
    ParentBlendFloat,
    ParentSimulatedBinaryFloat,
    ParentArithmeticFloat,
    GaussianMutation,
    UniformMutation,
    PolynomialMutation,
    InsertionDeletionMutationFloat
)

from .layers.string_layers import StringPool

from .layers.universal_layers import (
    Populate,
    SortByFitness,
    CapPopulation
)

from .layers.array import (
    ArrayPool,
    ParentNPoint,
    ParentUniform,
    SwapMutation,
    InversionMutation,
    ScrambleMutation,
    ReplaceMutation,
    InsertionDeletionMutation
)

from .layers.image_layers import ImagePool

# Define public API
__all__ = [
    # Core components
    "GenePool", "Individual", "Layer", "Environment", "Competition",

    # Utility functions
    "ARRAY_MANAGER", "use_cupy",

    # Selectors
    "Select", "TournamentSelection", "RandomSelection", "RankBasedSelection",

    # Rate-related utilities
    "Rate", "make_switcher", "make_callable",

    # Float array layers
    "FloatPool", "ParentBlendFloat", "ParentSimulatedBinaryFloat", "ParentArithmeticFloat",
    "GaussianMutation", "UniformMutation", "PolynomialMutation", "InsertionDeletionMutationFloat",

    # String layers
    "StringPool",

    # Universal layers
    "Populate", "SortByFitness", "CapPopulation",

    # Array layers
    "ArrayPool", "ParentNPoint", "ParentUniform", "SwapMutation", "InversionMutation",
    "ScrambleMutation", "ReplaceMutation", "InsertionDeletionMutation",

    # Image layers
    "ImagePool"
]

# Version information
__version__ = "3.5.0"

# Package-level docstring
__doc__ = """
Finch: Evolutionary Algorithm Framework

Finch is a flexible and powerful framework for implementing various
evolutionary algorithms in Python. It provides a modular approach to
building and experimenting with different evolutionary computation techniques.

Key Features:
- Modular design with customizable components
- Support for different types of genes (float arrays, strings, arrays, images)
- Various selection, crossover, and mutation operators
- GPU acceleration support using CuPy
- Visualization tools for monitoring evolution progress

For more information, visit: https://github.com/your-username/finch
"""