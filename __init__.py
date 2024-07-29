# Finch: Evolutionary Algorithm Framework
# Version: 3.5.2
from . import generic, universal, selectors, rates, layers

# Define public API
__all__ = [
    "generic",
    "universal",
    "selectors",
    "rates",
    "layers"
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