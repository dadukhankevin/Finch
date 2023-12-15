from typing import Union, Callable
import numpy as np
from Finch.genetics import Individual

cp = None
try:
    import cupy as cp
except ImportError:
    pass

class GeneSelector:
    def __init__(self):
        """
        Base class for gene selectors.
        """
        pass

    def select(self, individual: Individual) -> np.ndarray:
        """
        Abstract method to be implemented by subclasses.

        Parameters:
        - individual: An instance of Individual from which genes will be selected.

        Returns:
        - np.ndarray: An array containing the selected genes.
        """
        pass


class PercentSelector(GeneSelector):
    def __init__(self, percent: Union[float, int, Callable]):
        """
        Initialize PercentSelector with the specified percentage.

        Parameters:
        - percent: The percentage of genes to be selected. It can be a float, int, or a callable (function) that takes an individual as an argument.
        """
        super().__init__()
        self.percent = percent

    def select(self, individual: Individual) -> np.ndarray:
        """
        Select a certain percentage of genes from an individual.

        Parameters:
        - individual: An instance of Individual from which genes will be selected.

        Returns:
        - np.ndarray: An array containing the indices of the selected genes.
        """
        device = individual.device
        genes = individual.genes

        if callable(self.percent):
            percent = self.percent(individual)
        else:
            percent = float(self.percent)

        amount = int(len(genes) * percent)
        indices = np.arange(len(genes))

        if device == 'gpu':
            selected_indices = cp.random.choice(indices, size=amount, replace=True)
        elif device == 'cpu':
            selected_indices = np.random.choice(indices, size=amount, replace=True)

        return selected_indices


class AmountSelector(GeneSelector):
    def __init__(self, amount: Union[int, Callable]):
        """
        Initialize AmountSelector with the specified number of genes to be selected.

        Parameters:
        - amount: The number of genes to be selected. It can be an int or a callable (function) that takes an individual as an argument.
        """
        super().__init__()
        self.amount = amount

    def select(self, individual: Individual) -> np.ndarray:
        """
        Select a specified number of genes from an individual.

        Parameters:
        - individual: An instance of Individual from which genes will be selected.

        Returns:
        - np.ndarray: An array containing the indices of the selected genes.
        """
        if callable(self.amount):
            amount = self.amount()
        else:
            amount = self.amount
        device = individual.device
        genes = individual.genes
        indices = np.arange(len(genes))

        if device == 'gpu':
            selected_indices = cp.random.choice(indices, size=amount, replace=True)
        elif device == 'cpu':
            selected_indices = np.random.choice(indices, size=amount, replace=True)

        return selected_indices

# Example usage:
# individual = Individual(genes=np.arange(10), device='cpu')
# selector = PercentSelector(percent=0.3)
# selected_indices = selector.select(individual)
# print(selected_indices)
