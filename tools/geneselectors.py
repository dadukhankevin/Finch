from typing import Union, Callable
import numpy as np
from Finch3.genetics import Individual

cp = None
try:
    import cupy as cp
except ImportError:
    pass


class GeneSelector:
    def __init__(self):
        pass

    def select(self, individual: Individual) -> Union[np.ndarray]:
        pass


class PercentSelector(GeneSelector):
    """
    A gene selector that randomly selects a certain percentage of genes from an individual.

    Parameters:
    - percent: The percentage of genes to be selected. It can be a float, int, or a callable (function) that takes an individual as an argument.

    Example Usage:
    ```
    selector = PercentSelector(percent=0.3)
    selected_genes = selector.select(individual)
    ```

    or

    ```
    def dynamic_percent(individual):
        # Some custom logic to calculate the percentage dynamically
        return 0.5

    selector = PercentSelector(percent=dynamic_percent)
    selected_genes = selector.select(individual)
    ```
    """

    def __init__(self, percent: Union[float, int, Callable]):
        super().__init__()
        self.percent = percent

    def select(self, individual: Individual) -> Union[np.ndarray]:
        """
        Select a certain percentage of genes from an individual.

        Parameters:
        - individual: An instance of genetics.Individual from which genes will be selected.

        Returns:
        - Union[np.ndarray, cp.ndarray]: A NumPy or Cupy array containing the selected genes.

        Example Usage:
        ```
        selector = PercentSelector(percent=0.3)
        selected_genes = selector.select(individual)
        ```
        """
        device = individual.device
        genes = individual.genes

        if callable(self.percent):
            percent = self.percent(individual)
        else:
            percent = float(self.percent)

        amount = int(len(genes) * percent)

        if device == 'gpu':
            selected_genes = cp.random.choice(genes, size=amount, replace=True)
        if device == 'cpu':
            selected_genes = np.random.choice(genes, size=amount, replace=True)

        return selected_genes


class AmountSelector(GeneSelector):
    """
    A gene selector that randomly selects a specified number of genes from an individual.

    Parameters:
    - amount: The number of genes to be selected.

    Example Usage:
    ```
    selector = AmountSelector(amount=5) # amount can be a rate
    selected_genes = selector.select(individual)
    ```
    """

    def __init__(self, amount: Union[int, Callable]):
        """
        Initialize the AmountSelector.

        Parameters:
        - amount: The number of genes to be selected.
        """
        super().__init__()
        self.amount = amount

    def select(self, individual: Individual) -> Union[np.ndarray]:
        """
        Select a specified number of genes from an individual.

        Parameters:
        - individual: An instance of genetics.Individual from which genes will be selected.

        Returns:
        - Union[np.ndarray, cp.ndarray]: A NumPy or Cupy array containing the selected genes.

        Example Usage:
        ```
        selector = AmountSelector(amount=5)
        selected_genes = selector.select(individual)
        ```
        """
        if callable(self.amount):
            amount = self.amount()
        else:
            amount = self.amount
        device = individual.device
        genes = individual.genes

        if device == 'gpu':
            selected_genes = cp.random.choice(genes, size=amount, replace=True)
        elif device == 'cpu':
            selected_genes = np.random.choice(genes, size=amount, replace=True)

        return selected_genes

