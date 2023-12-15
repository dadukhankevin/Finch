

tools.geneselectors
=================================

Classes
-------

``AmountSelector(amount: Union[int, Callable])``
:   Initialize AmountSelector with the specified number of genes to be selected.Parameters:- amount: The number of genes to be selected. It can be an int or a callable (function) that takes an individual as an argument.
### Ancestors (in MRO)
* Finch3.tools.geneselectors.GeneSelector
### Methods
``select(self, individual: Finch3.genetics.Individual) ‑> numpy.ndarray``:

   Select a specified number of genes from an individual.        Parameters:    - individual: An instance of Individual from which genes will be selected.        Returns:    - np.ndarray: An array containing the indices of the selected genes.

``GeneSelector()``
:   Base class for gene selectors.
### Descendants
* Finch3.tools.geneselectors.AmountSelector* Finch3.tools.geneselectors.PercentSelector
### Methods
``select(self, individual: Finch3.genetics.Individual) ‑> numpy.ndarray``:

   Abstract method to be implemented by subclasses.        Parameters:    - individual: An instance of Individual from which genes will be selected.        Returns:    - np.ndarray: An array containing the selected genes.

``PercentSelector(percent: Union[float, int, Callable])``
:   Initialize PercentSelector with the specified percentage.Parameters:- percent: The percentage of genes to be selected. It can be a float, int, or a callable (function) that takes an individual as an argument.
### Ancestors (in MRO)
* Finch3.tools.geneselectors.GeneSelector
### Methods
``select(self, individual: Finch3.genetics.Individual) ‑> numpy.ndarray``:

   Select a certain percentage of genes from an individual.        Parameters:    - individual: An instance of Individual from which genes will be selected.        Returns:    - np.ndarray: An array containing the indices of the selected genes.