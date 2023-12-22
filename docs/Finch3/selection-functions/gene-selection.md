# Gene Selection

### GeneSelector Class

The `GeneSelector` class serves as the base class for gene selectors.

**Methods:**

* **\_\_init\_\_(self):**
  * Constructor for the base class.
* **select(self, individual: Individual) -> np.ndarray:**
  * Abstract method to be implemented by subclasses. This should be the function you pass to layers when needed.
  * **Parameters:**
    * **individual:** An instance of `Individual` from which genes will be selected.
  * **Returns:**
    * `np.ndarray`: An array containing the selected genes.

### PercentSelector Class

The `PercentSelector` class is a subclass of `GeneSelector` that selects a certain percentage of genes from an individual.

**Parameters:**

* **percent:** `Union[float, int, Callable]`
  * The percentage of genes to be selected. It can be a float, int, or a callable (function) that takes an individual as an argument.

**Methods:**

* **\_\_init\_\_(self, percent: Union\[float, int, Callable]):**
  * Initializes `PercentSelector` with the specified percentage.
* **select(self, individual: Individual) -> np.ndarray:**
  * Selects a certain percentage of genes from an individual.
  * **Parameters:**
    * **individual:** An instance of `Individual` from which genes will be selected.
  * **Returns:**
    * `np.ndarray`: An array containing the indices of the selected genes.

### AmountSelector Class

The `AmountSelector` class is a subclass of `GeneSelector` that selects a specified number of genes from an individual.

**Parameters:**

* **amount:** `Union[int, Callable]`
  * The number of genes to be selected. It can be an int or a callable (function) that takes an individual as an argument.

**Methods:**

* **\_\_init\_\_(self, amount: Union\[int, Callable]):**
  * Initializes `AmountSelector` with the specified number of genes.
* **select(self, individual: Individual) -> np.ndarray:**
  * Selects a specified number of genes from an individual.
  * **Parameters:**
    * **individual:** An instance of `Individual` from which genes will be selected.
  * **Returns:**
    * `np.ndarray`: An array containing the indices of the selected genes.
