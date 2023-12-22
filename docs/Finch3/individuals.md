---
description: Finch is nothing without it's Individuals 💖
---

# Individuals

### Individual Class

The `Individual` class represents an individual in a population.

**Attributes:**

* **genes:** The genetic information or characteristics of the individual.
* **fitness:** The default fitness value for this individual.
* **parents:** List of parents of the individual.
* **children:** List of any children the parents have had.
* **gene\_pool:** The gene pool from which genes are selected.
* **check\_fitness:** Flag indicating whether to check the fitness during evolution.
* **device:** The computing device on which the individual's genes are stored ('cpu' or 'gpu').
* **age:** The age of the individual.

**Methods:**

* **\_\_init\_\_(self, gene\_pool, genes: np.array, fitness: float = 0, device="cpu"):**
  * Constructor for the `Individual` class.
  * **Parameters:**
    * **gene\_pool:** The gene pool from which genes are selected.
    * **genes (np.array):** The genetic information or characteristics of the individual.
    * **fitness (float):** The default fitness value for this individual.
    * **device (str):** The computing device on which the individual's genes are stored ('cpu' or 'gpu').
* **copy(self):**
  * Creates and returns a new copy of the individual.
  * **Returns:**
    * A new `Individual` object with the same genes, fitness, and device.
* **to\_cpu(self):**
  * Moves the individual's genes to the CPU and switches to using NumPy.
  * **Returns:**
    * The modified `Individual` object.
* **to\_gpu(self):**
  * Moves the individual's genes to the GPU and switches to using CuPy.
  * **Returns:**
    * The modified `Individual` object.
