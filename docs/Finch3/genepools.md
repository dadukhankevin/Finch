# GenePools

### GenePool Class

The `GenePool` class serves as the base class for creating gene pools, determining the genetic makeup of individuals.

**Params:**

* **length:** `int`
  * Amount of genes to place in an individual.
* **device:** `str`
  * Where genes in individuals should be kept ('cpu' or 'gpu').

**Methods:**

* **init(self, length: int, device='cpu'):**
  * Initializes the `GenePool` with specified parameters.
* **generate\_individual(self):**
  * Generates a new individual with genes of type int.
* **generate\_genes(self, amount: int):**
  * Abstract method to be implemented by subclasses for generating genes.

### FloatPool Class

The `FloatPool` class is a specialized gene pool for the creation of individuals containing floats.

**Params:**

* **length:** `int`
  * Amount of genes in each individual.
* **minimum:** `float`
  * Minimum value for a gene.
* **maximum:** `float`
  * Maximum value for a gene.
* **default:** `float`, optional
  * If set, will initialize all genes to the same value.
* **device:** `str`
  * Where genes in individuals should be kept ('cpu' or 'gpu').

**Methods:**

* **init(self, length: int, minimum: float = 0, maximum: float = 1, default: float = None, device="cpu"):**
  * Initializes the `FloatPool` with specified parameters.
* **generate\_genes(self, amount: int):**
  * Generates numpy or cupy array containing float genes.

### IntPool Class

The `IntPool` class is a specialized gene pool for the creation of individuals containing integers.

**Params:**

* **length:** `int`
  * Amount of genes in each individual.
* **minimum:** `int`
  * Minimum value for a gene.
* **maximum:** `int`
  * Maximum value for a gene.
* **default:** `int`, optional
  * If set, will initialize all genes to the same value.
* **device:** `str`
  * Where genes in individuals should be kept ('cpu' or 'gpu').

**Methods:**

* **init(self, length: int, minimum: int = 0, maximum: int = 1, default: int = None, device="cpu"):**
  * Initializes the `IntPool` with specified parameters.
* **generate\_genes(self, amount: int):**
  * Generates numpy or cupy array containing integer genes.

### BinaryPool Class

The `BinaryPool` class is a specialized gene pool for the creation of individuals containing binary genes (0 or 1).

**Params:**

* **length:** `int`
  * Amount of genes in each individual.
* **default:** `int`, optional
  * If set, will initialize all genes to the same value.
* **device:** `str`
  * Where genes in individuals should be kept ('cpu' or 'gpu').

**Methods:**

* **init(self, length: int, default: int = None, device="cpu"):**
  * Initializes the `BinaryPool` with specified parameters.
* **generate\_genes(self, amount: int):**
  * Generates numpy or cupy array containing binary genes (0 or 1).

### ArrayPool Class

The `ArrayPool` class is a gene pool for the creation of individuals by picking items from an array.

**Params:**

* **gene\_array:** `np.ndarray`
  * The array from which genes will be picked.
* **length:** `int`
  * Amount of genes in each individual.
* **device:** `str`
  * Where genes in individuals should be kept ('cpu' or 'gpu').

**Methods:**

* **init(self, gene\_array: np.ndarray, length: int, device="cpu"):**
  * Initializes the `ArrayPool` with specified parameters.
* **generate\_genes(self, amount: int):**
  * Generates numpy or cupy array containing genes picked from the array.
