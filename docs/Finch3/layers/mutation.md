# Mutation

### Mutate Layer

The `Mutate` layer provides a general base mutation layer for individuals in an environment.

**Params:**

* **individual\_selection:** `Union[float, int, Callable]`
  * The method used to select individuals for mutation.
* **gene\_selection:** `Union[float, int, Callable]`
  * The method used to select genes within individuals for mutation.
* **refit:** `bool`
  * If true, will retest fitness after mutation.

### **Methods:**

* **init(self, individual\_selection, gene\_selection, refit=True):**
  * Initializes the `Mutate` layer with specified parameters.
* **mutate\_one(self, individual, environment):**
  * Mutates a single individual.

### FloatMutateRange Layer

The `FloatMutateRange` layer is a float-based layer that mutates genes by changing their values rather than replacing them. This allows for more fine-grained mutations.

**Params:**

* **individual\_selection:** `Union[float, int, Callable]`
  * The method used to select individuals for mutation.
* **gene\_selection:** `Union[float, int, Callable]`
  * The method used to select genes within individuals for mutation.
* **max\_mutation:** `Union[float, int]`
  * Maximum mutation.
* **min\_mutation:** `Union[float, int]`
  * Minimum mutation.
* **refit:** `bool`
  * If true, will retest fitness after mutation.
* **keep\_within\_genepool\_bounds:** `bool`
  * If true, keeps the values of the floats within the max and min set in their gene pools.

**Methods:**

* **init(self, individual\_selection, gene\_selection, max\_mutation=1.0, min\_mutation=-1.0, refit=True, keep\_within\_genepool\_bounds=False):**
  * Initializes the `FloatMutateRange` layer with specified parameters.
* **mutate\_one(self, individual, environment):**
  * Mutates a single individual.

### IntMutateRange Layer

The `IntMutateRange` layer is an integer-based layer that mutates genes by changing their values rather than replacing them. This allows for more fine-grained mutations.

**Params:**

* **individual\_selection:** `Union[float, int, Callable]`
  * The method used to select individuals for mutation.
* **gene\_selection:** `Union[float, int, Callable]`
  * The method used to select genes within individuals for mutation.
* **max\_mutation:** `Union[int, Callable]`
  * Maximum mutation.
* **min\_mutation:** `Union[int, Callable]`
  * Minimum mutation.
* **refit:** `bool`
  * If true, will retest fitness after mutation.
* **keep\_within\_genepool\_bounds:** `bool`
  * If true, keeps the values of the integers within the bounds of possible genes.

**Methods:**

* **init(self, gene\_selection, individual\_selection, max\_mutation=1, min\_mutation=-1, refit=True, keep\_within\_genepool\_bounds=False):**
  * Initializes the `IntMutateRange` layer with specified parameters.
* **mutate\_one(self, individual, environment):**
  * Mutates a single individual.

### BinaryMutate Layer

The `BinaryMutate` layer is a binary-based layer that mutates genes by changing their values to 0 or 1.

**Params:**

* **individual\_selection:** `Union[float, int, Callable]`
  * The method used to select individuals for mutation.
* **gene\_selection:** `Union[float, int, Callable]`
  * The method used to select genes within individuals for mutation.
* **refit:** `bool`
  * If true, will retest fitness after mutation.

**Methods:**

* **init(self, individual\_selection, gene\_selection, refit=True):**
  * Initializes the `BinaryMutate` layer with specified parameters.
* **mutate\_one(self, individual, environment):**
  * Mutates a single individual.
