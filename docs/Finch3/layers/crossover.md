---
description: How to perform parenting/crossover in Finch
---

# crossover



### Parent Layer

The `Parent` layer serves as the foundational class for implementing parenting/crossover operations. It should not be used directly but serves as a base class for creating specific parenting layers.

**params:**

* **families:** `callable, int`
  * The method used to select individuals as parents.
* **children:** `callable, int`
  * The method to determine the number of children each pair of parents should produce.
* **device:** `str`
  * The device on which the layer will execute ('cpu' or 'gpu').
* **refit:** `bool`
  * Flag indicating whether to recalculate fitness for the newly generated individuals.
* **track\_genealogies:** `bool`
  * Flag indicating whether to track the genealogies of individuals.&#x20;

**methods:**

* **init(self, families, children=1, device='cpu', refit=True, track\_genealogies=False):**
  * Initializes the `Parent` layer with specified parameters.
* **parent(self, parent1, parent2, environment) -> list:**
  * Abstract method to be implemented by subclasses. Generates offspring from two-parent individuals.
* **execute(self, individuals):**
  * Executes the parenting/crossover operation on a list of individuals, creating offspring and adding them to the environment.

### Crossover Function

The `crossover` function performs single-point crossover between two parents.

**Params:**

* **parent1:** `Individual`
  * The first parent.
* **parent2:** `Individual`
  * The second parent.

**Returns:**

* `list` of `Individual`: Two offspring resulting from the crossover.

#### Uniform Crossover Function

The `uniform_crossover` function performs uniform crossover between two parents.

**Params:**

* **parent1:** `Individual`
  * The first parent.
* **parent2:** `Individual`
  * The second parent.
* **probability:** `float`, optional
  * The probability of selecting a gene from the first parent.

**Returns:**

* `list` of `Individual`: Two offspring resulting from the crossover.

### N-Point Crossover Function

The `n_point_crossover` function performs n-point crossover between two parents.

**Params:**

* **parent1:** `Individual`
  * The first parent.
* **parent2:** `Individual`
  * The second parent.
* **n:** `int`
  * Number of crossover points.

**Returns:**

* `list` of `Individual`: Offspring resulting from the crossover.

### ParentSimple Layer

The `ParentSimple` layer is a simple parenting layer with a single-point crossover.

**Params:**

* **families:** `callable`
  * Selection method for individuals.
* **children:** `int`
  * Number of children to generate.
* **refit:** `bool`
  * If true, will retest fitness in new children.

**Methods:**

* **init(self, families, children=2, refit=True):**
  * Initializes the `ParentSimple` layer with specified parameters.
* **parent(self, parent1: Individual, parent2: Individual, environment) -> list:**
  * Generates children through single-point crossover.

### ParentNPoint Layer

The `ParentNPoint` layer is an n-point parenting layer with crossover.

**Params:**

* **families:** `callable`
  * Selection method for individuals.
* **points:** `int`
  * Number of crossover points.
* **children:** `int`
  * Number of children to generate.
* **refit:** `bool`
  * If true, will retest fitness in new children.
* **track\_genealogies:** `bool`
  * If true, will have individuals remember their parents.

**Methods:**

* **init(self, families, points=1, children=2, refit=True, track\_genealogies=False):**
  * Initializes the `ParentNPoint` layer with specified parameters.
* **parent(self, parent1: Individual, parent2: Individual, environment) -> list:**
  * Generates children through n-point crossover.
