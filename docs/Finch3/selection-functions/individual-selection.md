# Individual Selection

### Select Class

The `Select` class is the base class for selection strategies.

**Parameters:**

* **percent\_to\_select:** A callable returning the percentage of individuals to select.
* **amount\_to\_select:** A callable returning the number of individuals to select.

**Methods:**

* **\_\_init\_\_(self, percent\_to\_select=None, amount\_to\_select=None):**
  * Constructor for the base class.
  * **Parameters:**
    * **percent\_to\_select:** A callable returning the percentage of individuals to select.
    * **amount\_to\_select:** A callable returning the number of individuals to select.
* **select(self, individuals: list\[Individual]) -> list\[Individual]:**
  * Abstract method for selecting individuals.
  * **Parameters:**
    * **individuals:** List of individuals to select from.
  * **Returns:**
    * `list[Individual]`: Selected individuals.

### TournamentSelection Class

The `TournamentSelection` class is a subclass of `Select` that selects individuals using tournament selection.

**Parameters:**

* **percent\_to\_select:** A callable returning the percentage of individuals to select.
* **amount\_to\_select:** A callable returning the number of individuals to select.

**Methods:**

* **\_\_init\_\_(self, percent\_to\_select=None, amount\_to\_select=None):**
  * Constructor for the `TournamentSelection` class.
  * **Parameters:** Same as the base class.
* **select(self, individuals) -> list\[Individual]:**
  * Selects individuals using tournament selection.
  * **Parameters:** Same as the base class.

### RandomSelection Class

The `RandomSelection` class is a subclass of `Select` that selects individuals randomly.

**Parameters:**

* **percent\_to\_select:** A callable returning the percentage of individuals to select.
* **amount\_to\_select:** A callable returning the number of individuals to select.

**Methods:**

* **\_\_init\_\_(self, percent\_to\_select=None, amount\_to\_select=None):**
  * Constructor for the `RandomSelection` class.
  * **Parameters:** Same as the base class.
* **select(self, individuals) -> list\[Individual]:**
  * Selects individuals randomly.
  * **Parameters:** Same as the base class.

### RankBasedSelection Class

The `RankBasedSelection` class is a subclass of `Select` that selects individuals using rank-based selection.

**Parameters:**

* **factor:** Selection pressure factor.
* **percent\_to\_select:** A callable returning the percentage of individuals to select.
* **amount\_to\_select:** A callable returning the number of individuals to select.

**Methods:**

* **\_\_init\_\_(self, factor, percent\_to\_select=None, amount\_to\_select=None):**
  * Constructor for the `RankBasedSelection` class.
  * **Parameters:** Same as the base class, with an additional `factor` parameter.
* **select(self, individuals) -> list\[Individual]:**
  * Selects individuals using rank-based selection.
  * **Parameters:** Same as the base class.
