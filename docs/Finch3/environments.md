---
description: Habitats for our individuals
---

# Environments

## Environment Class

The `Environment` class represents an evolutionary environment that manages the evolution of a population of individuals over generations. This is a base class, use something like `Sequential` for better functionality.

**Attributes:**

* **layers:** List of layers in the environment.
* **name:** Name of the environment.
* **verbose\_every:** Frequency of verbose output during evolution (print every n times).
* **device:** Device to run the environment on ('gpu' or 'cpu').
* **fitness\_function:** The fitness function used for evaluating individuals.
* **generations:** Number of generations to evolve for.
* **callback:** Callback function to be executed after each generation. Set this in `.compile()`.
* **individuals:** List of individuals in the population.
* **iteration:** Current iteration during evolution.
* **history:** List to store best fitness values over generations.
* **compiled:** Indicates whether the environment has been compiled.
* **deactivated:** Indicates whether the environment is deactivated.
* **best\_ever:** The best individual observed during evolution.
* **dead\_individuals:** List of individuals that are no longer evolving.

**Methods:**

* **\_\_init\_\_(self, layers=None, name="Environment", verbose\_every=1, device='cpu'):**
  * Constructor for the `Environment` class.
  * **Parameters:**
    * **layers:** List of layers in the environment.
    * **name:** Name of the environment.
    * **verbose\_every:** Frequency of verbose output during evolution (print every n times).
    * **device:** Device to run the environment on ('gpu' or 'cpu').
* **deactivate(self):**
  * Deactivates the environment, preventing further evolution.
* **get\_fitness\_metric(self):**
  * Returns the fitness metric of the best individual observed so far.
* **batch(self, individual):**
  * A utility function to process individuals in batch.
* **compile(self, fitness\_function, individuals=None, callback=None, verbose\_every=1):**
  * Compiles the environment with necessary parameters for evolution.
  * **Parameters:**
    * **fitness\_function:** The fitness function for evaluating individuals.
    * **individuals:** List of individuals to start evolution with.
    * **callback:** Callback function to be executed after each generation.
    * **verbose\_every:** Frequency of verbose output during evolution.
* **evolve(self, generations):**
  * Evolves the population over a specified number of generations.
  * **Parameters:**
    * **generations (int):** Number of generations for evolution.
  * **Returns:**
    * Tuple containing the final population and the fitness history.
* **execute(self, individuals):**
  * Executes one generation of evolution.
  * **Parameters:**
    * **individuals:** List of individuals in the population.
  * **Raises:**
    * `NoIndividualsAtEndOfRun`: If the environment has a population of 0 after running.
* **plot(self):**
  * Plots the history of the best fitness values over generations.

## Sequential Class

The `Sequential` class represents a sequential evolutionary environment.

**Attributes:**

* **layers:** List of layers in the environment.
* **name:** Name of the environment.

**Methods:**

* **\_\_init\_\_(self, layers, name="default"):**
  * Constructor for the `Sequential` class.
  * **Parameters:**
    * **layers:** List of layers in the environment.
    * **name:** Name of the environment.
* **compile(self, fitness\_function, individuals=None, callback=None, verbose\_every=1):**
  * Compiles the environment with necessary parameters for evolution.
  * **Parameters:**
    * **fitness\_function:** The fitness function for evaluating individuals.
    * **individuals:** List of individuals to start evolution with.
    * **callback:** Callback function to be executed after each generation.
    * **verbose\_every:** Frequency of verbose output during evolution.
* **evolve(self, generations: int):**
  * Evolves the population over a specified number of generations.
  * **Parameters:**
    * **generations (int):** Number of generations for evolution.
  * **Returns:**
    * Tuple containing the final population and the fitness history.
* **reset(self):**
  * Resets the environment by clearing individuals, history, and resetting the iteration count.

## Adversarial Class

The `Adversarial` class represents an adversarial evolutionary environment managing multiple sub-environments.

**Attributes:**

* **environments:** List of sub-environments.
* **name:** Name of the adversarial environment.

**Methods:**

* **\_\_init\_\_(self, environments, name="Adversarial Environment"):**
  * Constructor for the `Adversarial` class.
  * **Parameters:**
    * **environments:** List of sub-environments.
    * **name:** Name of the adversarial environment.
* **compile(self, fitness\_function, individuals=None, callback=None, verbose\_every=1):**
  * Compiles the adversarial environment with necessary parameters for evolution.
  * **Parameters:**
    * **fitness\_function:** The fitness function for evaluating individuals.
    * **individuals:** List of individuals to start evolution with.
    * **callback:** Callback function to be executed after each generation.
    * **verbose\_every:** Frequency of verbose output during evolution.
* **evolve(self, generations: int = 1):**
  * Evolves the population over a specified number of generations for each sub-environment.
  * **Parameters:**
    * **generations (int):** Number of generations for evolution.
  * **Returns:**
    * Tuple containing information about the best-performing sub-environment.
* **plot\_fitness\_histories(self, results):**
  * Plots the fitness histories of multiple sub-environments.
  * **Parameters:**
    * **results:** List containing information about each sub-environment's evolution.

## ChronologicalEnvironment Class

The `ChronologicalEnvironment` class represents a chronological evolutionary environment with varying sub-environments.

**Attributes:**

* **environments\_and\_generations:** List of tuples containing sub-environments and their corresponding generations.
* **name:** Name of the chronological environment.
* **combined\_history:** List to store the combined fitness history.
* **best\_environment:** Information about the best-performing sub-environment.
* **individuals:** List of individuals in the population.

**Methods:**

* **\_\_init\_\_(self, environments\_and\_generations, name="mixed\_environment"):**
  * Constructor for the `ChronologicalEnvironment` class.
  * **Parameters:**
    * **environments\_and\_generations:** List of tuples containing sub-environments and their corresponding generations.
    * **name:** Name of the chronological environment.
* **evolve(self, generations=1):**
  * Evolves the population over a specified number of generations, using different sub-environments for each period.
  * **Parameters:**
    * **generations (int):** Number of generations for evolution.
  * **Returns:**
    * Tuple containing the final population and the combined fitness history.
* **compile(self, fitness\_function, individuals=None, callback=None, verbose\_every=1):**
  * Compiles the chronological environment with necessary parameters for evolution.
  * **Parameters:**
    * **fitness\_function:** The fitness function for evaluating individuals.
    * **individuals:** List of individuals to start evolution with.
    * **callback:** Callback function to be executed after each generation.
    * **verbose\_every:** Frequency of verbose output during evolution.
* **plot(self):**
  * Plots the combined fitness history of sub-environments over time.

## AdaptiveEnvironment Class

The `AdaptiveEnvironment` class represents an adaptive evolutionary environment that switches between different sub-environments.

**Attributes:**

* **environments:** List of sub-environments.
* **switch\_every:** Frequency of environment switching.
* **go\_for:** Number of generations to run each sub-environment before switching.
* **name:** Name of the adaptive environment.
* **current\_environment:** The current active sub-environment.
* **current\_generation:** Current generation count.
* **iteration:** Current iteration count.

**Methods:**

* **\_\_init\_\_(self, environments, switch\_every=10, go\_for=0, name="AdaptiveEnvironment"):**
  * Constructor for the `AdaptiveEnvironment` class.
  * **Parameters:**
    * **environments:** List of sub-environments.
    * **switch\_every:** Frequency of environment switching.
    * **go\_for:** Number of generations to run each sub-environment before switching.
    * **name:** Name of the adaptive environment.
* **compile(self, fitness\_function, individuals=None, callback=None, verbose\_every=1):**
  * Compiles the adaptive environment with necessary parameters for evolution.
  * **Parameters:**
    * **fitness\_function:** The fitness function for evaluating individuals.
    * **individuals:** List of individuals to start evolution with.
    * **callback:** Callback function to be executed after each generation.
    * **verbose\_every:** Frequency of verbose output during evolution.
* **evolve(self, generations: int):**
  * Evolves the population over a specified number of generations with adaptive switching between sub-environments.
  * **Parameters:**
    * **generations (int):** Number of generations for evolution.
  * **Returns:**
    * Tuple containing the final population and the fitness history.
* **smart\_evolve(self):**
  * Implements adaptive evolution by intelligently switching between sub-environments.
  * **Returns:**
    * The selected sub-environment for the next generation.
* **switch\_environment(self):**
  * Switches to the sub-environment with the highest fitness increase.
* **plot(self):**
  * Plots the fitness history of the current sub-environment.
