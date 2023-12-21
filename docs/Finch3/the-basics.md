---
description: Learn the gist of Finch
---

# The Basics

Finch is somewhat different from other genetic algorithm frameworks you may be familiar with. Finch takes inspiration from `keras`, an ML library you may be familiar with. As such, the main class you should be familiar with is the `Sequential` class. Instead of defining a neural network, this defines a genetic algorithm. In Finch Genetic algorithms are made up of:

* GenePools
  * GenePools tell the environments what types of genes to generate. There are some built-in gene pools like `FloatPool` `IntPool` `ArrayPool` and `BinaryPool`. More will come soon to support edge genetic algorithm types such as the backpack problem.
* Fitness Functions
  * A fitness function is the most important part of a genetic algorithm.
* Environments
  * The main environment is the `Sequential` environment. Environments are made up of layers, and define how our individuals will evolve.
* Layers
  * There are many built-in layers. Layers build environments.&#x20;

```python
pool = FloatPool(length=length, minimum=pool_minimum, maximum=pool_maximum)

# Creating the Sequential environment
environment = Sequential(layers=[
    Populate(pool, population=population_size),
    FloatMutateRange(...),
    ParentSimple(...),
    SortByFitness(),
    CapPopulation(...),
])
```
