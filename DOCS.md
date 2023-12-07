
Then, create an environment and add layers to it:

```python
env = environments.Sequential(layers=[
    layers.Populate(gene_pool, 100),
    layers.SortByFitness(),
    layers.CapPopulation(99),
])
```

### Examples Module

The `examples` module contains standalone scripts that can be run directly. For example, to run the `basic.py` script, navigate to the `examples` directory and run `python basic.py`.

### Functions Module

The `functions` module contains utility functions that can be imported and used as needed. For example, to import the `RandomSelection` function:

```python
from Finch.functions import selection
randomSelect = selection.RandomSelection(percent_to_select=.2)
```

### ML Module

To use the `ml` module, first import the desired classes. For example, to import the `KerasPool` class:

```python
from Finch.ml import neuro_pools
```

Then, create a `KerasPool` object and use it to evolve a Keras model:

```python
gene_pool = neuro_pools.KerasPool(evo, fitness_function)
