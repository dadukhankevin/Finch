# Finch: Evolutionary Algorithm Framework

Finch is a Python framework for implementing evolutionary algorithms. It provides a modular approach to building and experimenting with various evolutionary computation techniques.

## Key Features

- Modular design with customizable components
- Support for different types of genes (float arrays, strings, arrays)
- Various selection, crossover, and mutation operators
- GPU acceleration support using CuPy
- Visualization tools for monitoring evolution progress

## Main Components

1. **GenePool**: Generates initial populations
   - FloatPool, StringPool, ArrayPool, ImagePool

2. **Individual**: Represents a single solution in the population

3. **Layer**: Defines genetic operators
   - Selection layers
   - Crossover layers (e.g., N-Point, Uniform)
   - Mutation layers (e.g., Gaussian, Uniform, Polynomial, Swap, Inversion, Scramble)

4. **Environment**: Manages the evolution process

5. **Competition**: Allows comparing multiple evolutionary strategies

## Usage

1. Define your fitness function
2. Create a GenePool
3. Set up Layers for selection, crossover, and mutation
4. Initialize an Environment with your layers and individuals
5. Run the evolution process

## Example

```python
from Finch import FloatPool, Environment, TournamentSelection, GaussianMutation

# Define fitness function
def fitness_function(individual):
    return -sum((x - 1)**2 for x in individual.item)  # Minimize distance from 1

# Create gene pool and initial population
gene_pool = FloatPool(ranges=[[-5, 5]] * 10, length=10, fitness_function=fitness_function)
initial_population = gene_pool.generate_individuals(100)

# Set up layers
layers = [
    TournamentSelection(percent_to_select=0.5),
    GaussianMutation(mutation_rate=0.1, sigma=0.5)
]

# Create and run environment
env = Environment(layers, initial_population)
env.evolve(generations=100)

# Plot results
env.plot()
```

## Installation

```
pip install finch-genetics
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
