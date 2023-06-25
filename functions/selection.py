import numpy as np

# Try importing CuPy
try:
    import cupy as cp

    # Check if GPU is available
    if cp.cuda.check_cuda_available():
        print("GPU detected. Using CuPy.")

        array_module = cp
    else:
        print("GPU not detected. Using NumPy.")
        array_module = np

except ImportError:
    print("CuPy not found. Using NumPy.")
    array_module = np
np = array_module


# Define the selection functions

def tournament_selection(individuals, num_selections, tournament_size):
    selected_individuals = []

    for _ in range(num_selections):
        tournament_individuals = np.random.choice(individuals, size=tournament_size, replace=False)
        winner = max(tournament_individuals, key=lambda individual: individual.fitness)
        selected_individuals.append(winner)

    return selected_individuals


def random_selection(individuals, num_selections):
    selected_individuals = np.random.choice(individuals, size=num_selections)
    return selected_individuals


def rank_based_selection(individuals, factor):
    population_size = individuals.size
    ranks = np.arange(1, population_size + 1)

    # Adjusted formula to assign higher probabilities to lower ranks
    selection_probs = np.exp(-factor * ranks / population_size)

    # Normalize probabilities
    selection_probs /= np.sum(selection_probs)

    selected_indices = np.random.choice(population_size, size=population_size, p=selection_probs)
    selected_individuals = individuals[selected_indices]

    return selected_individuals


if __name__ == '__main__':  # thanks to chatgpt for this!
    # Example usage

    # Generate some dummy individuals
    class Individual:
        def __init__(self, fitness):
            self.fitness = fitness


    individuals = np.array(
        [Individual(fitness) for fitness in [100, 100, 10, 7, 5, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    # Tournament selection
    tournament_size = 3
    num_selections = 2
    selected_tournament = tournament_selection(individuals, num_selections, tournament_size)
    print("Tournament Selection:")
    for individual in selected_tournament:
        print(individual.fitness)

    # Random selection
    num_selections = 3
    selected_random = random_selection(individuals, num_selections)
    print("\nRandom Selection:")
    for individual in selected_random:
        print(individual.fitness)

    # Rank-based selection
    factor = -10
    selected_rank_based = rank_based_selection(individuals, factor)
    print("\nRank-based Selection:")
    for individual in selected_rank_based:
        print(individual.fitness)
