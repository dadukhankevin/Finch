import random


# Define the selection functions

def tournament_selection(individuals, num_selections, tournament_size):
    selected_individuals = []

    for _ in range(num_selections):
        tournament_individuals = random.sample(individuals, k=tournament_size)
        winner = max(tournament_individuals, key=lambda individual: individual.fitness)
        selected_individuals.append(winner)

    return selected_individuals


def random_selection(individuals, num_selections):
    selected_individuals = random.choices(individuals, k=num_selections)
    return selected_individuals


def rank_based_selection(individuals, factor):
    population_size = len(individuals)
    ranks = list(range(1, population_size + 1))

    # Adjusted formula to assign higher probabilities to lower ranks
    selection_probs = [pow(2.71828, -factor * rank / population_size) for rank in ranks]

    # Normalize probabilities
    sum_probs = sum(selection_probs)
    selection_probs = [prob / sum_probs for prob in selection_probs]

    selected_indices = random.choices(range(population_size), weights=selection_probs, k=population_size)
    selected_individuals = [individuals[i] for i in selected_indices]

    return selected_individuals