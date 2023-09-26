import random
from Finch.genetics.population import Individual

from typing import Union

class Select:
    def __init__(self):
        pass

    def select(self, individuals: list[Individual], amount: callable(any)):
        pass


class TournamentSelection(Select):
    # Removed the num_selections and tournament_size parameters from the init
    def __init__(self):
        super().__init__()

    # Added an amount parameter to the select function
    def select(self, individuals, amount):
        selected_individuals = []

        for _ in range(amount):
            # Use the length of individuals as the tournament size
            tournament_size = len(individuals)
            tournament_individuals = random.sample(individuals, k=tournament_size)
            winner = max(tournament_individuals, key=lambda individual: individual.fitness)
            selected_individuals.append(winner)

        return selected_individuals


class RandomSelection(Select):
    # Removed the num_selections parameter from the init
    def __init__(self):
        super().__init__()

    # Added an amount parameter to the select function
    def select(self, individuals, amount):
        selected_individuals = random.choices(individuals, k=amount)
        return selected_individuals


class RankBasedSelection(Select):
    # Keep the factor parameter in the init
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    # Add an amount parameter to the select function
    def select(self, individuals, amount):
        population_size = len(individuals)
        ranks = list(range(1, population_size + 1))

        # Adjusted formula to assign higher probabilities to lower ranks
        selection_probs = [pow(2.71828, -self.factor * rank / population_size) for rank in ranks]

        # Normalize probabilities
        sum_probs = sum(selection_probs)
        selection_probs = [prob / sum_probs for prob in selection_probs]

        # Use the random.choices function with the selection_probs as weights
        selected_indices = random.choices(range(population_size), weights=selection_probs, k=amount)
        selected_individuals = [individuals[i] for i in selected_indices]

        return selected_individuals
