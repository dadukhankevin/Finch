import random


class TournamentSelection:
    def __init__(self, num_selections, tournament_size):
        self.num_selections = num_selections
        self.tournament_size = tournament_size

    def select(self, individuals):
        selected_individuals = []

        for _ in range(self.num_selections):
            tournament_individuals = random.sample(individuals, k=self.tournament_size)
            winner = max(tournament_individuals, key=lambda individual: individual.fitness)
            selected_individuals.append(winner)

        return selected_individuals


class RandomSelection:
    def __init__(self, num_selections):
        self.num_selections = num_selections

    def select(self, individuals):
        selected_individuals = random.choices(individuals, k=self.num_selections)
        return selected_individuals


class RankBasedSelection:
    def __init__(self, factor):
        self.factor = factor

    def select(self, individuals):
        population_size = len(individuals)
        ranks = list(range(1, population_size + 1))

        # Adjusted formula to assign higher probabilities to lower ranks
        selection_probs = [pow(2.71828, -self.factor * rank / population_size) for rank in ranks]

        # Normalize probabilities
        sum_probs = sum(selection_probs)
        selection_probs = [prob / sum_probs for prob in selection_probs]

        selected_indices = random.choices(range(population_size), weights=selection_probs, k=population_size)
        selected_individuals = [individuals[i] for i in selected_indices]

        return selected_individuals
