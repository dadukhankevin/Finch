import random
from Finch.genetics.population import Individual
from Finch.tools.rates import make_callable


class Select:
    def __init__(self, percent_to_select=None, amount_to_select=None):
        if percent_to_select is not None and amount_to_select is not None:
            raise ValueError("Only one of percent_to_select or amount_to_select can be given")

        self.percent_to_select = make_callable(percent_to_select)
        self.amount_to_select = make_callable(amount_to_select)

    def select(self, individuals: list[Individual]):
        pass


class TournamentSelection(Select):
    def __init__(self, percent_to_select=None, amount_to_select=None):
        super().__init__(percent_to_select, amount_to_select)

    def select(self, individuals):
        selected_individuals = []

        if self.percent_to_select is not None:
            amount = int(self.percent_to_select() * len(individuals))
        else:
            amount = self.amount_to_select()

        for _ in range(amount):
            tournament_size = len(individuals)
            tournament_individuals = random.sample(individuals, k=tournament_size)
            winner = max(tournament_individuals, key=lambda individual: individual.fitness)
            selected_individuals.append(winner)

        return selected_individuals


class RandomSelection(Select):
    def __init__(self, percent_to_select=None, amount_to_select=None):
        super().__init__(percent_to_select, amount_to_select)

    def select(self, individuals):
        if self.percent_to_select is not None:
            amount = int(self.percent_to_select() * len(individuals))
        else:
            amount = self.amount_to_select()

        selected_individuals = random.choices(individuals, k=amount)
        return selected_individuals


class RankBasedSelection(Select):
    def __init__(self, factor, percent_to_select=None, amount_to_select=None):
        super().__init__(percent_to_select, amount_to_select)
        self.factor = factor

    def select(self, individuals):
        population_size = len(individuals)
        ranks = list(range(1, population_size + 1))

        selection_probs = [pow(2.71828, -self.factor * rank / population_size) for rank in ranks]

        sum_probs = sum(selection_probs)
        selection_probs = [prob / sum_probs for prob in selection_probs]

        if self.percent_to_select is not None:
            amount = int(self.percent_to_select() * len(individuals))
        else:
            amount = self.amount_to_select()

        selected_indices = random.choices(range(population_size), weights=selection_probs, k=amount)
        selected_individuals = [individuals[i] for i in selected_indices]

        return selected_individuals
