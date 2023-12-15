import random
from Finch.genetics import Individual
from Finch.tools.rates import make_callable


class Select:
    """
    Base class for selection strategies.

    Parameters:
    - percent_to_select: A callable returning the percentage of individuals to select.
    - amount_to_select: A callable returning the number of individuals to select.

    Usage:
    ```
    selector = Select(percent_to_select=lambda: 0.2)
    selected_individuals = selector.select(individuals)
    ```
    """

    def __init__(self, percent_to_select=None, amount_to_select=None):
        if percent_to_select is not None and amount_to_select is not None:
            raise ValueError("Only one of percent_to_select or amount_to_select can be given")

        self.percent_to_select = make_callable(percent_to_select)
        self.amount_to_select = make_callable(amount_to_select)

    def select(self, individuals: list[Individual]):
        """
        Abstract method for selecting individuals.

        Parameters:
        - individuals: List of individuals to select from.

        Returns:
        - list[Individual]: Selected individuals.
        """
        pass


class TournamentSelection(Select):
    """
    Tournament selection strategy.

    Parameters:
    - percent_to_select: A callable returning the percentage of individuals to select.
    - amount_to_select: A callable returning the number of individuals to select.

    Usage:
    ```
    selector = TournamentSelection(percent_to_select=lambda: 0.2)
    selected_individuals = selector.select(individuals)
    ```
    """

    def __init__(self, percent_to_select=None, amount_to_select=None):
        super().__init__(percent_to_select, amount_to_select)

    def select(self, individuals):
        """
        Select individuals using tournament selection.

        Parameters:
        - individuals: List of individuals to select from.

        Returns:
        - list[Individual]: Selected individuals.
        """
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
    """
    Random selection strategy.

    Parameters:
    - percent_to_select: A callable returning the percentage of individuals to select.
    - amount_to_select: A callable returning the number of individuals to select.

    Usage:
    ```
    selector = RandomSelection(percent_to_select=lambda: 0.2)
    selected_individuals = selector.select(individuals)
    ```
    """

    def __init__(self, percent_to_select=None, amount_to_select=None):
        super().__init__(percent_to_select, amount_to_select)

    def select(self, individuals):
        """
        Select individuals randomly.

        Parameters:
        - individuals: List of individuals to select from.

        Returns:
        - list[Individual]: Selected individuals.
        """
        if self.percent_to_select is not None:
            amount = int(self.percent_to_select() * len(individuals))
        else:
            amount = self.amount_to_select()
        selected_individuals = random.choices(individuals, k=amount)
        return selected_individuals


class RankBasedSelection(Select):
    """
    Rank-based selection strategy.

    Parameters:
    - factor: Selection pressure factor.
    - percent_to_select: A callable returning the percentage of individuals to select.
    - amount_to_select: A callable returning the number of individuals to select.

    Usage:
    ```
    selector = RankBasedSelection(factor=1.5, percent_to_select=lambda: 0.2)
    selected_individuals = selector.select(individuals)
    ```
    """

    def __init__(self, factor, percent_to_select=None, amount_to_select=None):
        super().__init__(percent_to_select, amount_to_select)
        self.factor = factor

    def select(self, individuals):
        """
        Select individuals using rank-based selection.

        Parameters:
        - individuals: List of individuals to select from.

        Returns:
        - list[Individual]: Selected individuals.
        """
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
