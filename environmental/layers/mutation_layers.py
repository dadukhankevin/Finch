from Finch.environmental.layers.standard_layers import Layer
from Finch.functions.selection import Select
from Finch.tools import rates
from Finch.functions import selection
from Finch.genetics.population import NPCP
from Finch.genetics.genepools import Pool
from typing import Union

randomSelect = selection.RandomSelection(amount_to_select=3)


class MutateAmount(Layer):
    def __init__(self, amount_genes: int, gene_pool: Pool, refit=True,
                 selection_function: callable(Select) = randomSelect):
        super().__init__()
        self.amount_genes = rates.make_callable(amount_genes)
        self.gene_pool = gene_pool
        self.refit = refit
        self.selection_function = selection_function

    @Layer.Measure
    def run(self, individuals, environment):
        self.amount_individuals = rates.make_callable(
            min(len(individuals), self.selection_function.amount_to_select()))  # TODO: fix
        selected_individuals = self.selection_function.select(individuals)
        for individual in selected_individuals:
            individual = self.mutate_one(individual)
            if self.refit:
                individual.fit()
        return individuals

    def mutate_one(self, individual):
        amount = self.amount_genes()
        # filter out the frozen genes from the random choice
        available_indices = NPCP.setdiff1d(NPCP.arange(len(individual.genes)), individual.frozen_genes)
        random_indices = NPCP.random.choice(available_indices, amount, replace=False)
        individual.genes[random_indices] = self.gene_pool.generate_genes(amount)
        return individual


class FloatMutateAmountUniform(MutateAmount):
    def __init__(self, amount_genes: int, gene_pool: Pool, max_mutation=0.1
                 , refit=True, selection_function: callable(Select) = randomSelect):
        super().__init__(amount_genes, gene_pool, refit, selection_function)
        self.max_mutation = rates.make_callable(max_mutation)

    def mutate_one(self, individual):
        random_indices = NPCP.random.choice(individual.genes.size, self.amount_genes(), replace=False)
        mutation = NPCP.random.uniform(-self.max_mutation(), self.max_mutation())
        mutated_genes = individual.genes.copy()
        # filter out the frozen genes from the mutation
        mutated_genes[NPCP.isin(random_indices, individual.frozen_genes)] -= mutation
        individual.genes = mutated_genes
        return individual


class OverPoweredMutation(MutateAmount):  # TODO: determine if genes are frozen correctly in this
    def __init__(self, amount_genes: int, gene_pool: Pool, tries=1,
                 selection_function: callable(Select) = randomSelect):
        super().__init__(amount_genes, gene_pool, False)
        self.tries = rates.make_callable(tries)
        self.selection_function = selection_function

    @Layer.Measure
    def run(self, individuals, environment):
        selected_individuals = self.selection_function.select(individuals)
        for individual in selected_individuals:
            for i in range(self.tries()):
                copied = individual.copy()  # Use a custom copy method instead of deepcopy
                super().mutate_one(copied)
                copied.fit()
                if copied.fitness > individual.fitness:
                    individual.genes = copied.genes  # Assign the genes directly without deepcopy
        return individuals


class FloatMutateAmount(MutateAmount):
    def __init__(self, amount_genes: int, gene_pool: Pool, max_negative_mutation=-0.1,
                 max_positive_mutation=0.1
                 , refit=True, selection_function: callable(Select.select) = randomSelect.select):
        super().__init__(amount_genes, gene_pool, refit, selection_function)
        self.max_negative_mutation = rates.make_callable(max_negative_mutation)
        self.max_positive_mutation = rates.make_callable(max_positive_mutation)

    def mutate_one(self, individual):
        # filter out the frozen genes from the random choice
        available_indices = NPCP.setdiff1d(NPCP.arange(individual.genes.size), individual.frozen_genes)
        random_indices = NPCP.random.choice(available_indices, self.amount_genes(), replace=False)
        mutation = NPCP.random.uniform(self.max_negative_mutation(), self.max_positive_mutation(),
                                       size=self.amount_genes())
        mutated_genes = individual.genes.copy()
        mutated_genes[random_indices] += mutation
        individual.genes = mutated_genes
        return individual

    @Layer.Measure
    def run(self, individuals, environment):
        self.amount_individuals = rates.make_callable(min(len(individuals), self.selection_function.amount_to_select()))
        selected_individuals = self.selection_function.select(individuals)
        for individual in selected_individuals:
            individual = self.mutate_one(individual)
            if self.refit:
                individual.fit()
        return individuals


class IntMutateAmount(MutateAmount):
    def __init__(self, amount_genes: int, gene_pool: Pool, min_mutation=-1, max_mutation=1
                 , refit=True, selection_function: callable(Select) = randomSelect):
        super().__init__(amount_genes, gene_pool, refit, selection_function)
        self.min_mutation = rates.make_callable(min_mutation)
        self.max_mutation = rates.make_callable(max_mutation)

    def mutate_one(self, individual):
        # filter out the frozen genes from the random choice
        available_indices = NPCP.setdiff1d(NPCP.arange(individual.genes.size), individual.frozen_genes)
        random_indices = NPCP.random.choice(available_indices, self.amount_genes(), replace=False)
        mutation = NPCP.random.randint(self.min_mutation(), self.max_mutation() + 1, size=self.amount_genes())
        mutated_genes = individual.genes.copy()
        mutated_genes[random_indices] += mutation
        individual.genes = mutated_genes
        return individual

    @Layer.Measure
    def run(self, individuals, environment):
        self.selection_function.amount_to_select = rates.make_callable(
            min(len(individuals), self.selection_function.amount_to_select()))
        selected_individuals = self.selection_function.select(individuals)
        for individual in selected_individuals:
            individual = self.mutate_one(individual)
            if self.refit:
                individual.fit()
        return individuals


# TODO: I was here
class IntOverPoweredMutation(OverPoweredMutation):
    def __init__(self, gene_pool: Pool, min_mutation=0, max_mutation=1, amount_genes: int = 1, tries=1,
                 selection_function: callable(Select) = randomSelect):
        super().__init__(amount_genes, gene_pool, tries, selection_function)
        self.min_mutation = rates.make_callable(min_mutation)
        self.max_mutation = rates.make_callable(max_mutation)

    def mutate_one(self, individual):
        random_indices = NPCP.random.choice(individual.genes.size, self.amount_genes, replace=False)
        mutation = NPCP.random.randint(self.min_mutation(), self.max_mutation() + 1, size=self.amount_genes())
        mutated_genes = individual.genes.copy()
        mutated_genes[random_indices] += mutation
        individual.genes = mutated_genes
        return individual

    @Layer.Measure
    def run(self, individuals, environment):
        selected_individuals = self.selection_function.select(individuals)
        for individual in selected_individuals:
            for i in range(self.tries()):
                copied = individual.copy()
                copied = self.mutate_one(copied)
                copied.fit()
                if copied.fitness > individual.fitness:
                    individual.genes = copied.genes
                    individual.fit()
        return individuals


class FloatOverPoweredMutation(OverPoweredMutation):
    def __init__(self, amount_genes: int, gene_pool: Pool, max_negative_mutation=-0.1,
                 max_positive_mutation=0.1
                 , tries=1, selection_function: callable(Select) = randomSelect):
        super().__init__(amount_genes, gene_pool, tries, selection_function)
        self.max_negative_mutation = rates.make_callable(max_negative_mutation)
        self.max_positive_mutation = rates.make_callable(max_positive_mutation)

    def mutate_one(self, individual):
        random_indices = NPCP.random.choice(individual.genes.size, self.amount_genes(), replace=False)
        mutation = NPCP.random.uniform(self.max_negative_mutation(), self.max_positive_mutation(),
                                       size=self.amount_genes())
        mutated_genes = individual.genes.copy()
        mutated_genes[random_indices] += mutation
        individual.genes = mutated_genes
        return individual

    @Layer.Measure
    def run(self, individuals, environment):
        selected_individuals = self.selection_function.select(individuals)
        for individual in selected_individuals:
            for i in range(self.tries()):
                copied = individual.copy()
                copied = self.mutate_one(copied)
                copied.fit()
                if copied.fitness > individual.fitness:
                    individual.genes = copied.genes
                    individual.fit()  # TODO: I feel like this should be = copied.fitness
                    break
        return individuals


class FloatMomentumMutation(Layer):  # TODO: de-deprecate
    def __init__(self, divider: float, amount_individuals: None, amount_genes: int, execute_every=1,
                 selection_function: callable(Select) = randomSelect, reset_baseline=False):
        super().__init__()
        self.divider = rates.make_callable(divider)
        self.amount_individuals = rates.make_callable(amount_individuals)
        self.amount_genes = rates.make_callable(amount_genes)
        self.execute_every = rates.make_callable(execute_every)
        self.selection_function = selection_function
        self.reset_baseline = reset_baseline

    @Layer.Measure
    def run(self, individuals, environment):
        if not individuals:
            return individuals
        selected_individuals = self.selection_function.select(individuals)
        if environment.iteration > 0:  # diff will be none until after
            for individual in selected_individuals:
                random_indices = NPCP.random.permutation(len(individual.genes) - 1)
                random_indices = random_indices[:self.amount_genes()]
                individual.genes[random_indices] += (environment.diff[random_indices] / self.divider())
                individual.fit()
        return individuals
