from .mutation_layers import Layer
from Finch.ml.llm import LLM
from Finch.tools.rates import make_callable
from typing import Union
from Finch.genetics.population import Individual
from Finch.functions.selection import RankBasedSelection
from Finch.genetics.genepools import Pool


rank = RankBasedSelection(2).select

class LlmPromptMutation(Layer):
    def __init__(self, llm: LLM, temperature = .8, amount=2, selection_function: callable = rank, adjective: str =
    'one or two words'):
        super().__init__()
        self.adjective = adjective
        self.llm = llm

        self.instructions = "You are the mutation layer in a genetic algorithm, simply change a given text by " + \
                            adjective
        self.llm.system_prompt = self.instructions
        self.amount = make_callable(amount)
        self.selection_function = selection_function
        self.temperature = temperature
        self.llm.temperature = temperature

    def run(self, individuals: list[Individual], environment: any):
        selected_individuals = self.selection_function(individuals, self.amount())
        for individual in selected_individuals:
            self.llm.system_prompt = self.instructions
            self.llm.temperature = self.temperature
            individual.genes = self.llm.run(individual.genes)
        return individuals

class PromptParenting(Layer):

    def __init__(self, llm, temperature=.8, num_children=1, selection_function=rank):
        super().__init__()
        self.llm = llm
        self.num_children = make_callable(num_children)
        self.temperature = temperature
        self.llm.temperature = temperature
        self.selection_function = selection_function

    def run(self, individuals, environment):
        parents = self.selection_function(individuals, self.num_children())

        parent1_prompt = parents[0].genes
        parent2_prompt = parents[1].genes

        # Generate multiple children
        for i in range(self.num_children):
            # Recombine prompts with LLM
            self.llm.system_prompt = "You are a parenting function in a genetic algorithm, combine or parent these " \
                                     "two prompts producing a child prompt of similar lenght.: "
            self.llm.temperature = self.temperature
            child_prompt = self.llm.run(parent1_prompt + "\n" + parent2_prompt)

            # Add child to population
            child = Individual(child_prompt, individuals[0].fitness_function)
            individuals.append(child)

        return individuals

