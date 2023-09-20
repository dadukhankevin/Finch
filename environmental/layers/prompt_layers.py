from .mutation_layers import Layer
from Finch.ml.llm import LLM
from Finch.tools.rates import make_callable
from typing import Union
from Finch.genetics.population import Individual
from Finch.functions.selection import RankBasedSelection

rank = RankBasedSelection.select

class LlmPromptMutation(Layer):
    def __init__(self, llm: LLM, amount: Union[int|callable] = 2, selection_function: callable = rank, adjective: str =
    'just a little bit'):
        super().__init__()
        self.adjective = adjective
        self.llm = llm
        self.llm.system_prompt = "You are the mutation layer in a genetic algorithm, simply change a given text by " + \
                                 adjective
        self.amount = make_callable(amount)
        self.selection_function = selection_function

    def run(self, individuals: list[Individual], environment: any):
        selected_individuals = self.selection_function(individuals, self.amount())
        for individual in selected_individuals:
            individual.genes = self.llm.run(individual.genes)
        return individuals

