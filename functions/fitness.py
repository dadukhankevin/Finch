from typing import Union


class Fitness:
    def __init__(self, functions: list[tuple[callable, Union[float | int]]]):
        self.functions = functions
        self.index = 0

    def fit(self, genes):
        self.functions[self.index][0](genes)

    def callback(self, environment):
        if environment.individuals[-1].fitness > self.functions[self.index][1]:
            self.index += 1
