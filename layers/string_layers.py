from Finch.universal import ARRAY_MANAGER
from Finch.generic import GenePool, Individual, Layer, make_callable
from typing import Callable, List, Union
import random
import string

class StringPool(GenePool):
    def __init__(self, char_set: str, length: int, fitness_function: Callable):
        """
        A GenePool for creating individuals with string genes.
        :param char_set: The set of characters to choose from
        :param length: Length of the string
        :param fitness_function: Function to evaluate fitness
        """
        super().__init__(generator_function=self.generate_string, fitness_function=fitness_function)
        self.char_set = char_set
        self.length = length

    def generate_string(self):
        genes = ''.join(random.choice(self.char_set) for _ in range(self.length))
        return Individual(item=genes, fitness_function=self.fitness_function)
