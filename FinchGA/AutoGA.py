from Finch.FinchGA.GenePools import GenePool
from Finch.FinchGA import Layers
from Finch.FinchGA.Environments import *
import Finch.FinchGenetics.FitnessFunctions as ff
import math

# TODO: add a bunch more auto classes
class ValueWeight: # for problems like the backpack problem
    def __init__(self, items, max_weight, stop_thresh=math.inf, epochs=40):
        """
        :param items: A list of items that, for example could go into a backpack, each item is formatted as such ["item name", value, weight]
        where "value" is how much you "want" the item and "weight" is how much the item weighs. Both of these concepts can be applied to things
        outside of backpacks.
        :param max_weight: The max amount of weight allowed in a backpack
        :param stop_thresh: Stop when the value reaches this amount
        :param epochs: amount of iters
        """
        self.stop_thresh = stop_thresh
        self.max_weight = max_weight
        self.epochs = epochs
        self.pool = GenePool(items, ff.ValueWeightFunction(max_weight).func,
                             replacement=False)  # TO avoid duplicates "replacement" must be false

    def simulate_env(self):
        env = SequentialEnvironment(layers=[
            Layers.GenerateData(self.pool, population=10, array_length=4, delay=0),  # generates data or fills in gaps
            Layers.SortFitness(),  # sort individuals by fitness
            Layers.NarrowGRN(self.pool, delay=1, method="best", amount=2, reward=.05, penalty=.05, mn=.1, mx=100,
                             every=1),  # promotes the best genes
            Layers.UpdateWeights(self.pool, every=1, end=200),  # updates the likelihood if each gene being generated
            Layers.Parents(self.pool, gene_size=1, family_size=3, delay=0, every=4, method="best", percent=100, amount=2),
            # parents them together
            Layers.Mutate(self.pool, delay=0, select_percent=100, likelihood=40),  # mutate it then determines fitness on mutated individuals
            Layers.SortFitness(),  # re sorts it
            Layers.KeepLength(100),  # keeps population low
        ])
        env.compile(epochs=self.epochs, fitness=ff.ValueWeightFunction(self.max_weight).func, every=1,
                    stop_threshold=self.stop_thresh)
        data, hist = env.simulate_env()
        return data, hist
