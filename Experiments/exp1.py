from Finch.FinchGenetics import *

fitness = ValueWeightFunction(maxweight=10, force_unique_items=True) #max wieght of our backpack is 15 weight units. Feel free to make your own fitness function whenever.

# In the format [name, value, weight] all of these have little bearing on reality.

template = [100]*10
pool = GenePool(backpack, fitness.func, replacement=False, shape=(5,3), treat_sublists_as_genes=1)  # To avoid duplicates "replacement" must be false. 5 items in each backpack

