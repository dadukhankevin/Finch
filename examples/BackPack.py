from Finch.FinchGenetics import *

fitness = ValueWeightFunction(maxweight=13, force_unique_items=True) # max wieght of our backpack is 15 weight units. Feel free to make your own fitness function whenever.

# In the format [name, value, weight] all of these have little bearing on reality.
backpack = np.array(
    [("apple", .1, 1), ("phone", 6, 2), ("lighter", .5, .1), ("Book", 3, 33), ("compass", .5, .01), ("flashlight", 1, 4),
     ("water", 10, 6), ("passport", 7, .5), ("computer", 11, 15), ("clothes", 10, 2), ("glasses", 3, .1), ("covid", -100, 0), ("pillow", 1.4, 1)], dtype=object)
print(backpack.shape)

pool = GenePool(backpack, fitness.func, replacement=False, shape=(5,3), treat_sublists_as_genes=1)  # To avoid duplicates "replacement" must be false. 5 items in each backpack
env = Environment([
    Generate(pool, 10),
    Parents(pool, delay=1, gene_size=1, family_size=4, method="best", amount=4),
    OPMutation(pool, fitness_function=fitness.func, amount=10, genes=1),
    SortFitness(),
    KeepLength(9)
])


data, history = env.evolve(30, verbose=10)
print(data[-1].genes.reshape(pool.shape))
plt.plot(history)
plt.show()
