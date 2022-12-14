from Finch.FinchGenetics import *
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(12334)
real = np.random.uniform(low=0, high=1, size=(32,32,3))
filter = np.random.uniform(low=.5, high=.5, size=(32, 32, 3))  # has an alpha chanel
image = np.ones(shape=(32, 32, 3))
n = 0
np
#plt.figure()
#plt.imshow(real)
#plt.show()
def fit(solution):
    both = solution.flatten()*image.flatten()
    actual = sum(np.abs(both-real.flatten()))
    return -actual


def callback(data):
    global n
    n += 1


pool = FloatPool(0, 1, initialization=random, fitfunc=fit, shape=(32, 32, 3))
env = Environment([
    Generate(pool, 5),
    MutatePercent(percent=.1,change_amount=.019, iterations=1),
    #FastMutateTop(pool, amount=3, individual_mutation_amount=32, adaptive=1),
    #OverPoweredMutation(pool, 200, -1, range_rate=.1, fitness_function=fit, method="smart"),
    Parents(pool, gene_size=int(3072/32), family_size=2, percent=1, method="best", amount=4),
    SortFitness(),
    KeepLength(10),
])

data, history = env.evolve(200, verbose=10, callback=callback)
plt.figure()
plt.imshow((image * (data[-1].genes.reshape(32, 32, 3))))
plt.show()
plt.plot(history)
plt.show()

