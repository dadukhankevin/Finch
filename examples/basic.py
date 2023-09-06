from Finch.environmental import environments
from Finch.environmental import layers
from Finch.genetics import genepools
from textblob import TextBlob
import matplotlib.pyplot as plt
from Finch.functions import selection
rank = selection.RankBasedSelection(2)
history = {}

def fit(individual):
    global history
    score = 0
    text = "".join(individual)
    words = text.split(" ")
    for i in words:
        try:
            points = history[i]
            score += points
        except:
            points = abs(TextBlob(i).polarity)
            history.update({i: points})
            score += points
    return score


gene_pool = genepools.StringPool("qwertyuiopasdfghjklzxcvbnm      ", length=100, fitness_function=fit)

environment = environments.Sequential(layers=[
    layers.Populate(gene_pool=gene_pool, population=4),
    layers.MutateAmount(amount_individuals=10, amount_genes=2, gene_pool=gene_pool, selection_function=rank.select),
    layers.ParentBestChildBinary(5, rank.select),
    layers.SortByFitness(),
    layers.CapPopulation(max_population=39),
])

environment.evolve(50000, verbose_every=50)

print(environment.individuals[0].genes)
print("VOCAB SIZE:")
print(len(history))
plt.plot(environment.history)
plt.show()