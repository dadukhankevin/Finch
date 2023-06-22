from finch import selection, genepools, layers, environments


def fit(individual):
    return "".join(individual).count("a")


gene_pool = genepools.StringPool("qwertyuiopasdfghjklzxcvbnm", length=20, fitness_function=fit)

environment = environments.Sequential(layers=[
    layers.Populate(gene_pool=gene_pool, population=4),
    layers.MutateAmount(amount_individuals=4, amount_genes=3, gene_pool=gene_pool),
    layers.Parent(num_children=2, num_families=4, selection_function=selection.random_selection),
    layers.SortByFitness(),
    layers.Kill(percent=.3),
])

environment.evolve(100)

print(environment.individuals[0].genes)