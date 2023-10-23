from Finch.genetics import genepools
from Finch.environmental import environments
from Finch.environmental.layers import mutation_layers, standard_layers

valid_environments = environments.AdaptiveEnvironment

valid_layers = [standard_layers.Populate, standard_layers.SortByFitness, standard_layers.ParentSinglePointCrossover, standard_layers.ParentNPointCrossover, standard_layers.ParentUniformCrossover, standard_layers.P]