from Finch.environmental.layers import standard_layers, mutation_layers, prompt_layers

float_compatible = [
  standard_layers.Populate,
  standard_layers.SortByFitness,
  standard_layers.ParentSinglePointCrossover,
  standard_layers.ParentNPointCrossover,
  standard_layers.ParentUniformCrossover,
  standard_layers.ParentUniformCrossoverMultiple,
  standard_layers.DuplicateSelection,
  standard_layers.KillBySelection,
  standard_layers.KillByFitnessPercentile,
  standard_layers.RemoveDuplicatesFromTop,
  standard_layers.CapPopulation,
  standard_layers.RemoveAllButBest,
  mutation_layers.FloatMutateAmount,
  mutation_layers.FloatOverPoweredMutation,
  mutation_layers.FloatMutateAmountUniform,
  mutation_layers.FloatMomentumMutation,
  mutation_layers.MutateAmount
]

int_compatible = [
  standard_layers.Populate,
  standard_layers.SortByFitness,
  standard_layers.ParentSinglePointCrossover,
  standard_layers.ParentNPointCrossover,
  standard_layers.ParentUniformCrossover,
  standard_layers.ParentUniformCrossoverMultiple,
  standard_layers.DuplicateSelection,
  standard_layers.KillBySelection,
  standard_layers.KillByFitnessPercentile,
  standard_layers.RemoveDuplicatesFromTop,
  standard_layers.CapPopulation,
  standard_layers.RemoveAllButBest,
  mutation_layers.IntMutateAmount,
  mutation_layers.IntOverPoweredMutation,
  mutation_layers.MutateAmount
]

string_compatible = [
  standard_layers.Populate,
  standard_layers.SortByFitness,
  standard_layers.ParentSinglePointCrossover,
  standard_layers.ParentNPointCrossover,
  standard_layers.ParentUniformCrossover,
  standard_layers.ParentUniformCrossoverMultiple,
  standard_layers.DuplicateSelection,
  standard_layers.KillBySelection,
  standard_layers.KillByFitnessPercentile,
  standard_layers.RemoveDuplicatesFromTop,
  standard_layers.CapPopulation,
  standard_layers.RemoveAllButBest,
  mutation_layers.MutateAmount
]
  # prompt_layers.LlmPromptMutation,
  # prompt_layers.PromptParenting


binary_compatible = [
  standard_layers.SortByFitness,
  standard_layers.ParentSinglePointCrossover,
  standard_layers.ParentNPointCrossover,
  standard_layers.ParentUniformCrossover,
  standard_layers.ParentUniformCrossoverMultiple,
  standard_layers.DuplicateSelection,
  standard_layers.KillBySelection,
  standard_layers.KillByFitnessPercentile,
  standard_layers.RemoveDuplicatesFromTop,
  standard_layers.CapPopulation,
  standard_layers.RemoveAllButBest
]

universal = [
  standard_layers.Populate,
  standard_layers.SortByFitness,
  standard_layers.ParentSinglePointCrossover,
  standard_layers.ParentNPointCrossover,
  standard_layers.ParentUniformCrossover,
  standard_layers.ParentUniformCrossoverMultiple,
  standard_layers.DuplicateSelection,
  standard_layers.KillBySelection,
  standard_layers.KillByFitnessPercentile,
  standard_layers.RemoveDuplicatesFromTop,
  standard_layers.CapPopulation,
  standard_layers.RemoveAllButBest
]

keras_compatible = [
  standard_layers.Populate,
  standard_layers.SortByFitness,
  standard_layers.ParentSinglePointCrossover,
  standard_layers.ParentNPointCrossover,
  standard_layers.ParentUniformCrossover,
  standard_layers.ParentUniformCrossoverMultiple,
  standard_layers.DuplicateSelection,
  standard_layers.KillBySelection,
  standard_layers.KillByFitnessPercentile,
  standard_layers.RemoveDuplicatesFromTop,
  standard_layers.CapPopulation,
  standard_layers.RemoveAllButBest,
  standard_layers.KerasTrain,
  mutation_layers.FloatMutateAmount,
  mutation_layers.FloatOverPoweredMutation,
  mutation_layers.FloatMutateAmountUniform,
  mutation_layers.FloatMomentumMutation,
  mutation_layers.MutateAmount
]

pytorch_compatible = [
  standard_layers.Populate,
  standard_layers.SortByFitness,
  standard_layers.ParentSinglePointCrossover,
  standard_layers.ParentNPointCrossover,
  standard_layers.ParentUniformCrossover,
  standard_layers.ParentUniformCrossoverMultiple,
  standard_layers.DuplicateSelection,
  standard_layers.KillBySelection,
  standard_layers.KillByFitnessPercentile,
  standard_layers.RemoveDuplicatesFromTop,
  standard_layers.CapPopulation,
  standard_layers.RemoveAllButBest,
  mutation_layers.FloatMutateAmount,
  mutation_layers.FloatOverPoweredMutation,
  mutation_layers.FloatMutateAmountUniform,
  mutation_layers.FloatMomentumMutation,
  mutation_layers.MutateAmount
]