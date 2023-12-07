# Finch

Finch is a genetic algorithm library for Python focused on flexibility and ease of use.
## Docs
These are very much a work in progress but should be helpful!
- https://finch-1.gitbook.io/finch/
## Key Features

- Layer-based environments for composable evolution
- Multiple built-in environments like Sequential and Adversarial  
- Genepools for genes like integers, floats, strings, and binary
- Flexible selection and mutation functions
- Built-in fitness functions for common tasks 
- GPU acceleration with CuPy for larger populations

## Environments

- **Sequential**: Evolve through a predefined sequence of layers
- **Adversarial**: Compare multiple environments and evolve the best performer
- **Chronological**: Switch environments at specified generations
- **Adaptive**: Automatically switch environments based on performance

## Layers

- **Populate**: Initialize a population from a genepool
- **MutateAmount**: Mutate a number of genes in selected individuals
- **Parent**: Recombine selected parents into new individuals
- **SortByFitness**: Sort the population by fitness
- **CapPopulation**: Cull the population down to a max size
  
## Genepools  

- **IntPool**: Generate genes as integer values
- **FloatPool**: Generate genes as float values
- **BinaryPool**: Generate binary integer genes 
- **StringPool**: Generate string genes
- **OutlinesPool**: Generate guided text using the Outlines library

## AI Features (very early)

Finch includes tools for evolving AI models and prompts:

### Layers

- **LlmPromptMutation**: Mutate prompts with a language model  
- **PromptParenting**: Generate new prompts by recombining with a language model

### Genepools

- **KerasPool**: Evolve the weights of a Keras model
- **PyTorchPool**: Evolve the weights of a PyTorch model
- **TensorFlowPool**: Evolve the weights of a TensorFlow model
- **PromptPool**: Generate prompt genes with a language model

Finch makes it easy to customize and combine predefined building blocks into a tailored evolutionary algorithm. The adversarial and adaptive environments provide automated optimization not found in other libraries. With CuPy support and fitness analysis tools, Finch enables scaling up experiments and tracking progress.
# Detailed Documentation

Detailed documentation for Finch provides comprehensive insights into its design principles, architecture, functionalities, and usage examples. The detailed documentation includes specifics on the genepools, layers, selection and mutation functions, and guidance on how to optimize and evolve AI models using the library. It is an essential resource for practitioners looking to harness the power of genetic algorithms for scalable and flexible solutions.

For more detailed documentation, please refer to the `DOCS.md` file.
