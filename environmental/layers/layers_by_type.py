from .ascii_art_layer import AsciiArtLayer
from .mutation_layers import MutationLayer
from .outlines_layer import OutlinesLayer
from .prompt_layers import PromptLayer

LAYERS_BY_TYPE = {
    'mutation': MutationLayer,
    'prompt': PromptLayer,
    'outlines': OutlinesLayer,
    'ascii_art': AsciiArtLayer
}
