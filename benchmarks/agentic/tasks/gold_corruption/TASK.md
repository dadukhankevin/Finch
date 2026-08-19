# Gold corruption

Evolve a language-agnostic

`flag(source, candidate, examples, corpus) -> bool`.

Return true when the candidate looks like a broken copy of the human verse.

```
source:     English
candidate:  a target-language string
examples:   may be empty
corpus:     list of {source, gold} from the rest of that language
```

Fitness is half “right on human verses” and half “mean accuracy on each
break type.” Break types: `wrong_verse`, `omit`, `neighbor_graft`,
`name_number`, `char_edit`. Finch injects your current fitness.

Cache corpus work on `id(corpus)`. Subsample if you need to.

Invented smoke shape, not an evaluation language:

```
source = "The child sat down."
examples = [
    {"source": "The dog ran.", "gold": "dog-run."},
    {"source": "The woman stood.", "gold": "woman-stand."},
]
corpus = [
    {"source": "The man walked.", "gold": "man-walk."},
    {"source": "The bird flew.", "gold": "bird-fly."},
]
# broken:  candidate = "xxx child sit down sit."
# human:   candidate = "child-sit."
```
