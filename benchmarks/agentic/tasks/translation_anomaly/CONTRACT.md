# Translation anomaly GAR

Evolve a language-agnostic `flag(source, candidate, examples) -> bool`.

Workers see the public task brief, their champion, their fitness, and
the shared `Decoder.md`. Finch scores hidden practice.

## Interface

```
source:     English
candidate:  a target-language string (human gold or a model draft)
examples:   list of {source, gold} from the same few-shot memory
return:     True to flag the candidate as a bad draft
```

Fitness: `flags(drafts)/n - flags(gold)/n`. Flag drafts; leave gold alone.

Drafts are model outputs at several keep rates. Gold is the human verse
with the same memory. A full-keep draft is still a draft.

## The one example you may study

Invented text, not an evaluation language:

```
source = "The child sat down."
examples = [
    {"source": "The dog ran.", "gold": "dog-run."},
    {"source": "The woman stood.", "gold": "woman-stand."},
]
# a broken draft (should be easy to flag):
candidate = "xxx child sit down sit."
# a clean gold (must not flag):
candidate = "child-sit."
```

`python3 score.py artifact` runs only this smoke pair. That printed
score is an interface check, not campaign fitness.

## What to write

One general predicate. No verse names, no language names, no bands
fitted to a cell you have not been shown.
