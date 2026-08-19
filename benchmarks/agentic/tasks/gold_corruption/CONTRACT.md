# Gold corruption GAR

Evolve a language-agnostic

`flag(source, candidate, examples, corpus) -> bool`.

Workers see `TASK.md`, their champion, their fitness, and the shared
`Decoder.md`. Finch scores hidden practice and keeps when that number
rises. Holdout is logged and never used to pick.

## Interface

```
source:     English
candidate:  a target-language string (the human verse, or a broken copy)
examples:   may be empty; do not depend on it
corpus:     list of {source, gold} from the rest of that language
            (no verse ids, no language names)
return:     True to flag the candidate as broken
```

Cache corpus work on `id(corpus)`. Subsample if you need to.

## Fitness

`should_flag` is false on the human verse and true on every broken copy.

```
right_on_humans = accuracy on human verses
right_on_breaks = mean accuracy on each break type
score = (right_on_humans + right_on_breaks) / 2
```

Break types: `wrong_verse`, `omit`, `neighbor_graft`, `name_number`,
`char_edit`. Always-flag and never-flag are both 0.5. A crash is “did
not flag.”

Every worker is trying to make the best `flag()` it can. Shared
research is the one campaign `Decoder.md`.
