# Notice

Agr<br>
Copyright 2026 Command Code

## Base models

The Agr models are modified versions of these base models, both under the Apache License 2.0:

| Model | Base model |
| --- | --- |
| Agr | google/gemma-4-31B-it |
| Agr-flash | HuggingFaceTB/SmolLM2-360M |

Modifications: fine-tuned weights merged into the base model. Agr keeps the base model's vocabulary, chat template and output layer, and adds a fitted answer temperature to its configuration. Agr-flash adds marker tokens to the vocabulary and a scoring head on top.

## Adapted work

Parts of Agr's letters readout are adapted from Decider (github.com/Mapika/decider), Copyright 2026 Mark Marosi, Apache License 2.0: the answer-slot prompt layout, the single-token option codes, the handling of a thinking block the chat template leaves open, the position indexes written into long JSON arrays, the "no:"/"yes:" option rendering, and the temperature as a function of the number of options. Option-order averaging follows Invergent AI's Surogate.

## Design

Agr's question-branch design follows TypeSafe's Jev, as described in Archer Hume's "Jev's Architecture Unmasked", and Kev. The request and response format follows TypeSafe's System One API, so TypeSafe's SDK works against an Agr server.

Agr is an independent project, not affiliated with or endorsed by TypeSafe, Decider's authors, or the authors of any model it is compared with.
