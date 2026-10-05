# How Agr works

Agr answers typed questions about a state with a probability for every possible answer. It reads the state and all the questions in one forward pass of a pretrained transformer, and it never generates text.

## A request becomes one sequence

The server turns each question into a list of options: `noul` (or `boolean`) becomes no / yes, `choice` becomes one option per name (with its description, if given), and `score` becomes one option per level. The state and the questions are then laid out as one token sequence, in one of two layouts set by the checkpoint's `readout`.

`letters` (Agr) keeps the instruction-tuned backbone's own chat format. The state opens the user turn; each question continues it with its options under single-token codes and ends where the model's reply begins:

```
<user turn> Context: the state ... | Question: q1  Options: (A) option (B) option <end of turn> <model turn> Answer: ( | Question: q2 ...
```

`bilinear` (Agr-flash) wraps the request in five marker tokens:

```
<doc> the state ... <ask> question 1 <choice> option </choice> <choice> option </choice> <answer> <ask> question 2 ... <answer>
```

Two rules keep the questions independent in both layouts:

- Each question's block attends to the state and to itself, never to another question's block.
- Each block's positions start where the state ends, as if it were the only question.

So an answer is the same whichever other questions come with it, and in whatever order. In fp32 this holds to within 0.00001; in bf16 on a GPU, rounding moves answers by up to about 0.02. The state is read once however many questions there are. Gemma's sliding-window layers get the same rule, further limited to keys within the window by position, and SDPA applies the mask.

## Scoring

`letters`: the hidden vector at the end of each question's block goes through the backbone's own output layer (tied to its input embeddings), restricted to that question's option codes, with the backbone's logit soft-cap. The probabilities are the softmax of those logits divided by a temperature that depends on the number of options, `T(n) = max(min, a + b ln n)`, fitted by log loss on our development data and stored in the checkpoint's `config.json` (`temperature_by_options`). A question with more than ten options lists them as `(` code `) text` with the code kept a single token. `order_averaging: true` in the config reads each choice and yes/no question a second time with its options reversed and averages the two; Agr does not set it.

`bilinear`: the hidden vector at each `<answer>` token is projected to a query, and each option's tokens are averaged and projected to a key. The option probabilities are the softmax of their dot products.

Either way, a `noul` answer is the probability of yes, a `choice` answer is the most likely option, and a `score` answer is the expected level. `confidence` uses System One's definitions:

- choice: `(p_max - 1/K) / (1 - 1/K)` for `K` options, 0 for a uniform spread and 1 for certainty
- score: `max(0, 1 - E|level - mode| / D)`, where `D` is the same distance for a uniform spread

Neither is an accuracy rate. The temperature was fitted on our development data, not yours, so check thresholds on your own data before relying on them (`agr eval` below reports calibration error).

Text in a request stays text: control tokens spelled out in it (the chat template's turn markers, or the five bilinear markers) are tokenized as plain text, so a request cannot forge them. With the `letters` readout the server writes a JSON state compactly, with each element's position written into arrays of eight or more, and puts a yes/no question's true/false descriptions into its two options.

## A checkpoint

| File | Holds |
| --- | --- |
| `config.json` | the readout, the token limits, and the temperature (`letters`) or the scorer size (`bilinear`) |
| `backbone/` | the complete transformer in bf16, with its unmodified tokenizer and chat template |
| `head.safetensors` | `bilinear` only: the marker embeddings and the query and key projections |

What was changed from each base model is listed in NOTICE.md.

## Serving

One request is one forward pass. Requests on the same GPU run one at a time, because models that share a device gain nothing from overlapping. A request takes up to 64 questions and 1 MB of body; the body limit is counted as it arrives, so a chunked upload is capped too. Token limits belong to the checkpoint: its `config.json` sets how much of the state and of each question is read (`max_state_tokens`, `max_question_tokens`) and the most tokens one request may hold (`max_request_tokens`, 12,288 when unset). A request over the token limit is refused with a 422; a state or question longer than its budget is cut, and the response says `"truncated": true`.

## Measuring it yourself

`agr eval` scores any System One endpoint (Agr, or anything else that speaks the schema) on a file of labelled requests, and reports accuracy, Brier score, calibration error and latency. `agr compare` pairs two runs question by question, matched on each request's content rather than its line number, and reports each accuracy with a 95% interval and an exact McNemar test of the difference.

```bash
agr eval http://localhost:8000 labelled.jsonl --out agr.jsonl
agr compare agr.jsonl other.jsonl
```

Each line of a labelled file is a request whose questions also carry a `label`:

| Type | Label |
| --- | --- |
| `choice` | one of the option names |
| `noul` | `true` or `false` |
| `score` | `0` for the first level, `1` for the next, and so on |
