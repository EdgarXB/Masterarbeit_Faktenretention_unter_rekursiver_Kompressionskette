# Human Validation Instructions

Thank you for helping validate this thesis experiment.

The study tests how well facts survive when news articles are recursively summarized by language models. The automatic evaluation pipeline creates factual questions from source articles, asks those questions against generated summaries, and labels facts as retained, lost, or unsupported. Your task is to check whether that automatic evaluation is reliable.

## Files

You should receive:

- `human_validation_annotation.csv`
- `REVIEWER_INSTRUCTIONS.md`

Please fill in only the empty human-review columns in the CSV.

## What Each Row Contains

- `article_excerpt`: beginning of the original news article, for context.
- `source_evidence`: source text fragment supporting the fact.
- `canonical_fact`: the fact extracted from the source article.
- `generated_question`: the question generated from that fact.
- `reference_answer`: the expected answer according to the source article.
- `summary_text`: the generated summary being evaluated.
- `model_answer`: the answer produced from the summary.
- `model_support_rationale`: the model's explanation for its answer.

The row intentionally does not show the automatic retained/lost/unsupported label. Please judge independently.

## Task 1: Question Quality

Fill `question_quality` with one of:

- `good`
- `partial`
- `bad`
- `unclear`

Use:

- `good`: the question is clear, answerable from the source evidence/fact, and the reference answer is correct.
- `partial`: the question is mostly usable but slightly vague, awkward, too broad, or has a reference answer that is only partly ideal.
- `bad`: the question is misleading, not supported by the source evidence/fact, asks the wrong thing, or has an incorrect reference answer.
- `unclear`: you cannot confidently judge it from the provided context.

Use `question_quality_notes` for a short explanation when helpful, especially for `partial`, `bad`, or `unclear`.

## Task 2: Summary Answer Label

Judge whether the `summary_text`, together with the `model_answer`, supports the `reference_answer`.

Fill `human_answer_label` with one of:

- `retained`
- `lost`
- `unsupported`
- `unclear`

Use:

- `retained`: the summary contains enough information to answer the generated question correctly, even if worded differently.
- `lost`: the summary does not contain enough information to answer the question.
- `unsupported`: the model answer gives information that is not supported by the summary, contradicts the reference answer, or overclaims beyond the summary.
- `unclear`: the case is ambiguous or cannot be judged from the provided context.

Important: judge the summary, not the original article. A fact can be true in the article but still `lost` if it is missing from the summary.

Use `human_answer_label_notes` for a short explanation when helpful.

## Practical Guidance

- Do not spend too long on one row. If it is genuinely ambiguous, mark `unclear`.
- Paraphrases count as correct if the meaning is preserved.
- Exact wording is not required.
- For numbers, dates, locations, and names, be stricter.
- If the question itself is `bad`, still label the answer if possible, but note the problem.

## Time Estimate

The CSV has 60 rows. A careful review usually takes about 45-75 minutes.

