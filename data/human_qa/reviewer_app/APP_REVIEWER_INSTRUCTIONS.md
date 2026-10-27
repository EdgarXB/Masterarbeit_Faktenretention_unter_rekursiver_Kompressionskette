# Browser Review App Instructions

Open `human_validation_review_app.html` in a browser.

You will see one review item at a time in a randomized order. For each item:

1. Read the main comparison: **question**, **summary text**, and **generated answer**.
2. Choose a **Question Quality** label.
3. Choose an **Answer Label**.
4. Add notes only when helpful.
5. Click **Save and next**.

Your progress is saved automatically in your browser. When finished, click **Export answers** and send back the downloaded `human_validation_answers.csv`.

If you cannot finish all 60 items, please still export and send back your answers. Because the order is randomized, partial reviews are still useful.

The source evidence and article excerpt are in **Question source context**. Use that context for judging whether the generated question is valid. For the answer label, focus on whether the **summary text** supports the generated/reference answer.

The model support rationale is hidden by default. It appears after you choose `Lost`, `Unsupported`, or `Unclear`, so you can inspect difficult cases without being biased before making the normal retained judgement.

## Question Quality

- `Good`: clear, source-supported, answerable, correct reference answer.
- `Partial`: usable but slightly vague, awkward, broad, or imperfect.
- `Bad`: misleading, unsupported, wrong question, or wrong reference answer.
- `Unclear`: not enough context to judge.

## Answer Label

- `Retained`: the summary supports the reference answer.
- `Lost`: the summary does not contain enough information to answer.
- `Unsupported`: the model answer is not supported by the summary, contradicts the reference, or overclaims.
- `Unclear`: ambiguous.

Key rule: judge the summary, not the original article. A fact can be true in the article but still lost if it is missing from the summary.
