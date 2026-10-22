# Few-Shot-CoT prompts from the shared Auto-CoT/Automate-CoT pool

This package contains four stage-specific Five-Shot-CoT prompt templates for
the 500, 300, 150, and 75 word compression stages.

## Pool and eligibility rule

The source is the retained 100-candidate Automate-CoT pool for each stage. A
candidate enters the common pool only if it also passes the Auto-CoT quality
gate. Consequently, every sampled demonstration has:

- a non-empty stage input, rationale, and final summary;
- at least 25 substantive rationale words;
- no more than 512 rationale words;
- the expected input length for its compression stage;
- a final summary inside the registered +/-5% word window; and
- a complete final sentence.

Five unique demonstrations are sampled uniformly without replacement from
each stage pool. The selection uses one `random.Random(32)` stream and processes
the stages in the fixed order 500, 300, 150, 75.

The complete source hashes, eligibility exclusions, selected candidate IDs,
content hashes, and prompt hashes are recorded in
`few_shot_cot_manifest.json`.

## Rebuild

From the project directory:

```bash
python3 build_shared_pool_fewshot_prompts.py
```

Rebuilding from unchanged source artifacts produces the same demonstration
selection and the same four prompt files. It only refreshes the manifest's
creation timestamp.
