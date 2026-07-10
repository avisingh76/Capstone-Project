# Part 4 — LLM-Powered Feature

## Setup
1. Create `.env` file:
2. Open `.env` and paste your own OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your-openrouter-api-key-here
   ```
   (Get a free key at https://openrouter.ai)
3. Install dependencies: `pip install -r requirements.txt`
4. Make sure `best_model.pkl` (from Part 3) is in the same folder, then run `llm_feature.ipynb`.

## Required documents
- Serialized best model (`best_model.pkl`) carried over from Part 3.
- Code file (`llm_feature.ipynb`).
- README.md (this file).

## Task to do
- Choose one LLM feature track and declare the choice.
- Load `best_model.pkl` and predict on three hand-crafted feature-vector inputs.
- Write a reusable `call_llm()` function with an environment-variable API key.
- Design a system prompt and a user prompt template for the LLM explanation.
- Define a JSON schema with at least 5 required scalar fields and validate every LLM response against it.
- Apply a PII guardrail before every LLM call.
- Run the full pipeline end-to-end on all three inputs and report the results.
- Compare outputs at `temperature=0` vs `temperature=0.7`.

## Output result
- Three model predictions (class + probability) from `best_model.pkl`.
- A working `call_llm()` function demonstrated with a visible test response.
- The system prompt and user prompt template, written out verbatim.
- A defined JSON schema plus validation and fallback behaviour on failure.
- PII guardrail test results (one blocked input, one allowed input).
- A 3-row demonstration table (input, predicted class, probability, explanation JSON, validation status).
- A temperature A/B comparison table (temp=0 vs temp=0.7) with an explanatory note.

## Dataset chosen
- California Housing (same as Parts 1–3). Target being explained: `expensive`, predicted by the tuned Random Forest
  pipeline saved as `best_model.pkl`.

## Track chosen
**Track (C) — Model Prediction Explanation Pipeline.**
This track was picked because it builds directly on the work already done in Parts 1–3 instead of starting fresh —
`best_model.pkl` already exists and makes real predictions, so the most natural next step is to make those predictions
**explainable** to a non-technical user. Tracks A and B would have meant inventing a new, disconnected dataset-parsing
task; Track C instead adds a genuine layer on top of the modeling pipeline: predict → explain →
validate, end to end.

## LLM API connection
- API key is read from `os.environ["OPENROUTER_API_KEY"]` (via `python-dotenv`).
- `call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=800)` builds the JSON payload, posts it with a 30-second
  timeout, retries automatically on HTTP 429 (rate limiting), checks `response.status_code == 200`, and returns
  `response.json()['choices'][0]['message']['content']` (or `None` on failure, with the status code and body printed).
- Provider: OpenRouter (`https://openrouter.ai/api/v1/chat/completions`), model `openrouter/auto`.
- Test call: `call_llm("You are a helpful assistant.", "Reply with only the word: hello")` → returns `"hello"`.

## Prompt design

**System prompt (verbatim):**
```
You are a structured model explanation generator. Return ONLY valid JSON with these exact keys:
prediction_label, confidence_level, top_reason, second_reason, next_step. prediction_label must be
a descriptive string such as 'expensive' or 'not_expensive' — never a bare number or the raw
predicted_class value. Do not include markdown, bullets, or extra text. Keep top_reason and
second_reason to a single short sentence each. confidence_level must be one of low, medium, high.
```

**User prompt template (verbatim, with placeholders):**
```
Feature values:
{feature_json}

Predicted class:
{predicted_class}

Predicted probability:
{predicted_probability}

Return a JSON explanation using the required schema.
Example of a valid prediction_label value: "expensive" or "not_expensive" (not a number).
```

- `temperature=0` is used for the main pipeline so the model always selects its highest-probability next token, giving
  consistent, reproducible JSON — important since the output is parsed and validated as structured data by code.

## JSON schema and fallback

```json
{
  "type": "object",
  "properties": {
    "prediction_label": {"type": "string"},
    "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
    "top_reason": {"type": "string"},
    "second_reason": {"type": "string"},
    "next_step": {"type": "string"}
  },
  "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"],
  "additionalProperties": false
}
```

- Every raw response is stripped and cleaned with `extract_json()` (removes markdown code fences and isolates the `{...}`
  block) before `json.loads()`, inside a `try/except json.JSONDecodeError`.
- `normalize_explanation()` coerces a numeric `prediction_label` (e.g. `1`) into the expected descriptive string
  (`"expensive"` / `"not_expensive"`) before validation, since some free-tier model responses echo the raw class value.
- Each parsed response is validated with `jsonschema.validate()` inside a `try/except jsonschema.ValidationError`.
- On any decode or validation failure, a fallback dict with all 5 fields set to `None` is returned and the error is logged.

## PII guardrail
- `has_pii(text)` regex-checks for an email address or a 10-digit / hyphenated phone number before every `call_llm(...)` call.
- Test 1 (`"Contact the owner at jane.doe@example.com for details."`) → **BLOCKED**.
- Test 2 (`"This record has no personal contact details, just housing statistics."`) → **ALLOWED**.

## End-to-end demonstration (temperature=0)

| Feature Input | Predicted Class | Probability | Validation Status |
|---|---|---|---|
| sample_1 (NEAR BAY, income=8.3, Very High) | 1 (expensive) | 0.970 | pass |
| sample_2 (INLAND, income=3.1, Low) | 0 (not expensive) | 0.965 | pass |
| sample_3 (NEAR OCEAN, income=5.2, Medium) | 1 (expensive) | 0.830 | pass |

- All three inputs produced valid, schema-passing JSON explanations (full JSON text for each is printed in the notebook's
  output cells).

## Temperature A/B comparison
- Each input was sent twice: once at `temperature=0`, once at `temperature=0.7`; both runs passed schema validation for all
  three samples.
- At `temperature=0`, the model always selects the single highest-probability next token, so `prediction_label` and
  `confidence_level` stay stable across repeated calls.
- At `temperature=0.7`, the model samples from a wider slice of the probability distribution, so the wording and the
  choice/order of `top_reason` and `second_reason` can vary between calls even when the underlying prediction is unchanged —
  which is why structured, machine-parsed output favors `temperature=0`, while `temperature=0.7` suits open-ended generation.

## Final files
- `llm_feature.ipynb`
- `README_Part4.md`