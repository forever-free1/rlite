# Experiment record

The checked-in CSV contains held-out GSM8K exact-match measurements from the
completed GRPO and GSPO runs described in the project README. It is source data
for the figures, not a claim of a statistically significant ranking. Raw
runtime logs remain machine-local because they contain environment-specific
paths and verbose engine output.

Regenerate the figures with:

```bash
python scripts/plot_training.py
```
