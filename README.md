# Master Thesis VT-26

## Evaluation workflow

- Run downstream evaluation end to end with `evaluate.py --checkpoints-dir models/<model_name>/checkpoints --datasets lipo,hiv`.
- `evaluate.py` computes embeddings on the fly for each checkpoint and caches Morgan fingerprints once per dataset under `evaluation_cache/<dataset>_morgan.npz`.
- The final summary is written to `models/<model_name>/evaluation/results.json` unless you pass `--output`.

