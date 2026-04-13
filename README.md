# Master Thesis VT-26

## Evaluation workflow

- Run downstream evaluation end to end with `evaluate.py --checkpoints-dir models/<model_name>/checkpoints --datasets lipo,hiv`.
- `evaluate.py` computes embeddings on the fly for each checkpoint and caches Morgan fingerprints once per dataset under `evaluation_cache/<dataset>_morgan.npz`.
- The evaluation and online tracking workflow is kNN-only (no linear-probe metrics).
- The final summary is written to `models/<model_name>/evaluation/results.json` unless you pass `--output`.

## Precomputed graph workflow (optional)

- To reduce CPU load from repeated SMILES parsing, you can precompute PyG graphs into shards.
- Build shards once:
	- `python precompute_graphs.py --input data/zinc/zinc_data --output data/zinc/precomputed_graphs --pattern "*.smi" --shard-size 50000`
- Then switch training to precomputed mode in your JSON config:
	- `"use_precomputed": true`
	- `"precomputed_data_path": "data/zinc/precomputed_graphs"`
- Keep `"use_precomputed": false` to use the previous on-the-fly SMILES loader.
- Optional RAM caching toggles in config:
	- `"cache_data_in_memory": true` caches source rows (CSV/.smi) or uses precomputed cache fallback.
	- `"precomputed_cache_in_memory": true` caches all precomputed graphs in RAM.

Notes:
- Augmentations are still generated online, so DINO local/global matching remains correct via `graph_idx`.
- The precomputed folder should contain `shard_*.pt` files and `metadata.json`.
- With `num_workers > 0`, each worker has its own dataset instance, so RAM usage scales with worker count.

