# Hybrid model organization

- `late_stage_fusion/`: trains each modality separately and blends their
  predictions. It contains the `hybrid_artist` and `hybrid_no_artist`
  experiments.
- `intermediate_concatenation/`: concatenates embeddings and tabular features
  before fitting a regressor.
- `embedding_extraction/`: creates the neural embedding caches consumed by
  both fusion approaches.

Logs use matching subdirectories under `logs/hybrid/`. Downloaded model
artifacts use matching subdirectories under `models/hybrid/`.
