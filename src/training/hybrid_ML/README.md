# Hybrid model organization

- `late_stage_fusion/`: trains each modality separately and selects a convex
  validation blend of CNN, CNN-LSTM, LSTM, ResNet, ViT, engineered-audio
  XGBoost, engineered-audio Random Forest, and lyrics Random Forest
  predictions. `hybrid_artist` additionally includes artist XGBoost;
  `hybrid_no_artist` excludes all artist features.
- `intermediate_concatenation/`: concatenates embeddings and tabular features
  before fitting a regressor.
- `embedding_extraction/`: creates the neural embedding caches consumed by
  both fusion approaches.

Logs use matching subdirectories under `logs/hybrid/`. Downloaded model
artifacts use matching subdirectories under `models/hybrid/`.
