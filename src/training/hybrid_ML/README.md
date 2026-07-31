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

Use `late_stage_fusion/plot_results.ipynb` to create the abstract-facing
comparison of leakage-free, fixed-split held-out MAE, RMSE, and R² for the
neural and traditional models.

Late-fusion training also writes `<model>.architecture.png`, a graph of every
modality, base learner, and its learned convex-blend weight. To regenerate a
graph without retraining:

```bash
python -m training.hybrid_ML.late_stage_fusion.generate_architecture_graph \
  /path/to/hybrid_artist.metrics.json \
  --output figures/final_hybrid_architecture.png
```
