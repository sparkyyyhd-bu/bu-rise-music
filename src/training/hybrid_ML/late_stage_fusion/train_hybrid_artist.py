"""Train late-stage fusion with all modalities, including artist features."""

from training.hybrid_ML.late_stage_fusion.late_fusion_pipeline import main


if __name__ == "__main__":
    main(include_artist=True)
