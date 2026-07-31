"""Render the late-fusion hybrid model architecture from saved metadata."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    Arc,
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)


ARCHITECTURES = {
    "cnn": (
        "Mel spectrogram\n1 × 128 × T",
        "4 × [Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2]\n"
        "channels 1→64→128→256→512\n"
        "AdaptiveAvgPool 2×2 → Flatten 2048 → FC 512 → FC 64 → FC 1 → Sigmoid",
    ),
    "cnn_lstm": (
        "Mel spectrogram\n1 × 128 × 2,584",
        "4 × [Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2]\n"
        "channels 1→32→64→128→256; frequency mean → sequence\n"
        "2-layer LSTM (input 256, hidden 256, dropout .5) → FC 128 → FC 1",
    ),
    "lstm": (
        "Mel sequence\nT × 128 bins",
        "2-layer LSTM (input 128, hidden 512, dropout .2)\n"
        "final hidden state 512 → FC 1 → Sigmoid",
    ),
    "resnet": (
        "Mel spectrogram\n1 × 128 × T",
        "ImageNet ResNet-34 (mono 7×7 Conv)\n"
        "[3, 4, 6, 3] residual BasicBlock stages → global average pool\n"
        "512-D embedding → Dropout .5 → FC 1 → Sigmoid",
    ),
    "vit": (
        "Mel spectrogram\n1 × 128 × 2,584",
        "Overlapping Conv patch embed (16×16, stride 10) + [CLS] + positions\n"
        "ViT-B/16: 12 Transformer blocks, 12 heads, dim 768, MLP 3072\n"
        "[CLS] 768-D → Dropout .1 → FC 1 → Sigmoid",
    ),
    "audio_xgboost": (
        "13 Spotify +\nlow-level audio features",
        "Median imputation → XGBoost regressor\n"
        "300 trees; learning rate .05; max depth 6",
    ),
    "audio_random_forest": (
        "13 Spotify +\nlow-level audio features",
        "Median imputation → Random Forest regressor\n"
        "300 trees; max depth 10; min split 10; min leaf 5",
    ),
    "lyrics_random_forest": (
        "12 engineered\nlyrics features",
        "Median imputation → Random Forest regressor\n"
        "500 trees; max depth 35; sqrt features; min leaf 2",
    ),
    "artist_xgboost": (
        "Artist + track metadata\npopularity, followers,\ncount, top-100 genres",
        "XGBoost regressor\n"
        "300 trees; learning rate .05; max depth 6\n"
        "subsample .8; column sample .8",
    ),
}

MODEL_TITLES = {
    "cnn": ("CNN", "local spectro-temporal patterns"),
    "cnn_lstm": ("CNN–LSTM", "local patterns + temporal sequence"),
    "lstm": ("LSTM", "temporal mel-bin sequence"),
    "resnet": ("ResNet-34", "deep residual spectrogram features"),
    "vit": ("Audio Spectrogram Transformer", "global patch attention"),
    "audio_xgboost": ("Audio XGBoost", "engineered acoustic descriptors"),
    "audio_random_forest": ("Audio Random Forest", "engineered acoustic descriptors"),
    "lyrics_random_forest": ("Lyrics Random Forest", "linguistic characteristics"),
    "artist_xgboost": ("Artist XGBoost", "artist identity and genre signal"),
}

MODEL_DETAILS = {
    "cnn": (
        "Conv channels 64→128→256→512  •  each: 3×3 Conv, BatchNorm, "
        "ReLU, 2×2 MaxPool  •  FC 2048→512→64→1"
    ),
    "cnn_lstm": (
        "Conv channels 32→64→128→256  •  BatchNorm/ReLU/2×2 Pool  •  "
        "2-layer LSTM, hidden 256  •  FC 256→128→1"
    ),
    "lstm": (
        "2 stacked LSTM layers  •  input 128 mel bins  •  hidden 512  •  "
        "dropout .2  •  final hidden state→FC 1"
    ),
    "resnet": (
        "7×7 Conv→BN→ReLU→3×3 Pool  •  residual stages [3,4,6,3]  •  "
        "widths 64/128/256/512  •  GAP→Dropout .5→FC 1"
    ),
    "vit": (
        "16×16 patches, stride 10  •  12 Transformer blocks  •  12 heads  •  "
        "dim 768 / MLP 3072  •  [CLS]→Dropout .1→FC 1"
    ),
    "audio_xgboost": (
        "Pre-extracted audio features  •  300 sequential boosted trees  •  "
        "depth 6  •  learning rate .05"
    ),
    "audio_random_forest": (
        "Pre-extracted audio features  •  300 independent bootstrap trees  •  "
        "depth 10  •  average predictions"
    ),
    "lyrics_random_forest": (
        "Engineered text features  •  500 bootstrap trees  •  depth 35  •  "
        "sqrt feature sampling  •  average predictions"
    ),
    "artist_xgboost": (
        "Aggregated artist + genre features  •  300 sequential boosted trees  •  "
        "depth 6  •  learning rate .05"
    ),
}


def _box(axis, center, width, height, text, color, alpha=1.0, fontsize=9):
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=color,
        edgecolor="#263238",
        linewidth=1.1,
        alpha=alpha,
    )
    axis.add_patch(patch)
    axis.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#172126",
        alpha=alpha,
    )


def _arrow(axis, start, end, alpha=1.0, width=1.2):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=width,
            color="#455a64",
            alpha=alpha,
            shrinkA=2,
            shrinkB=2,
        )
    )


def _flow_box(
    axis, center, width, height, label, color, alpha=1.0, fontsize=8.5
):
    x, y = center
    axis.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            facecolor=color,
            edgecolor="#37474f",
            linewidth=0.9,
            alpha=alpha,
        )
    )
    axis.text(x, y, label, ha="center", va="center", fontsize=fontsize,
              fontweight="bold", color="#263238", alpha=alpha)


def _fan_out(axis, source, target_x, target_ys, alpha=1.0):
    """Connect one preprocessing output to several model lanes via a bus."""
    source_x, source_y = source
    _arrow(axis, (source_x, source_y), (target_x, source_y), alpha, 0.9)
    axis.plot(
        [target_x, target_x],
        [min(target_ys), max(target_ys)],
        color="#607d8b",
        linewidth=0.9,
        alpha=alpha,
    )
    for target_y in target_ys:
        _arrow(axis, (target_x, target_y), (0.215, target_y), alpha, 0.8)


def _layer(axis, x, y, width, height, color, label, alpha=1.0):
    axis.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.003,rounding_size=0.006",
            facecolor=color,
            edgecolor="#37474f",
            linewidth=0.8,
            alpha=alpha,
        )
    )
    axis.text(x, y, label, ha="center", va="center", fontsize=7.2, alpha=alpha)


def _draw_input(axis, name, y, alpha):
    """Draw a spectrogram, feature grid, or metadata pictogram."""
    x = 0.10
    if name in {"cnn", "cnn_lstm", "lstm", "resnet", "vit"}:
        colors = ["#16324f", "#2f6690", "#3a7ca5", "#81c3d7", "#d9edf2"]
        for row in range(5):
            for column in range(9):
                value = (row * 3 + column * 5 + column * row) % len(colors)
                axis.add_patch(
                    Rectangle(
                        (x - 0.064 + column * 0.014, y - 0.029 + row * 0.012),
                        0.013,
                        0.011,
                        facecolor=colors[value],
                        edgecolor="none",
                        alpha=alpha,
                    )
                )
    elif name in {"audio_xgboost", "audio_random_forest"}:
        # The audio table is a concatenation of Spotify descriptors and the
        # exact low-level families present in low_level_audio_features.csv.
        groups = [
            ("Spotify", "13"),
            ("Chroma", "12"),
            ("Mel", "128"),
            ("MFCC", "48"),
            ("Contrast", "7"),
            ("Tonnetz+", "12"),
        ]
        for index, (label, count) in enumerate(groups):
            row, column = divmod(index, 3)
            chip_x = 0.055 + column * 0.046
            chip_y = y + 0.014 - row * 0.029
            _feature_chip(axis, chip_x, chip_y, label, count, alpha)
    elif name == "lyrics_random_forest":
        # Document pictogram followed by the four semantic groups represented
        # by the twelve engineered lyric variables.
        axis.add_patch(
            Rectangle(
                (0.040, y - 0.024),
                0.030,
                0.049,
                facecolor="#fffdf5",
                edgecolor="#455a64",
                linewidth=0.7,
                alpha=alpha,
            )
        )
        for offset, width in ((0.014, 0.020), (0.004, 0.017),
                              (-0.006, 0.021), (-0.016, 0.014)):
            axis.plot([0.045, 0.045 + width], [y + offset, y + offset],
                      color="#78909c", linewidth=0.8, alpha=alpha)
        lyric_groups = [
            ("Counts", "words/sents"),
            ("Lexical", "wealth/density"),
            ("Sound", "syllables"),
            ("Style", "repeat/lang"),
        ]
        for index, (label, detail) in enumerate(lyric_groups):
            row, column = divmod(index, 2)
            chip_x = 0.092 + column * 0.052
            chip_y = y + 0.014 - row * 0.029
            _feature_chip(axis, chip_x, chip_y, label, detail, alpha, 0.047)
    else:
        # Artist features are aggregated over every credited artist before
        # modeling: mean popularity/followers, artist count, and genre flags.
        for index, person_x in enumerate((0.045, 0.062)):
            axis.add_patch(
                Circle(
                    (person_x, y + 0.012 - index * 0.004),
                    0.006,
                    facecolor="#7fc8a9",
                    edgecolor="#37474f",
                    linewidth=0.5,
                    alpha=alpha,
                )
            )
            axis.add_patch(
                FancyBboxPatch(
                    (person_x - 0.009, y - 0.019 - index * 0.004),
                    0.018,
                    0.023,
                    boxstyle="round,pad=0.001,rounding_size=0.005",
                    facecolor="#a8dadc",
                    edgecolor="#37474f",
                    linewidth=0.5,
                    alpha=alpha,
                )
            )
        artist_groups = [
            ("Popularity", "mean"),
            ("Followers", "mean"),
            ("Artists", "count"),
            ("Genres", "top 100"),
        ]
        for index, (label, detail) in enumerate(artist_groups):
            row, column = divmod(index, 2)
            chip_x = 0.098 + column * 0.050
            chip_y = y + 0.014 - row * 0.029
            _feature_chip(axis, chip_x, chip_y, label, detail, alpha, 0.045)


def _feature_chip(
    axis, x, y, label, detail, alpha=1.0, width=0.040
):
    axis.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - 0.011),
            width,
            0.022,
            boxstyle="round,pad=0.002,rounding_size=0.004",
            facecolor="#d7ebf7",
            edgecolor="#607d8b",
            linewidth=0.55,
            alpha=alpha,
        )
    )
    axis.text(x, y + 0.003, label, ha="center", va="center",
              fontsize=4.5, fontweight="bold", alpha=alpha)
    axis.text(x, y - 0.005, detail, ha="center", va="center",
              fontsize=4.0, color="#455a64", alpha=alpha)


def _draw_conv_stack(axis, y, channels, alpha, x0=0.235, x1=0.47):
    xs = [x0 + index * (x1 - x0) / (len(channels) - 1)
          for index in range(len(channels))]
    for index, (x, channel) in enumerate(zip(xs, channels)):
        height = 0.064 - index * 0.006
        for offset in (0.008, 0.004, 0.0):
            axis.add_patch(
                Polygon(
                    [
                        (x - 0.014 + offset, y - height / 2 + offset),
                        (x + 0.014 + offset, y - height / 2 + offset),
                        (x + 0.014 + offset, y + height / 2 + offset),
                        (x - 0.014 + offset, y + height / 2 + offset),
                    ],
                    closed=True,
                    facecolor=["#b8e0d2", "#95b8d1", "#809bce", "#9b5de5"][
                        min(index, 3)
                    ],
                    edgecolor="#37474f",
                    linewidth=0.55,
                    alpha=alpha,
                )
            )
        if index < len(xs) - 1:
            _arrow(
                axis,
                (x + 0.021, y),
                (xs[index + 1] - 0.013, y),
                alpha,
                0.7,
            )


def _draw_lstm_cells(axis, y, count, alpha, x0=0.31, x1=0.48):
    xs = [x0 + index * (x1 - x0) / max(1, count - 1) for index in range(count)]
    for index, x in enumerate(xs):
        _layer(axis, x, y, 0.045, 0.032, "#f4a261", "LSTM", alpha)
        if index:
            _arrow(axis, (xs[index - 1] + 0.023, y), (x - 0.023, y), alpha, 0.8)


def _draw_tree(axis, x, y, scale, color, alpha):
    """Draw a conventional depth-two binary decision tree."""
    root = (x, y + 1.15 * scale)
    internal = [
        (x - 0.72 * scale, y + 0.15 * scale),
        (x + 0.72 * scale, y + 0.15 * scale),
    ]
    leaves = [
        (x - 1.08 * scale, y - 1.0 * scale),
        (x - 0.36 * scale, y - 1.0 * scale),
        (x + 0.36 * scale, y - 1.0 * scale),
        (x + 1.08 * scale, y - 1.0 * scale),
    ]
    for child in internal:
        axis.plot(
            [root[0], child[0]],
            [root[1], child[1]],
            color="#5d4037",
            linewidth=0.9,
            alpha=alpha,
        )
    for parent, children in zip(internal, (leaves[:2], leaves[2:])):
        for child in children:
            axis.plot(
                [parent[0], child[0]],
                [parent[1], child[1]],
                color="#5d4037",
                linewidth=0.8,
                alpha=alpha,
            )
    for node in [root, *internal]:
        axis.add_patch(
            Circle(
                node,
                scale * 0.28,
                facecolor=color,
                edgecolor="#37474f",
                linewidth=0.55,
                alpha=alpha,
            )
        )
    for leaf in leaves:
        axis.add_patch(
            Rectangle(
                (leaf[0] - scale * 0.22, leaf[1] - scale * 0.18),
                scale * 0.44,
                scale * 0.36,
                facecolor="#fff3bf",
                edgecolor="#37474f",
                linewidth=0.5,
                alpha=alpha,
            )
        )


def _draw_operation_chain(axis, y, operations, x0, x1, alpha, repeat=None):
    """Draw named operations as distinct layers with arrows between them."""
    colors = {
        "Conv": "#67b7dc",
        "BN": "#a8dadc",
        "ReLU": "#80cfa9",
        "Pool": "#ffd166",
        "Drop": "#f6bd60",
        "FC": "#ef476f",
        "LN": "#a8dadc",
        "MHSA": "#b892e4",
        "MLP": "#9b5de5",
        "Add": "#ff9f1c",
    }
    xs = [x0 + index * (x1 - x0) / max(1, len(operations) - 1)
          for index in range(len(operations))]
    for index, (x, operation) in enumerate(zip(xs, operations)):
        key = operation.split()[0]
        _layer(
            axis,
            x,
            y,
            max(0.027, min(0.044, (x1 - x0) / max(1, len(operations)) * 0.82)),
            0.041,
            colors.get(key, "#d9d9d9"),
            operation,
            alpha,
        )
        if index:
            _arrow(axis, (xs[index - 1] + 0.014, y), (x - 0.014, y),
                   alpha, 0.55)
    if repeat:
        axis.add_patch(
            FancyBboxPatch(
                (x0 - 0.021, y - 0.030),
                x1 - x0 + 0.042,
                0.060,
                boxstyle="round,pad=0.002,rounding_size=0.006",
                fill=False,
                edgecolor="#6d597a",
                linewidth=0.8,
                linestyle="--",
                alpha=alpha,
            )
        )
        axis.text(x1 + 0.025, y, repeat, ha="left", va="center",
                  fontsize=7.0, fontweight="bold", alpha=alpha)


def _draw_architecture(axis, name, y, alpha):
    """Draw the actual computational pattern instead of a prose model box."""
    if name == "cnn":
        _draw_conv_stack(axis, y - 0.004, [64, 128, 256, 512], alpha,
                         0.235, 0.415)
        _draw_operation_chain(
            axis,
            y,
            [
                "AvgPool\n2×2",
                "FC 512\nReLU\nDrop .5",
                "FC 64\nReLU\nDrop .5",
                "FC 1\nSigmoid",
            ],
            0.46,
            0.59,
            alpha,
        )
        _arrow(axis, (0.437, y), (0.446, y), alpha, 0.7)
    elif name == "cnn_lstm":
        _draw_conv_stack(axis, y - 0.004, [32, 64, 128, 256], alpha,
                         0.225, 0.34)
        _layer(axis, 0.390, y, 0.045, 0.039, "#bde0fe", "Freq\nmean", alpha)
        _layer(axis, 0.460, y, 0.065, 0.044, "#f4a261",
               "LSTM ×2\n256 hidden", alpha)
        _arrow(axis, (0.362, y), (0.3675, y), alpha, 0.7)
        _arrow(axis, (0.4125, y), (0.4275, y), alpha, 0.7)
        _draw_operation_chain(
            axis,
            y,
            ["FC 128\nReLU\nDrop .5", "FC 1"],
            0.52,
            0.59,
            alpha,
        )
        _arrow(axis, (0.493, y), (0.506, y), alpha, 0.7)
    elif name == "lstm":
        for row, row_y in enumerate((y + 0.020, y - 0.020), start=1):
            _draw_lstm_cells(axis, row_y, 4, alpha, 0.235, 0.43)
            axis.text(0.218, row_y, f"L{row}", ha="right", va="center",
                      fontsize=5.7, alpha=alpha)
        for x in (0.235, 0.30, 0.365, 0.43):
            _arrow(axis, (x, y + 0.004), (x, y - 0.004), alpha, 0.55)
        _draw_operation_chain(
            axis, y, ["Final\nstate", "FC\n512→1", "Sigmoid"],
            0.485, 0.59, alpha,
        )
        _arrow(axis, (0.453, y), (0.470, y), alpha, 0.7)
    elif name == "resnet":
        _draw_operation_chain(
            axis, y, ["Conv\n7×7", "BN", "ReLU", "Pool\n3×3"],
            0.22, 0.31, alpha,
        )
        xs = [0.35, 0.405, 0.46, 0.515]
        _arrow(axis, (0.324, y), (0.328, y), alpha, 0.7)
        labels = ["3×", "4×", "6×", "3×"]
        for index, (x, label) in enumerate(zip(xs, labels)):
            _layer(axis, x, y, 0.044, 0.046, "#70c1b3",
                   label + "\nConv3-BN\nReLU\nConv3-BN", alpha)
            if index:
                _arrow(axis, (xs[index - 1] + 0.022, y), (x - 0.022, y),
                       alpha, 0.7)
            axis.add_patch(
                Arc((x, y), 0.038, 0.061, theta1=5, theta2=175,
                    color="#e76f51", linewidth=1.0, alpha=alpha)
            )
        _layer(axis, 0.57, y, 0.050, 0.050, "#ef476f",
               "GAP\nDrop .5\nFC 512→1", alpha)
        _arrow(axis, (0.537, y), (0.545, y), alpha, 0.7)
    elif name == "vit":
        for index in range(4):
            axis.add_patch(
                Rectangle((0.225 + index * 0.007, y - 0.027 + index * 0.004),
                          0.036, 0.054, facecolor="#5aa9e6",
                          edgecolor="#37474f", linewidth=0.5, alpha=alpha)
            )
        _draw_operation_chain(
            axis, y, ["LN", "MHSA", "Add", "LN", "MLP", "Add"],
            0.31, 0.51, alpha, "×12",
        )
        _arrow(axis, (0.282, y), (0.296, y), alpha, 0.7)
        _layer(axis, 0.59, y, 0.050, 0.052, "#ef476f",
               "[CLS]\nDrop .1\nFC 1", alpha)
        _arrow(axis, (0.552, y), (0.565, y), alpha, 0.7)
    else:
        is_forest = "random_forest" in name
        if is_forest:
            # Random Forest: trees are fitted independently on bootstrap
            # samples, then their predictions are averaged.
            tree_xs = [0.27, 0.33, 0.39, 0.45, 0.51]
            for index, x in enumerate(tree_xs):
                _draw_tree(
                    axis, x, y - 0.006 + (index % 2) * 0.010, 0.013,
                    "#70c1b3", alpha,
                )
                _arrow(axis, (x + 0.014, y), (0.555, y), alpha, 0.45)
            axis.add_patch(
                Circle((0.565, y), 0.020, facecolor="#ffd166",
                       edgecolor="#37474f", linewidth=0.7, alpha=alpha)
            )
            axis.text(0.565, y, "AVG", ha="center", va="center",
                      fontsize=6.5, fontweight="bold", alpha=alpha)
            detail = (
                (
                    "independent bootstrap trees • all features/split • "
                    "300 trees / depth 10"
                )
                if name == "audio_random_forest"
                else (
                    "independent bootstrap trees • sqrt features/split • "
                    "500 trees / depth 35"
                )
            )
        else:
            # XGBoost: each tree is added sequentially to correct the current
            # ensemble residual, scaled by the learning rate.
            tree_xs = [0.27, 0.335, 0.40, 0.465, 0.53]
            for index, x in enumerate(tree_xs):
                _draw_tree(axis, x, y - 0.006, 0.013, "#90be6d", alpha)
                if index:
                    _arrow(
                        axis,
                        (tree_xs[index - 1] + 0.017, y),
                        (x - 0.017, y),
                        alpha,
                        0.75,
                    )
            _layer(axis, 0.575, y, 0.038, 0.035, "#ffd166", "SUM", alpha)
            detail = "sequential gradient-boosted trees • 300 trees / depth 6 / η=.05"


def render_architecture_graph(
    blend_weights,
    output,
    metrics=None,
    dpi=220,
    hide_zero_weight=False,
):
    """Render the model graph, emphasizing branches selected by the blend."""
    weights = {name: float(value) for name, value in blend_weights.items()}
    unknown = sorted(set(weights) - set(ARCHITECTURES))
    if unknown:
        raise ValueError(f"unknown predictor names: {', '.join(unknown)}")
    if not weights:
        raise ValueError("blend_weights must not be empty")

    ordered = [
        name
        for name in ARCHITECTURES
        if name in weights
        and (not hide_zero_weight or weights[name] > 0.0005)
    ]
    if not ordered:
        raise ValueError("no predictors remain after zero-weight filtering")
    count = len(ordered)
    # A tall canvas leaves visible gutters between independently readable
    # model lanes and avoids distorting layer glyphs.
    if count <= 5:
        figure_width, figure_height = 12.0, 7.6
        top, bottom = 0.72, 0.24
    else:
        figure_width = 17.0
        figure_height = max(21.0, 2.15 * count + 2.0)
        top, bottom = 0.94, 0.04
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    axis.set_xlim(0, 0.86)
    axis.set_ylim(
        bottom - (0.08 if count <= 5 else 0.06),
        top + (0.09 if count <= 5 else 0.08),
    )
    axis.axis("off")

    ys = (
        [0.5]
        if count == 1
        else [top - index * (top - bottom) / (count - 1) for index in range(count)]
    )
    lane_y = dict(zip(ordered, ys))

    # Shared input graph: audio is decoded once and split into spectrogram and
    # pre-extracted-feature paths. Text and artist inputs have their own paths.
    spectrogram_ys = [
        lane_y[name] - 0.010
        for name in ("cnn", "cnn_lstm", "lstm", "resnet", "vit")
        if name in lane_y
    ]
    audio_feature_ys = [
        lane_y[name] - 0.010
        for name in ("audio_xgboost", "audio_random_forest")
        if name in lane_y
    ]
    spectrogram_y = sum(spectrogram_ys) / len(spectrogram_ys)
    audio_feature_y = (
        sum(audio_feature_ys) / len(audio_feature_ys)
        if audio_feature_ys
        else None
    )
    audio_y = (
        (spectrogram_y + audio_feature_y) / 2
        if audio_feature_y is not None
        else spectrogram_y
    )
    _flow_box(axis, (0.060, audio_y), 0.035, 0.046, "Audio", "#cfe8f3")
    _flow_box(
        axis, (0.125, spectrogram_y), 0.080, 0.050,
        "Mel\nspectrogram", "#d7ebf7",
    )
    _arrow(axis, (0.0775, audio_y), (0.085, spectrogram_y), width=1.0)
    _fan_out(axis, (0.165, spectrogram_y), 0.185, spectrogram_ys)
    if audio_feature_y is not None:
        _flow_box(
            axis, (0.125, audio_feature_y), 0.080, 0.050,
            "Pre-extracted\naudio features", "#d7ebf7",
        )
        _arrow(axis, (0.0775, audio_y), (0.085, audio_feature_y), width=1.0)
        _fan_out(axis, (0.165, audio_feature_y), 0.185, audio_feature_ys)

    if "lyrics_random_forest" in lane_y:
        text_y = lane_y["lyrics_random_forest"] - 0.010
        _flow_box(
            axis, (0.060, text_y), 0.035, 0.044, "Text", "#f7e1c7",
            fontsize=7.5,
        )
        _flow_box(
            axis, (0.125, text_y), 0.080, 0.050,
            "Engineered\ntext features", "#f8ead8", fontsize=7.0,
        )
        _arrow(axis, (0.0775, text_y), (0.085, text_y), width=1.0)
        _arrow(axis, (0.165, text_y), (0.205, text_y), width=1.0)
    if "artist_xgboost" in lane_y:
        artist_y = lane_y["artist_xgboost"] - 0.010
        _flow_box(
            axis, (0.060, artist_y), 0.035, 0.050,
            "Artist\nfeatures", "#d9ead3", fontsize=7.0,
        )
        _flow_box(
            axis, (0.125, artist_y), 0.080, 0.050,
            "Aggregated artist\n+ genre features", "#e5f2df", fontsize=6.4,
        )
        _arrow(axis, (0.0775, artist_y), (0.085, artist_y), width=1.0)
        _arrow(axis, (0.165, artist_y), (0.205, artist_y), width=1.0)

    sum_x = 0.665
    sum_y = 0.46
    sum_radius = 0.024
    sum_horizontal_radius = 0.019
    target_offsets = [
        (count - 1 - 2 * index) * 0.0045 for index in range(count)
    ]
    for index, (name, y) in enumerate(zip(ordered, ys)):
        weight = weights[name]
        alpha = 1.0
        line_width = 0.9

        # One discrete card per model with a genuine gap before the next card.
        axis.add_patch(
            FancyBboxPatch(
                (0.205, y - 0.043),
                0.420,
                0.086,
                boxstyle="round,pad=0.004,rounding_size=0.009",
                facecolor="#ffffff",
                edgecolor="#78909c",
                linewidth=0.8,
                alpha=1.0,
            )
        )
        title, purpose = MODEL_TITLES[name]
        axis.text(
            0.212,
            y + 0.058,
            title,
            ha="left",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            color="#172126",
            alpha=alpha,
        )
        axis.text(
            0.415,
            y + 0.034,
            MODEL_DETAILS[name],
            ha="center",
            va="center",
            fontsize=5.6,
            color="#455a64",
        )
        visual_y = y - 0.010
        _draw_architecture(axis, name, visual_y, alpha)
        target_y = sum_y + target_offsets[index]
        target_x = sum_x - sum_horizontal_radius * math.sqrt(
            max(0.0, 1.0 - (target_offsets[index] / sum_radius) ** 2)
        ) - 0.002
        model_output_x = 0.615 if name == "vit" else 0.60
        axis.plot(
            [model_output_x, 0.625],
            [visual_y, visual_y],
            color="#455a64",
            linewidth=0.9,
            alpha=alpha,
        )
        _arrow(
            axis,
            (0.625, visual_y),
            (target_x, target_y),
            alpha,
            line_width,
        )

    # Weighted predictions converge on a compact summation node, followed by a
    # single output arrow.
    axis.add_patch(
        Ellipse(
            (sum_x, sum_y),
            width=2 * sum_horizontal_radius,
            height=2 * sum_radius,
            facecolor="#ffd166",
            edgecolor="#37474f",
            linewidth=1.0,
        )
    )
    axis.text(
        sum_x,
        sum_y,
        "Σ",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#263238",
    )
    _arrow(
        axis,
        (sum_x + sum_horizontal_radius, sum_y),
        (0.704, sum_y),
        width=1.2,
    )
    _flow_box(
        axis,
        (0.755, sum_y),
        0.095,
        0.060,
        "Predicted\npopularity",
        "#ffe0a8",
        fontsize=8.0,
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output


def _load_metadata(path):
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = metadata.get("blend_weights")
    if not isinstance(weights, dict):
        raise ValueError("metadata JSON does not contain a blend_weights object")
    variant = metadata.get("variant", "artist")
    test_metrics = (
        metadata.get("metrics", {})
        .get(f"hybrid_{variant}", {})
        .get("test")
    )
    return weights, test_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metadata",
        type=Path,
        help="*.metrics.json sidecar produced by late_fusion_pipeline.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output PNG/SVG/PDF (default: metadata name + .architecture.png)",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--hide-zero-weight",
        action="store_true",
        help="omit predictors whose learned blend weight is effectively zero",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    weights, metrics = _load_metadata(args.metadata)
    output = args.output or args.metadata.with_name(
        args.metadata.name.removesuffix(".metrics.json") + ".architecture.png"
    )
    render_architecture_graph(
        weights,
        output,
        metrics,
        args.dpi,
        hide_zero_weight=args.hide_zero_weight,
    )
    print(f"saved architecture graph to {output}")


if __name__ == "__main__":
    main()
