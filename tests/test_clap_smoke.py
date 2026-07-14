"""Milestone-1 smoke test: real CLAP checkpoint, bundled 10 s WAV fixtures.

Marked `clap` because it loads the ~2 GB checkpoint (downloaded on first run).
Run explicitly with:

    pytest -m clap -v

Asserts cosine(text, matching audio) > cosine(text, mismatched audio).
"""

import numpy as np
import pytest

pytestmark = pytest.mark.clap


@pytest.fixture(scope="module")
def encoder(cfg):
    from playlistgen.clap_model import ClapEncoder

    return ClapEncoder(cfg)


def test_text_audio_alignment(encoder, fixtures_dir):
    beep = encoder.embed_audio_file(fixtures_dir / "beep.wav")
    noise = encoder.embed_audio_file(fixtures_dir / "noise.wav")
    text = encoder.embed_texts(["a high pitched electronic beep tone"])[0]

    assert beep.shape == (512,) and noise.shape == (512,)
    assert np.allclose(np.linalg.norm(beep), 1.0, atol=1e-4)

    sim_match = float(text @ beep)
    sim_mismatch = float(text @ noise)
    assert sim_match > sim_mismatch, (
        f"expected beep text to match beep audio: {sim_match:.3f} vs {sim_mismatch:.3f}"
    )


def test_noise_text_alignment(encoder, fixtures_dir):
    beep = encoder.embed_audio_file(fixtures_dir / "beep.wav")
    noise = encoder.embed_audio_file(fixtures_dir / "noise.wav")
    text = encoder.embed_texts(["harsh white noise static"])[0]
    assert float(text @ noise) > float(text @ beep)
