"""Generate the tiny audio fixtures used by the test suite.

Environment: any (Mac or SCC). Run once; the generated WAVs are committed so
tests never need network or this script at runtime:

    python tests/fixtures/make_fixtures.py

Fixtures (10 s, 48 kHz mono, 16-bit):
- beep.wav  : a clean 880 Hz sine tone with slow tremolo ("a high pitched beep")
- noise.wav : white noise ("static noise")

They are acoustically far apart so the CLAP smoke test can assert
cosine(text, matching audio) > cosine(text, mismatched audio).
"""

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48_000
DUR = 10.0


def main() -> None:
    here = Path(__file__).parent
    t = np.linspace(0, DUR, int(SR * DUR), endpoint=False)

    tremolo = 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t)
    beep = 0.5 * tremolo * np.sin(2 * np.pi * 880.0 * t)
    sf.write(here / "beep.wav", beep.astype(np.float32), SR, subtype="PCM_16")

    rng = np.random.default_rng(0)
    noise = 0.3 * rng.standard_normal(len(t))
    sf.write(here / "noise.wav", noise.astype(np.float32), SR, subtype="PCM_16")

    print("wrote", here / "beep.wav", "and", here / "noise.wav")


if __name__ == "__main__":
    main()
