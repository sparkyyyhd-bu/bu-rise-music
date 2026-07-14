"""playlistgen CLI: generate a playlist from a free-text description.

Environment: Mac (or any machine with the built index). Loads only the CLAP
text encoder plus the saved index -- no cluster required.

    playlistgen "tropical beach sunset, laid-back, steel drums" -n 20
    playlistgen "focus music" --no-llm            # offline / ablation mode
    playlistgen "rainy jazz" --explain            # show captions + constraints
    playlistgen "synthwave night drive" --json    # machine-readable dump
"""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from .clap_model import ClapEncoder
from .config import load_config
from .expand import MissingAPIKeyError
from .pipeline import generate_playlist
from .retrieve import PlaylistIndex


def _print_explain(console: Console, result: dict) -> None:
    exp = result["expansion"]
    console.print("\n[bold]How the prompt was interpreted[/bold]")
    console.print(f"  source: {exp['source']}")
    for caption in exp["captions"]:
        console.print(f"  caption: [italic]{caption}[/italic]")
    for key in ("genres", "moods", "instruments"):
        if exp[key]:
            console.print(f"  {key}: {', '.join(exp[key])}")
    if exp["bpm_range"]:
        console.print(f"  bpm_range: {exp['bpm_range'][0]}-{exp['bpm_range'][1]}")
    if exp["energy"]:
        console.print(f"  energy: {exp['energy']}")
    console.print(f"  vocals: {exp['vocals']}")
    if exp["dropped_tags"]:
        console.print(f"  dropped out-of-vocab tags: {', '.join(exp['dropped_tags'])}")
    for note in result["notes"]:
        console.print(f"  [dim]note: {note}[/dim]")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="playlistgen",
        description="Generate a playlist from a free-text description via CLAP retrieval over MTG-Jamendo.",
    )
    ap.add_argument("prompt", help="free-text playlist description")
    ap.add_argument("-n", type=int, default=20, help="playlist length (default 20)")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip OpenAI query expansion; embed the raw prompt (offline/ablation)")
    ap.add_argument("--explain", action="store_true",
                    help="print the expanded captions and constraints used")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="dump the full result as JSON instead of a table")
    ap.add_argument("--config", default=None, help="config YAML (default: configs/default.yaml)")
    args = ap.parse_args(argv)

    console = Console(stderr=False)
    cfg = load_config(args.config)

    try:
        index = PlaylistIndex.load(cfg["paths"]["index_dir"])
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    encoder = ClapEncoder(cfg)
    try:
        result = generate_playlist(
            args.prompt, args.n, cfg, index, encoder, use_llm=not args.no_llm
        )
    except MissingAPIKeyError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    if args.as_json:
        print(json.dumps(result, indent=2))
        return 0

    table = Table(title=f"playlist for: {args.prompt!r}")
    table.add_column("#", justify="right")
    table.add_column("artist – title")
    table.add_column("sim", justify="right")
    table.add_column("matched tags")
    table.add_column("url", overflow="fold")
    for item in result["playlist"]:
        table.add_row(
            str(item["rank"]),
            f"{item['artist']} – {item['title']}",
            f"{item['similarity']:.3f}",
            ", ".join(item["matched_tags"]) or "—",
            item["stream_url"],
        )
    console.print(table)

    if args.explain:
        _print_explain(console, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
