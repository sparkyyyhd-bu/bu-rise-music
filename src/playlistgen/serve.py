"""Barebones local web player for playlistgen.

Environment: Mac (query side). Localhost only, no auth, no state, no DB --
this is a demo surface, not a product. Start with:

    playlistgen-web            # http://127.0.0.1:8501

Routes:
    GET  /          -> web/index.html (plus /static/app.js, /static/style.css)
    POST /generate  -> {"prompt": str, "n": int, "use_llm": bool}
                       runs the exact same pipeline as the CLI (imported from
                       playlistgen.pipeline -- zero duplicated logic).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .clap_model import ClapEncoder
from .config import load_config
from .expand import MissingAPIKeyError
from .pipeline import generate_playlist
from .retrieve import PlaylistIndex

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    n: int = Field(default=20, ge=1, le=100)
    use_llm: bool = True


def create_app(config_path: str | None = None) -> FastAPI:
    app = FastAPI(title="playlistgen", docs_url=None, redoc_url=None)
    cfg = load_config(config_path)
    app.state.cfg = cfg
    app.state.index = PlaylistIndex.load(cfg["paths"]["index_dir"])
    app.state.encoder = ClapEncoder(cfg)

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.post("/generate")
    def generate(req: GenerateRequest) -> dict:
        try:
            return generate_playlist(
                req.prompt,
                req.n,
                app.state.cfg,
                app.state.index,
                app.state.encoder,
                use_llm=req.use_llm,
            )
        except MissingAPIKeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return app


def main() -> None:
    import uvicorn

    cfg = load_config()
    uvicorn.run(
        create_app(),
        host=cfg["serve"]["host"],   # localhost only by default
        port=int(cfg["serve"]["port"]),
    )


if __name__ == "__main__":
    main()
