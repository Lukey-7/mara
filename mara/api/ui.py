"""The single-page UI (plain HTML + JS, no build step), served from mara/static/."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(include_in_schema=False)
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@router.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))
