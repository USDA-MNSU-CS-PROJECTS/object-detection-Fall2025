"""CLI entry point for the Gradio object-detection UI (integration / production run)."""

from __future__ import annotations

import argparse
import os


def launch(*, port: int, inbrowser: bool) -> None:
    import main as ui  # defer import: path setup and Gradio `app` live in main

    ui.app.queue()
    ui.app.launch(inbrowser=inbrowser, server_port=port)


def _resolve_port(cli_port: int | None) -> int:
    """CLI wins; then GRADIO_SERVER_PORT; then PORT (common on hosted platforms); default 7860."""
    if cli_port is not None:
        return cli_port
    for key in ("GRADIO_SERVER_PORT", "PORT"):
        val = os.environ.get(key)
        if val:
            return int(val)
    return 7860


def main() -> None:
    parser = argparse.ArgumentParser(description="Alfalfa stem object detection (Gradio UI)")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="N",
        help="Server port (default: GRADIO_SERVER_PORT, PORT, or 7860).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser window on startup.",
    )
    args = parser.parse_args()
    port = _resolve_port(args.port)
    launch(port=port, inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()
