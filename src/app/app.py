"""CLI entry point for the Gradio object-detection UI (integration / production run)."""

from __future__ import annotations

import argparse
import os


def launch(*, port: int, inbrowser: bool) -> None:
    import main as ui  # defer import: path setup and Gradio `app` live in main

    ui.app.queue()
    ui.app.launch(inbrowser=inbrowser, server_port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alfalfa stem object detection (Gradio UI)")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="N",
        help="Server port (default: GRADIO_SERVER_PORT environment variable or 7860).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser window on startup.",
    )
    args = parser.parse_args()
    if args.port is not None:
        port = args.port
    else:
        port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    launch(port=port, inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()
