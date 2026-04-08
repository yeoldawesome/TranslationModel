import argparse
import json
import os
import pathlib
import threading
import webbrowser
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Keep TensorFlow/Keras startup quiet for web usage.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import keras
from flask import Flask, render_template, request

from src.nmt_transformer.model import PositionalEmbedding, TransformerDecoder, TransformerEncoder
from src.nmt_transformer.preprocessing import load_vectorizers_from_vocab
from translate import configure_quiet_logging, decode_sequence


@dataclass
class LoadedModel:
    model: keras.Model
    eng_vectorization: object
    spa_vectorization: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run browser-based translation UI.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory with model and vocab files.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve.")
    parser.add_argument("--open-browser", action="store_true", help="Open browser automatically.")
    return parser.parse_args()


def collect_model_files(artifacts_dir: pathlib.Path, metadata: dict) -> List[str]:
    candidates: List[str] = []

    metadata_model = metadata.get("model_filename", "")
    if metadata_model and (artifacts_dir / metadata_model).exists():
        candidates.append(metadata_model)

    keras_models = sorted(artifacts_dir.glob("*.keras"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in keras_models:
        name = path.name
        if name not in candidates:
            candidates.append(name)

    if not candidates:
        raise FileNotFoundError(
            f"No .keras model files found in {artifacts_dir}. Train first or copy model artifacts here."
        )

    return candidates


def build_loader(artifacts_dir: pathlib.Path, metadata: dict):
    cache: Dict[str, LoadedModel] = {}

    def get_model_bundle(model_filename: str) -> LoadedModel:
        if model_filename in cache:
            return cache[model_filename]

        model_path = artifacts_dir / model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        eng_vectorization, spa_vectorization = load_vectorizers_from_vocab(
            artifacts_dir,
            vocab_size=metadata["vocab_size"],
            sequence_length=metadata["sequence_length"],
        )

        model = keras.models.load_model(
            model_path,
            custom_objects={
                "TransformerEncoder": TransformerEncoder,
                "TransformerDecoder": TransformerDecoder,
                "PositionalEmbedding": PositionalEmbedding,
            },
        )

        cache[model_filename] = LoadedModel(
            model=model,
            eng_vectorization=eng_vectorization,
            spa_vectorization=spa_vectorization,
        )
        return cache[model_filename]

    return get_model_bundle


def open_browser_prefer_chrome(url: str) -> None:
    try:
        possible_chrome_paths = [
            "C:/Program Files/Google/Chrome/Application/chrome.exe",
            "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
        ]
        chrome_path = next((p for p in possible_chrome_paths if pathlib.Path(p).exists()), None)
        if chrome_path:
            chrome = webbrowser.get(f'"{chrome_path}" %s')
            chrome.open_new(url)
            return
    except Exception:
        pass

    webbrowser.open_new(url)


def create_app(artifacts_dir: pathlib.Path) -> Tuple[Flask, List[str], dict]:
    configure_quiet_logging()

    metadata_path = artifacts_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    model_files = collect_model_files(artifacts_dir, metadata)
    get_model_bundle = build_loader(artifacts_dir, metadata)

    app = Flask(__name__, template_folder="web/templates", static_folder="web/static")

    @app.get("/")
    def index():
        default_model = model_files[0]
        return render_template(
            "index.html",
            model_files=model_files,
            selected_model=default_model,
            input_text="",
            translation_text="",
            error_text="",
            metadata=metadata,
        )

    @app.post("/translate")
    def translate_route():
        selected_model = request.form.get("model_filename", "").strip()
        input_text = request.form.get("input_text", "").strip()

        if selected_model not in model_files:
            selected_model = model_files[0]

        if not input_text:
            return render_template(
                "index.html",
                model_files=model_files,
                selected_model=selected_model,
                input_text=input_text,
                translation_text="",
                error_text="Enter an English sentence before translating.",
                metadata=metadata,
            )

        try:
            bundle = get_model_bundle(selected_model)
            max_length = int(metadata.get("sequence_length", 20))
            translation = decode_sequence(
                bundle.model,
                bundle.eng_vectorization,
                bundle.spa_vectorization,
                input_text,
                max_length=max_length,
            )
            error_text = ""
        except Exception as exc:
            translation = ""
            error_text = str(exc)

        return render_template(
            "index.html",
            model_files=model_files,
            selected_model=selected_model,
            input_text=input_text,
            translation_text=translation,
            error_text=error_text,
            metadata=metadata,
        )

    return app, model_files, metadata


def main() -> None:
    args = parse_args()
    artifacts_dir = pathlib.Path(args.artifacts_dir)

    app, model_files, metadata = create_app(artifacts_dir)

    print("Browser UI ready")
    print(f"Artifacts dir: {artifacts_dir}")
    print(f"Models found: {', '.join(model_files)}")
    print(f"Epochs in metadata: {metadata.get('epochs', 'unknown')}")

    if args.open_browser:
        url = f"http://{args.host}:{args.port}"
        timer = threading.Timer(1.0, lambda: open_browser_prefer_chrome(url))
        timer.daemon = True
        timer.start()

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
