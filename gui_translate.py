import argparse
import json
import os
import pathlib
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Dict, List

# Configure quiet TensorFlow/Keras logging before importing keras.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import keras

from src.nmt_transformer.model import PositionalEmbedding, TransformerDecoder, TransformerEncoder
from src.nmt_transformer.preprocessing import load_vectorizers_from_vocab
from translate import configure_quiet_logging, decode_sequence


@dataclass
class LoadedModel:
    model: keras.Model
    eng_vectorization: object
    spa_vectorization: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Desktop GUI for English to Spanish translation.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory with model and vocab files.")
    return parser.parse_args()


def collect_model_files(artifacts_dir: pathlib.Path, metadata: dict) -> List[str]:
    model_names: List[str] = []

    metadata_model = metadata.get("model_filename", "")
    if metadata_model and (artifacts_dir / metadata_model).exists():
        model_names.append(metadata_model)

    for model_path in sorted(artifacts_dir.glob("*.keras"), key=lambda p: p.stat().st_mtime, reverse=True):
        if model_path.name not in model_names:
            model_names.append(model_path.name)

    if not model_names:
        raise FileNotFoundError(
            f"No .keras model files found in {artifacts_dir}. Train first or copy a model into artifacts/."
        )

    return model_names


class TranslatorGui:
    def __init__(self, root: tk.Tk, artifacts_dir: pathlib.Path, metadata: dict, model_files: List[str]):
        self.root = root
        self.artifacts_dir = artifacts_dir
        self.metadata = metadata
        self.model_files = model_files
        self.model_cache: Dict[str, LoadedModel] = {}
        self.max_length = int(metadata.get("sequence_length", 20))

        self.root.title("NMT Desktop Translator")
        self.root.geometry("900x620")
        self.root.minsize(760, 520)

        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill="both", expand=True)

        title = ttk.Label(frame, text="English to Spanish Translator", font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w")

        subtitle = ttk.Label(
            frame,
            text="Select a trained model, enter English text, and translate using your Python model.",
        )
        subtitle.pack(anchor="w", pady=(2, 14))

        top_row = ttk.Frame(frame)
        top_row.pack(fill="x", pady=(0, 10))

        ttk.Label(top_row, text="Model:").pack(side="left")

        self.model_var = tk.StringVar(value=self.model_files[0])
        self.model_combo = ttk.Combobox(
            top_row,
            textvariable=self.model_var,
            state="readonly",
            values=self.model_files,
            width=58,
        )
        self.model_combo.pack(side="left", padx=(8, 10))

        self.translate_button = ttk.Button(top_row, text="Translate", command=self.translate_clicked)
        self.translate_button.pack(side="left")

        self.status_var = tk.StringVar(
            value=f"Epochs: {self.metadata.get('epochs', 'unknown')} | GPUs: {self.metadata.get('used_gpus', 'unknown')}"
        )
        status = ttk.Label(frame, textvariable=self.status_var)
        status.pack(anchor="w", pady=(0, 10))

        ttk.Label(frame, text="English input:").pack(anchor="w")
        self.input_text = tk.Text(frame, height=8, wrap="word")
        self.input_text.pack(fill="x", pady=(4, 12))

        ttk.Label(frame, text="Spanish output:").pack(anchor="w")
        self.output_text = tk.Text(frame, height=10, wrap="word")
        self.output_text.pack(fill="both", expand=True, pady=(4, 0))
        self.output_text.insert("1.0", "Translation output will appear here.")
        self.output_text.config(state="disabled")

    def get_model_bundle(self, model_filename: str) -> LoadedModel:
        if model_filename in self.model_cache:
            return self.model_cache[model_filename]

        model_path = self.artifacts_dir / model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        eng_vectorization, spa_vectorization = load_vectorizers_from_vocab(
            self.artifacts_dir,
            vocab_size=self.metadata["vocab_size"],
            sequence_length=self.metadata["sequence_length"],
        )

        model = keras.models.load_model(
            model_path,
            custom_objects={
                "TransformerEncoder": TransformerEncoder,
                "TransformerDecoder": TransformerDecoder,
                "PositionalEmbedding": PositionalEmbedding,
            },
        )

        bundle = LoadedModel(
            model=model,
            eng_vectorization=eng_vectorization,
            spa_vectorization=spa_vectorization,
        )
        self.model_cache[model_filename] = bundle
        return bundle

    def translate_clicked(self) -> None:
        input_sentence = self.input_text.get("1.0", "end").strip()
        if not input_sentence:
            messagebox.showerror("Missing input", "Enter an English sentence before translating.")
            return

        selected_model = self.model_var.get().strip()
        self.translate_button.config(state="disabled")
        self.status_var.set("Translating...")

        thread = threading.Thread(
            target=self._run_translation,
            args=(selected_model, input_sentence),
            daemon=True,
        )
        thread.start()

    def _run_translation(self, model_filename: str, input_sentence: str) -> None:
        try:
            bundle = self.get_model_bundle(model_filename)
            translation = decode_sequence(
                bundle.model,
                bundle.eng_vectorization,
                bundle.spa_vectorization,
                input_sentence,
                max_length=self.max_length,
            )
            self.root.after(0, lambda: self._on_success(translation))
        except Exception as exc:
            self.root.after(0, lambda: self._on_error(str(exc)))

    def _on_success(self, translation: str) -> None:
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.insert("1.0", translation)
        self.output_text.config(state="disabled")
        self.translate_button.config(state="normal")
        self.status_var.set(
            f"Ready | Epochs: {self.metadata.get('epochs', 'unknown')} | Replicas: {self.metadata.get('replicas', 'unknown')}"
        )

    def _on_error(self, error_message: str) -> None:
        self.translate_button.config(state="normal")
        self.status_var.set("Ready")
        messagebox.showerror("Translation failed", error_message)


def main() -> None:
    args = parse_args()
    artifacts_dir = pathlib.Path(args.artifacts_dir)

    configure_quiet_logging()

    metadata_path = artifacts_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    model_files = collect_model_files(artifacts_dir, metadata)

    root = tk.Tk()
    gui = TranslatorGui(root, artifacts_dir, metadata, model_files)
    _ = gui
    root.mainloop()


if __name__ == "__main__":
    main()
