import argparse
import json
import os
import pathlib
import sys
import warnings

# Set logging-related env vars before importing TensorFlow/Keras.
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
# Disable oneDNN custom-op logs that print at startup.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras
import tensorflow as tf
from keras import ops

try:
    from absl import logging as absl_logging
except ImportError:
    absl_logging = None

from src.nmt_transformer.model import PositionalEmbedding, TransformerDecoder, TransformerEncoder
from src.nmt_transformer.preprocessing import load_vectorizers_from_vocab


def configure_quiet_logging():
    # Hide known non-actionable warnings from TensorFlow/Keras during model load.
    warnings.filterwarnings(
        "ignore",
        message=r"`build\(\)` was called on layer.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"From .*tf\.placeholder is deprecated.*",
        category=UserWarning,
    )
    tf.get_logger().setLevel("ERROR")
    if absl_logging is not None:
        absl_logging.set_verbosity(absl_logging.ERROR)
        absl_logging.set_stderrthreshold("error")


def parse_args():
    parser = argparse.ArgumentParser(description="Translate an English sentence to Spanish.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory with trained model and vocab.")
    parser.add_argument("--sentence", type=str, default="", help="English sentence to translate.")
    parser.add_argument("--max-length", type=int, default=20, help="Maximum decoded sentence length.")
    return parser.parse_args()


def decode_sequence(model, eng_vectorization, spa_vectorization, input_sentence: str, max_length: int):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[start]"

    spa_vocab = spa_vectorization.get_vocabulary()
    spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))

    for step in range(max_length):
        tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]
        predictions = model(
            {
                "encoder_inputs": tokenized_input_sentence,
                "decoder_inputs": tokenized_target_sentence,
            },
            training=False,
        )

        sampled_token_index = ops.convert_to_numpy(ops.argmax(predictions[0, step, :])).item(0)
        sampled_token = spa_index_lookup.get(sampled_token_index, "[UNK]")
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break

    return decoded_sentence


def main():
    args = parse_args()
    configure_quiet_logging()
    artifacts_dir = pathlib.Path(args.artifacts_dir)

    sentence = args.sentence.strip()
    if not sentence:
        if sys.stdin.isatty():
            sentence = input("Enter an English sentence: ").strip()
        if not sentence:
            raise ValueError("No input sentence provided. Use --sentence or enter a sentence when prompted.")

    with open(artifacts_dir / "metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    model_filename = metadata.get("model_filename", "transformer_model.keras")
    model_path = artifacts_dir / model_filename
    if not model_path.exists():
        # Backward-compatible fallback for older artifact sets.
        model_path = artifacts_dir / "transformer_model.keras"

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

    translated = decode_sequence(
        model,
        eng_vectorization,
        spa_vectorization,
        sentence,
        max_length=args.max_length,
    )

    print("Input:", sentence)
    print("Translation:", translated)


if __name__ == "__main__":
    main()
