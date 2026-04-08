import argparse
import json
import os
import pathlib

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
from keras import ops

from src.nmt_transformer.model import PositionalEmbedding, TransformerDecoder, TransformerEncoder
from src.nmt_transformer.preprocessing import load_vectorizers_from_vocab


def parse_args():
    parser = argparse.ArgumentParser(description="Translate an English sentence to Spanish.")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Directory with trained model and vocab.")
    parser.add_argument("--sentence", type=str, required=True, help="English sentence to translate.")
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
    artifacts_dir = pathlib.Path(args.artifacts_dir)

    with open(artifacts_dir / "metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    eng_vectorization, spa_vectorization = load_vectorizers_from_vocab(
        artifacts_dir,
        vocab_size=metadata["vocab_size"],
        sequence_length=metadata["sequence_length"],
    )

    model = keras.models.load_model(
        artifacts_dir / "transformer_model.keras",
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
        args.sentence,
        max_length=args.max_length,
    )

    print("Input:", args.sentence)
    print("Translation:", translated)


if __name__ == "__main__":
    main()
