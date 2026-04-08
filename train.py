import argparse
import json
import os
import pathlib

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import tensorflow as tf

from src.nmt_transformer.config import NMTConfig
from src.nmt_transformer.data import load_text_pairs, make_dataset, split_pairs
from src.nmt_transformer.model import build_transformer
from src.nmt_transformer.preprocessing import (
    adapt_vectorizers,
    build_vectorizers,
    make_format_dataset_fn,
    save_vocabularies,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train English-Spanish Transformer NMT model.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs. Use >= 30 for better quality.")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Directory for model and vocab files.")
    parser.add_argument(
        "--limit-pairs",
        type=int,
        default=0,
        help="Optional limit on total sentence pairs for quick experiments (0 = full dataset).",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default="",
        help="Optional path to spa.txt (useful on HPC nodes with restricted internet).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs to use. 0 = auto (use all visible GPUs).",
    )
    return parser.parse_args()


def select_strategy(requested_gpus: int):
    visible_gpus = tf.config.list_physical_devices("GPU")
    available_gpu_count = len(visible_gpus)

    if available_gpu_count == 0:
        return tf.distribute.get_strategy(), available_gpu_count, 0

    if requested_gpus > 1:
        used_gpu_count = min(requested_gpus, available_gpu_count)
        if used_gpu_count > 1:
            devices = [f"/gpu:{index}" for index in range(used_gpu_count)]
            return tf.distribute.MirroredStrategy(devices=devices), available_gpu_count, used_gpu_count
        return tf.distribute.get_strategy(), available_gpu_count, 1

    if requested_gpus == 1:
        return tf.distribute.get_strategy(), available_gpu_count, 1

    # Auto mode: if multiple GPUs are visible, use them all.
    if available_gpu_count > 1:
        return tf.distribute.MirroredStrategy(), available_gpu_count, available_gpu_count

    return tf.distribute.get_strategy(), available_gpu_count, 1


def main():
    args = parse_args()
    cfg = NMTConfig()

    strategy, available_gpu_count, used_gpu_count = select_strategy(args.num_gpus)
    print(f"Visible GPUs: {available_gpu_count}")
    print(f"Requested GPUs: {args.num_gpus} (0 means auto)")
    print(f"Using GPUs: {used_gpu_count}")
    print(f"Distribution replicas: {strategy.num_replicas_in_sync}")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_text_pairs(dataset_file=args.dataset_file or None)
    if args.limit_pairs > 0:
        pairs = pairs[: args.limit_pairs]

    train_pairs, val_pairs, test_pairs = split_pairs(pairs, val_split=cfg.val_split)

    eng_vectorization, spa_vectorization = build_vectorizers(
        vocab_size=cfg.vocab_size,
        sequence_length=cfg.sequence_length,
    )
    adapt_vectorizers(eng_vectorization, spa_vectorization, train_pairs)

    format_fn = make_format_dataset_fn(eng_vectorization, spa_vectorization)
    train_ds = make_dataset(
        train_pairs,
        format_fn,
        batch_size=cfg.batch_size,
        shuffle_buffer=cfg.shuffle_buffer,
        prefetch_size=cfg.prefetch_size,
    )
    val_ds = make_dataset(
        val_pairs,
        format_fn,
        batch_size=cfg.batch_size,
        shuffle_buffer=cfg.shuffle_buffer,
        prefetch_size=cfg.prefetch_size,
    )

    with strategy.scope():
        model = build_transformer(
            vocab_size=cfg.vocab_size,
            sequence_length=cfg.sequence_length,
            embed_dim=cfg.embed_dim,
            latent_dim=cfg.latent_dim,
            num_heads=cfg.num_heads,
        )

        model.compile(
            optimizer="rmsprop",
            loss=keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
            metrics=["accuracy"],
        )

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    model.save(output_dir / "transformer_model.keras")
    save_vocabularies(eng_vectorization, spa_vectorization, output_dir)

    metadata = {
        "vocab_size": cfg.vocab_size,
        "sequence_length": cfg.sequence_length,
        "batch_size": cfg.batch_size,
        "embed_dim": cfg.embed_dim,
        "latent_dim": cfg.latent_dim,
        "num_heads": cfg.num_heads,
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "num_test_pairs": len(test_pairs),
        "epochs": args.epochs,
        "available_gpus": available_gpu_count,
        "used_gpus": used_gpu_count,
        "replicas": int(strategy.num_replicas_in_sync),
        "final_train_accuracy": float(history.history.get("accuracy", [0])[-1]),
        "final_val_accuracy": float(history.history.get("val_accuracy", [0])[-1]),
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved model and artifacts to", output_dir)


if __name__ == "__main__":
    main()
