import pathlib
import random

import tensorflow as tf
from keras.utils import get_file


DATASET_URL = "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"


def load_text_pairs(seed: int = 42):
    text_zip = get_file("spa-eng.zip", origin=DATASET_URL, extract=True)
    datasets_dir = pathlib.Path(text_zip).parent
    matches = list(datasets_dir.rglob("spa.txt"))
    if not matches:
        raise FileNotFoundError(f"Could not find spa.txt under {datasets_dir}")
    text_path = matches[0]

    with open(text_path, "r", encoding="utf-8") as handle:
        lines = handle.read().split("\n")[:-1]

    pairs = []
    for line in lines:
        eng, spa = line.split("\t")
        spa = f"[start] {spa} [end]"
        pairs.append((eng, spa))

    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs


def split_pairs(pairs, val_split: float = 0.15):
    num_val = int(val_split * len(pairs))
    num_train = len(pairs) - (2 * num_val)

    train_pairs = pairs[:num_train]
    val_pairs = pairs[num_train : num_train + num_val]
    test_pairs = pairs[num_train + num_val :]
    return train_pairs, val_pairs, test_pairs


def make_dataset(pairs, format_fn, batch_size: int, shuffle_buffer: int, prefetch_size: int):
    eng_texts, spa_texts = zip(*pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(eng_texts), list(spa_texts)))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.cache().shuffle(shuffle_buffer).prefetch(prefetch_size)
