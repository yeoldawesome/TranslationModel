import json
import re
import string

import tensorflow as tf
from keras.layers import TextVectorization


STRIP_CHARS = string.punctuation + "¿"
STRIP_CHARS = STRIP_CHARS.replace("[", "")
STRIP_CHARS = STRIP_CHARS.replace("]", "")


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, f"[{re.escape(STRIP_CHARS)}]", "")


def build_vectorizers(vocab_size: int, sequence_length: int):
    eng_vectorization = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    spa_vectorization = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length + 1,
        standardize=custom_standardization,
    )

    return eng_vectorization, spa_vectorization


def adapt_vectorizers(eng_vectorization, spa_vectorization, train_pairs):
    train_eng = [pair[0] for pair in train_pairs]
    train_spa = [pair[1] for pair in train_pairs]
    eng_vectorization.adapt(train_eng)
    spa_vectorization.adapt(train_spa)


def make_format_dataset_fn(eng_vectorization, spa_vectorization):
    def format_dataset(eng, spa):
        eng = eng_vectorization(eng)
        spa = spa_vectorization(spa)
        return (
            {
                "encoder_inputs": eng,
                "decoder_inputs": spa[:, :-1],
            },
            spa[:, 1:],
        )

    return format_dataset


def save_vocabularies(eng_vectorization, spa_vectorization, output_dir):
    eng_vocab = eng_vectorization.get_vocabulary()
    spa_vocab = spa_vectorization.get_vocabulary()

    with open(output_dir / "eng_vocab.json", "w", encoding="utf-8") as handle:
        json.dump(eng_vocab, handle, ensure_ascii=False)

    with open(output_dir / "spa_vocab.json", "w", encoding="utf-8") as handle:
        json.dump(spa_vocab, handle, ensure_ascii=False)


def load_vectorizers_from_vocab(vocab_dir, vocab_size: int, sequence_length: int):
    with open(vocab_dir / "eng_vocab.json", "r", encoding="utf-8") as handle:
        eng_vocab = json.load(handle)

    with open(vocab_dir / "spa_vocab.json", "r", encoding="utf-8") as handle:
        spa_vocab = json.load(handle)

    eng_vectorization, spa_vectorization = build_vectorizers(vocab_size, sequence_length)
    eng_vectorization.set_vocabulary(eng_vocab)
    spa_vectorization.set_vocabulary(spa_vocab)

    return eng_vectorization, spa_vectorization
