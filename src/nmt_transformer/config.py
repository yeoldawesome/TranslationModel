from dataclasses import dataclass


@dataclass
class NMTConfig:
    vocab_size: int = 15000
    sequence_length: int = 20
    batch_size: int = 64
    embed_dim: int = 256
    latent_dim: int = 2048
    num_heads: int = 8
    shuffle_buffer: int = 2048
    prefetch_size: int = 16
    val_split: float = 0.15

