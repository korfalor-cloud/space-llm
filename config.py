from dataclasses import dataclass, field
from typing import Optional
import json
import os


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    activation: str = "swiglu"  # swiglu, gelu, relu
    use_rope: bool = True
    tie_weights: bool = True
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def count_params(self) -> int:
        embed = self.vocab_size * self.d_model
        per_layer = 4 * self.d_model ** 2 + 2 * self.d_model * self.d_ff
        norms = self.n_layers * 4 * self.d_model
        return embed + self.n_layers * per_layer + norms

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainConfig:
    batch_size: int = 32
    grad_accum_steps: int = 4
    max_steps: int = 50000
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = True
    save_every: int = 5000
    eval_every: int = 1000
    log_every: int = 100
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data/tokenized"

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EvolveConfig:
    generations: int = 10
    train_steps_per_gen: int = 5000
    mutation_rate: float = 0.3
    mutations_per_gen: int = 3
    keep_top_k: int = 2
    evolve_log: str = "evolution/evolution_log.json"
    mutate_architecture: bool = True
    mutate_hyperparams: bool = True
    mutate_training: bool = True

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class GenerateConfig:
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    max_new_tokens: int = 256
    repetition_penalty: float = 1.1

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def save_config(config, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


def load_config(config_class, path: str):
    with open(path) as f:
        return config_class.from_dict(json.load(f))
