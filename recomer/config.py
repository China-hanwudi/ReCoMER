"""Configuration for the ReCoMER method reference implementation."""
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ReCoMERConfig:
    input_dims: tuple[int, int, int] = (768, 1024, 342)
    num_classes: int = 7
    d_model: int = 192
    num_heads: int = 6
    ff_dim: int = 576
    num_layers: int = 2
    history_k: int = 3
    adapter_rank: int = 32
    eta: float = 0.1
    routing_bound: float = 0.3
    dropout: float = 0.1
    admission_hidden: int = 32
    admission_temperature: float = 1.0
    null_beta: float = 1.0
    admission_threshold: float = 0.5
    include_history_tokens: bool = False
    weighted_pooling: bool = True
    use_attenuation: bool = True

    def __post_init__(self) -> None:
        self.input_dims = tuple(self.input_dims)
        if len(self.input_dims) != 3 or any(d <= 0 for d in self.input_dims):
            raise ValueError("input_dims must contain positive text/audio/vision dimensions")
        if self.num_classes < 2 or self.d_model <= 0 or self.num_heads <= 0:
            raise ValueError("classification needs at least two classes and positive model dimensions")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if min(self.ff_dim, self.num_layers, self.adapter_rank, self.admission_hidden) <= 0:
            raise ValueError("network widths and layer count must be positive")
        if self.history_k < 0 or self.eta < 0 or self.routing_bound < 0:
            raise ValueError("history_k, eta and routing_bound must be nonnegative")
        if not 0 <= self.dropout < 1 or not 0 <= self.admission_threshold <= 1:
            raise ValueError("invalid dropout or admission threshold")
        if self.admission_temperature <= 0 or self.null_beta < 0:
            raise ValueError("temperature must be positive and null_beta nonnegative")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ReCoMERConfig":
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
