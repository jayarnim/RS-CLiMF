from dataclasses import dataclass


@dataclass
class GMFCfg:
    num_users: int
    num_items: int
    embedding_dim: int


@dataclass
class MLPCfg:
    num_users: int
    num_items: int
    embedding_dim: int
    hidden_dim: list
    dropout: float


@dataclass
class RLNetCfg:
    num_users: int
    num_items: int
    projection_dim: int
    hidden_dim: list
    dropout: float


@dataclass
class MLNetCfg:
    num_users: int
    num_items: int
    projection_dim: int
    hidden_dim: list
    dropout: float