from ..config.model import (
    GMFCfg,
    MLPCfg,
    RLNetCfg,
    MLNetCfg,
)


def auto(cfg):
    model = cfg["model"]["name"]
    if model=="gmf":
        return gmf(cfg)
    elif model=="mlp":
        return mlp(cfg)
    elif model=="rlnet":
        return rlnet(cfg)
    elif model=="mlnet":
        return mlnet(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def gmf(cfg):
    return GMFCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        embedding_dim=cfg["model"]["embedding_dim"],
    )


def mlp(cfg):
    return MLPCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    )


def rlnet(cfg):
    return RLNetCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        projection_dim=cfg["model"]["projection_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    )


def mlnet(cfg):
    return MLNetCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        projection_dim=cfg["model"]["projection_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    )