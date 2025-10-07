import torch
import torch.nn as nn
from . import gmf, mlp


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        hidden: list,
        dropout: float,
    ):
        super(Module, self).__init__()
        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.hidden = hidden
        self.dropout = dropout

        # generate layers
        self._init_layers()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        return self.score(user_idx, item_idx)

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        with torch.no_grad():
            logit = self.score(user_idx, item_idx)
            pred = torch.sigmoid(logit)
        return pred

    def score(self, user_idx, item_idx):
        pred_vector_gmf = self.gmf.gmf(user_idx, item_idx)
        pred_vector_mlp = self.mlp.ncf(user_idx, item_idx)

        kwargs = dict(
            tensors=(pred_vector_gmf, pred_vector_mlp), 
            dim=-1,
        )
        pred_vector = torch.cat(**kwargs)
        logit = self.logit_layer(pred_vector).squeeze(-1)

        return logit

    def _init_layers(self):
        kwargs = dict(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors // 2,
        )
        self.gmf = gmf.Module(**kwargs)

        kwargs = dict(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            hidden=self.hidden,
            dropout=self.dropout,
        )
        self.mlp = mlp.Module(**kwargs)

        kwargs = dict(
            in_features=self.n_factors//2 + self.hidden[-1],
            out_features=1,
        )
        self.logit_layer = nn.Linear(**kwargs)