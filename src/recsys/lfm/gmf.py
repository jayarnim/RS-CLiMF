import torch
import torch.nn as nn
from .components.embedding.builder import embedding_builder
from .components.matching.builder import matching_fn_builder
from .components.prediction import ProjectionLayer


class Module(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        """
        Neural Collaborative Filtering (He et al., 2017)
        -----
        Implements the base structure of Generalized Matrix Factorization (GMF),
        MF & id embedding based latent factor model,
        sub-module of Neural Matrix Factorization (NeuMF)
        to learn low-rank linear representation.

        Args:
            num_users (int):
                total number of users in the dataset, U.
            num_items (int):
                total number of items in the dataset, I.
            embedding_dim (int):
                dimensionality of user and item latent representation vectors, K.
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.num_users = cfg.num_users
        self.num_items = cfg.num_items
        self.embedding_dim = cfg.embedding_dim
        self.pred_dim = cfg.embedding_dim

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        user_emb, item_emb = self.embedding(user_idx, item_idx)
        X_pred = self.matching(user_emb, item_emb)
        return X_pred

    def estimate(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair extracted logit (shape: [B,])
        """
        X_pred = self.forward(user_idx, item_idx)
        logit = self.prediction(X_pred)
        return logit

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            logit (torch.Tensor): (u,i) pair extracted logit (shape: [B,])
        """
        X_pred = self.forward(user_idx, item_idx)
        logit = self.prediction(X_pred)
        return logit

    def _set_up_components(self):
        self._create_components()

    def _create_components(self):
        kwargs = dict(
            name="idx",
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
        )
        self.embedding = embedding_builder(**kwargs)

        kwargs = dict(
            name="mf",
        )
        self.matching = matching_fn_builder(**kwargs)

        kwargs = dict(
            dim=self.embedding_dim,
        )
        self.prediction = ProjectionLayer(**kwargs)