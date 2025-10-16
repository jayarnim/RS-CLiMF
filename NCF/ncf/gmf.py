import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
    ):
        """
        Neural Collaborative Filtering (He et al., 2017)
        -----
        Implements the base structure of Generalized Matrix Factorization (GMF),
        MF & id embedding based latent factor model,
        sub-module of Neural Matrix Factorization (NeuMF)
        to learn low-rank linear representation.

        Args:
            n_users (int):
                total number of users in the dataset, U.
            n_items (int):
                total number of items in the dataset, I.
            n_factors (int):
                dimensionality of user and item latent representation vectors, K.
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        return self.score(user_idx, item_idx)

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            prob (torch.Tensor): (u,i) pair interaction probability (shape: [B,])
        """
        logit = self.score(user_idx, item_idx)
        prob = torch.sigmoid(logit)
        return prob

    def score(self, user_idx, item_idx):
        pred_vector = self.gmf(user_idx, item_idx)
        logit = self.pred_layer(pred_vector).squeeze(-1)
        return logit

    def gmf(self, user_idx, item_idx):
        user_embed_slice = self.user_embed(user_idx)
        item_embed_slice = self.item_embed(item_idx)
        pred_vector = user_embed_slice * item_embed_slice
        return pred_vector

    def _set_up_components(self):
        self._create_embeddings()
        self._init_embeddings()
        self._create_layers()

    def _create_embeddings(self):
        kwargs = dict(
            num_embeddings=self.n_users+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_users,
        )
        self.user_embed = nn.Embedding(**kwargs)

        kwargs = dict(
            num_embeddings=self.n_items+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_items,
        )
        self.item_embed = nn.Embedding(**kwargs)

    def _init_embeddings(self):
        nn.init.normal_(self.user_embed.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embed.weight, mean=0.0, std=0.01)

    def _create_layers(self):
        kwargs = dict(
            in_features=self.n_factors,
            out_features=1,
        )
        self.pred_layer = nn.Linear(**kwargs)