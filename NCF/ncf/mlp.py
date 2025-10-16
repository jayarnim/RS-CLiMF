import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        hidden: list,
        dropout: float,
    ):
        """
        Neural Collaborative Filtering (He et al., 2017)
        -----
        Implements the base structure of Multi-Layer Perceptron (MLP),
        MLP & id embedding based latent factor model,
        sub-module of Neural Matrix Factorization (NeuMF)
        to learn high-rank nonlinear user-item interactions.

        Args:
            n_users (int):
                total number of users in the dataset, U.
            n_items (int):
                total number of items in the dataset, I.
            n_factors (int):
                dimensionality of user and item latent representation vectors, K.
            hidden (list):
                layer dimensions for the MLP-based matching function.
                (e.g., [64, 32, 16, 8])
            dropout (float):
                dropout rate applied to MLP layers for regularization.
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
        self.hidden = hidden
        self.dropout = dropout

        # debugging args error
        self._assert_arg_error()

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
        pred_vector = self.ncf(user_idx, item_idx)
        logit = self.pred_layer(pred_vector).squeeze(-1)
        return logit

    def ncf(self, user_idx, item_idx):
        user_embed_slice = self.user_embed(user_idx)
        item_embed_slice = self.item_embed(item_idx)

        kwargs = dict(
            tensors=(user_embed_slice, item_embed_slice), 
            dim=-1,
        )
        concat = torch.cat(**kwargs)
        pred_vector = self.matching_fn(concat)

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
        components = list(self._yield_layers(self.hidden))
        self.matching_fn = nn.Sequential(*components)

        kwargs = dict(
            in_features=self.hidden[-1],
            out_features=1,
        )
        self.pred_layer = nn.Linear(**kwargs)

    def _yield_layers(self, hidden):
        idx = 1
        while idx < len(hidden):
            yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.LayerNorm(hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(self.dropout)
            idx += 1

    def _assert_arg_error(self):
        CONDITION = (self.hidden[0] == self.n_factors * 2)
        ERROR_MESSAGE = f"First MLP layer must match input size: {self.n_factors * 2}"
        assert CONDITION, ERROR_MESSAGE