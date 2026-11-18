import math

import torch
import torch.nn as nn


class QuorumGR(nn.Module):
    """
    QuorumGR representation model.

    Current components:
        1. User embeddings
        2. Item embeddings
        3. User-item preference scoring
        4. Individual member-item support
        5. Mean + dispersion group representation
        6. Learned group-item quality
        7. Collective member support
        8. Quality-sensitive recruitment

    """

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64,
    ):

        
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # Padding user ID
        #
        # Real users:0 ... num_users - 1
        #
        # Padding: num_users
        self.padding_user_id = num_users

        # User and item embeddings
        self.user_embedding = nn.Embedding(
            num_users + 1,
            embedding_dim,
            padding_idx=self.padding_user_id,
        )

        self.item_embedding = nn.Embedding(
            num_items,
            embedding_dim,
        )

        # Group representation
        #
        # Input:
        #     mean(member embeddings) [D]
        #     std(member embeddings)  [D]
        #
        # Total input = 2D
        self.group_transform = nn.Linear(
            2 * embedding_dim,
            embedding_dim,
        )

        # Group-item quality network
        #
        
        self.quality_mlp = nn.Sequential(
            nn.Linear(
                3 * embedding_dim,
                embedding_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                embedding_dim,
                1,
            ),
        )

        self.reset_parameters()

    # Parameter initialization

    def reset_parameters(self):

        nn.init.xavier_uniform_(
            self.user_embedding.weight[:-1]
        )

        # Keep padding embedding equal to zero.
        with torch.no_grad():
            self.user_embedding.weight[
                self.padding_user_id
            ].zero_()

        nn.init.xavier_uniform_(
            self.item_embedding.weight
        )

        nn.init.xavier_uniform_(
            self.group_transform.weight
        )

        nn.init.zeros_(
            self.group_transform.bias
        )

        for layer in self.quality_mlp:

            if isinstance(
                layer,
                nn.Linear,
            ):
                nn.init.xavier_uniform_(
                    layer.weight
                )

                nn.init.zeros_(
                    layer.bias
                )

    # Single user-item preference
    

    def predict_user_preference(
        self,
        user_ids,
        item_ids,
    ):
        """
        Predict how much a single user prefers an item.

        - Retrieve the user and item embeddings.
        - Calculate their dot product
        - Divide by sqrt(embedding_dim) to control the score's scale.
        - Apply sigmoid to convert the score to a value between 0 and 1.

        Returns:
            logits: Raw scaled preference scores.
            preference: Scores between 0 and 1.
        """

        user_embeddings = self.user_embedding(
            user_ids
        )

        item_embeddings = self.item_embedding(
            item_ids
        )

        scale = math.sqrt(
            self.embedding_dim
        )

        logits = (
            user_embeddings
            *
            item_embeddings
        ).sum(dim=-1)

        logits = logits / scale

        preference = torch.sigmoid(
            logits
        )

        return (
            logits,
            preference,
        )

    
    # User against multiple candidate items

    def score_user_candidates(
        self,
        user_ids,
        item_ids,
    ):
        """
        Predict a user's preference for multiple candidate items.

        - Retrieve the user embedding and candidate item embeddings.
        - Use einsum to calculate the dot product between each user
        and all their candidate items.
        - Scale the scores by sqrt(embedding_dim).
        - Apply sigmoid to obtain preference scores between 0 and 1.

        Returns:
            logits: Raw scores for all candidate items.
            support: Preference scores for all candidate items.
        """

        user_embeddings = self.user_embedding(
            user_ids
        )

        

        item_embeddings = self.item_embedding(
            item_ids
        )

        

        scale = math.sqrt(
            self.embedding_dim
        )

        logits = torch.einsum(
            "bd,bcd->bc",
            user_embeddings,
            item_embeddings,
        )

        logits = logits / scale

        support = torch.sigmoid(
            logits
        )

        return (
            logits,
            support,
        )

    # Group representation

    def encode_group(
        self,
        member_ids,
        member_mask,
    ):
        """
        Build a group representation using both:
        mean(member embeddings) and std(member embeddings)

        Returns:
        member_embeddings: Individual member representations.
        group_embedding: Learned group representation.
        group_mean: Average member embedding.
        group_std: Variation among member embeddings.
        """

        member_embeddings = (
            self.user_embedding(
                member_ids
            )
        )

        mask = (
            member_mask
            .unsqueeze(-1)
        )

        group_size = (
            member_mask
            .sum(
                dim=1,
                keepdim=True,
            )
            .clamp(min=1.0)
        )


        member_sum = (
            member_embeddings
            *
            mask
        ).sum(dim=1)

        group_mean = (
            member_sum
            /
            group_size
        )


        centered = (
            member_embeddings
            -
            group_mean.unsqueeze(1)
        )

        centered = (
            centered
            *
            mask
        )

        group_variance = (
            centered
            .pow(2)
            .sum(dim=1)
            /
            group_size
        )

        group_std = torch.sqrt(
            group_variance
            + 1e-8
        )

        # Combine mean and dispersion

        group_features = torch.cat(
            [
                group_mean,
                group_std,
            ],
            dim=-1,
        )


        group_embedding = torch.tanh(
            self.group_transform(
                group_features
            )
        )

        return (
            member_embeddings,
            group_embedding,
            group_mean,
            group_std,
        )

    # Member-item support
    

    def compute_member_support(
        self,
        member_embeddings,
        member_mask,
        item_embeddings,
    ):
        """
        Calculate how strongly each member supports
        each candidate item.

    

        Returns:
            member_support: Each member's support for each item
            collective_support: Average group support for each item.
        """

        scale = math.sqrt(
            self.embedding_dim
        )

        # Calculate the dot product between every group member and
        # every candidate item, summing over embedding dimension d.
        member_logits = torch.einsum(
            "bgd,bcd->bgc",
            member_embeddings,
            item_embeddings,
        )

        member_logits = (
            member_logits
            /
            scale
        )

        member_support = torch.sigmoid(
            member_logits
        )

        # Remove padded members

        support_mask = (
            member_mask
            .unsqueeze(-1)
        )

        masked_support = (
            member_support
            *
            support_mask
        )

        group_size = (
            member_mask
            .sum(
                dim=1,
                keepdim=True,
            )
            .clamp(min=1.0)
        )

        

        collective_support = (
            masked_support
            .sum(dim=1)
            /
            group_size
        )

        return (
            member_support,
            collective_support,
        )

    # Group-item quality

    def compute_group_quality(
        self,
        group_embedding,
        item_embeddings,
    ):
        
        """
        Predict how suitable each candidate item is for the group.

        

        Returns:
            quality_logits: group-item quality scores.
            group_quality: Predicted group-item compatibility scores.
        """

        batch_size = (
            item_embeddings.size(0)
        )

        num_candidates = (
            item_embeddings.size(1)
        )

        
        # Expand group representation across candidates

        group_expanded = (
            group_embedding
            .unsqueeze(1)
            .expand(
                batch_size,
                num_candidates,
                self.embedding_dim,
            )
        )

        
        # Element-wise interaction

        interaction = (
            group_expanded
            *
            item_embeddings
        )

        # Build quality features

        quality_features = torch.cat(
            [
                group_expanded,
                item_embeddings,
                interaction,
            ],
            dim=-1,
        )

        
        # Quality logits
        

        quality_logits = (
            self.quality_mlp(
                quality_features
            )
            .squeeze(-1)
        )

        

        group_quality = torch.sigmoid(
            quality_logits
        )

        return (
            quality_logits,
            group_quality,
        )

    # Forward pass

    def forward(
        self,
        member_ids,
        member_mask,
        item_ids,
    ):
    
        """
        Perform the main forward pass of QuorumGR.


        Returns: A dictionary containing individual support, collective
        support, group quality, recruitment, and group representations.
        """

        # Group representation

        (
            member_embeddings,
            group_embedding,
            group_mean,
            group_std,
        ) = self.encode_group(
            member_ids,
            member_mask,
        )

        # Candidate item embeddings

        item_embeddings = (
            self.item_embedding(
                item_ids
            )
        )

        
        # Individual + collective member support

        (
            member_support,
            collective_support,
        ) = self.compute_member_support(
            member_embeddings,
            member_mask,
            item_embeddings,
        )

        # Learned group-item quality

        (
            quality_logits,
            group_quality,
        ) = self.compute_group_quality(
            group_embedding,
            item_embeddings,
        )

        # Current quality-sensitive recruitment
        

        recruitment = (
            group_quality
            *
            collective_support
        )

        return {
            "member_support":
                member_support,

            "collective_support":
                collective_support,

            "quality_logits":
                quality_logits,

            "group_quality":
                group_quality,

            "recruitment":
                recruitment,

            "group_embedding":
                group_embedding,

            "group_mean":
                group_mean,

            "group_std":
                group_std,
        }