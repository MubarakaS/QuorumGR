import math
import random

import torch
import torch.nn.functional as F


# Reproducibility

def set_seed(seed):
    """
    Set random seeds for reproducible experiments.

    - Control randomness in Python and PyTorch.
    - Set CUDA seeds when a GPU is available.
    - Help produce consistent results across repeated runs.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Deterministic train-validation split

def split_indices(
    n,
    val_fraction=0.10,
    seed=42,
):

    """
    Split interaction records into training and validation sets.

    - Shuffle the record indices using a fixed random seed.
    - Select a fraction of the records for validation.
    - Use the remaining records for training.

    Returns:
        train_indices: Indices of training records.
        val_indices: Indices of validation records.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    indices = torch.randperm(n,generator=generator,)

    val_size = int( n * val_fraction)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return (
        train_indices.tolist(),
        val_indices.tolist(),
    )


# Build knowN user-item interactions

def build_known_user_items(data):
    """
    Build user-item history using ONLY userRatingTrain.
    
    Build a history of items each user has interacted with.

    - Read user-item interactions from the training dataset.
    - Store each user's observed items in a set.
    - Use this history to avoid sampling known items as negatives.

    Returns:
        history: Dictionary mapping each user to their observed items.
    """
    

    history = {
        u: set()
        for u in range(data.num_users)
    }

    for user_id, item_id in zip(
        data.user_train.entity_ids,
        data.user_train.item_ids,
    ):
        history[user_id].add(
            item_id
        )

    return history


# Sample user negatives

def sample_user_negatives(
    user_id,
    positive_item,
    history,
    num_items,
    num_negatives,
    rng,
):
    """
    
    Sample negative candidate items for a user.

    - Randomly select items from the available items.
    - Exclude the current positive item.
    - Exclude items in the user's known training history.
    - Avoid selecting the same negative item more than once.

    Returns:
        negatives: List of sampled negative item IDs.
    """
    

    known_items = history.get(
        user_id,
        set(),
    )

    negatives = []
    used = set()

    while len(negatives) < num_negatives:

        item = rng.randrange(
            num_items
        )

        if (
            item != positive_item
            and item not in known_items
            and item not in used
        ):
            negatives.append(
                item
            )

            used.add(
                item
            )

    return negatives


# Build user ranking batch

def make_user_ranking_batch(
    batch_indices,
    data,
    history,
    num_negatives,
    rng,
    device,
):
    """
    
    Prepare a batch of user-item examples for training.

    - Retrieve each user's observed positive item and rating.
    - Normalize ratings by dividing by 100.
    - Sample negative items for each positive interaction.
    - Place the positive item first, followed by negative items.
    - Convert the batch into PyTorch tensors.

    Returns:
        user_ids: User IDs in the batch.
        candidate_items: Positive and negative candidate items.
        rating_targets: Normalized ratings of positive items.
    """

    user_ids = []
    candidate_items = []
    rating_targets = []

    for idx in batch_indices:

        user_id = (
            data.user_train.entity_ids[idx]
        )

        positive_item = (
            data.user_train.item_ids[idx]
        )

        rating = (
            data.user_train.ratings[idx]
            / 100.0
        )

        negatives = sample_user_negatives(
            user_id=user_id,
            positive_item=positive_item,
            history=history,
            num_items=data.num_items,
            num_negatives=num_negatives,
            rng=rng,
        )

        user_ids.append(
            user_id
        )

        candidate_items.append(
            [positive_item]
            + negatives
        )

        rating_targets.append(
            rating
        )

    user_ids = torch.tensor(
        user_ids,
        dtype=torch.long,
        device=device,
    )

    candidate_items = torch.tensor(
        candidate_items,
        dtype=torch.long,
        device=device,
    )

    rating_targets = torch.tensor(
        rating_targets,
        dtype=torch.float32,
        device=device,
    )

    return (
        user_ids,
        candidate_items,
        rating_targets,
    )


# User ranking validation

@torch.no_grad()
def evaluate_user_ranking(
    model,
    data,
    indices,
    history,
    device,
    batch_size=256,
    num_negatives=100,
    seed=12345,
):

    """
    Evaluate how well the model ranks items for individual users.

    - Generate candidate items for each validation interaction.
    - Predict user-item preference scores.
    - Determine the rank of the positive item.
    - Calculate Hit Rate@10 and NDCG@10.
    - Calculate rating prediction error using MSE.

    Returns:
        A dictionary containing HR@10, NDCG@10, and rating MSE.
    """
    model.eval()

    rng = random.Random(
        seed
    )

    hr10_sum = 0.0
    ndcg10_sum = 0.0
    rating_error_sum = 0.0

    total = 0

    for start in range(
        0,
        len(indices),
        batch_size,
    ):

        batch_indices = indices[
            start:
            start + batch_size
        ]

        (
            user_ids,
            candidate_items,
            rating_targets,
        ) = make_user_ranking_batch(
            batch_indices=batch_indices,
            data=data,
            history=history,
            num_negatives=num_negatives,
            rng=rng,
            device=device,
        )

        logits, support = (
            model.score_user_candidates(
                user_ids,
                candidate_items,
            )
        )

        # Positive item is always candidate 0
        

        positive_scores = (
            logits[:, 0:1]
        )

        ranks = (
            1
            +
            (
                logits[:, 1:]
                > positive_scores
            )
            .sum(dim=1)
        )

        hit10 = (
            ranks <= 10
        )

        hr10_sum += (
            hit10
            .float()
            .sum()
            .item()
        )

        ndcg_values = torch.where(
            hit10,

            1.0
            /
            torch.log2(
                ranks.float()
                + 1.0
            ),

            torch.zeros_like(
                ranks,
                dtype=torch.float32,
            ),
        )

        ndcg10_sum += (
            ndcg_values
            .sum()
            .item()
        )

        # Rating prediction for positive item

        positive_support = (
            support[:, 0]
        )

        rating_error_sum += (
            F.mse_loss(
                positive_support,
                rating_targets,
                reduction="sum",
            )
            .item()
        )

        total += len(
            batch_indices
        )

    return {
        "hr10":
            hr10_sum
            / max(total, 1),

        "ndcg10":
            ndcg10_sum
            / max(total, 1),

        "rating_mse":
            rating_error_sum
            / max(total, 1),
    }


# Stage A:
# Ranking-aware user-preference training

def train_user_preference(
    model,
    data,
    device,

    epochs=50,
    batch_size=512,
    learning_rate=1e-3,

    num_train_negatives=20,
    num_val_negatives=100,

    rank_loss_weight=1.0,
    rating_loss_weight=0.25,

    val_fraction=0.10,
    seed=42,
    patience=8,
):
    """
    Train the user and item embeddings (Stage A).

    - Split user interactions into training and validation sets.
    - Sample negative items for each positive interaction.
    - Predict user preferences for candidate items.
    - Calculate ranking loss using cross-entropy.
    - Calculate rating prediction loss using MSE.
    - Combine both losses and update embeddings using Adam.
    - Evaluate performance after each epoch.
    - Stop early if validation NDCG@10 stops improving.
    - Restore the model parameters from the best epoch.

    Returns:
        model: Model with trained user and item embeddings.
    """

    set_seed(
        seed
    )

    model = model.to(
        device
    )

    # Only U and V are optimized in Stage A

    optimizer = torch.optim.Adam(
        [
            {
                "params":
                    model.user_embedding.parameters()
            },
            {
                "params":
                    model.item_embedding.parameters()
            },
        ],
        lr=learning_rate,
        weight_decay=1e-5,
    )

    (
        train_indices,
        val_indices,
    ) = split_indices(
        len(
            data.user_train.ratings
        ),
        val_fraction=val_fraction,
        seed=seed,
    )

    history = build_known_user_items(
        data
    )

    print(
        "User train / validation:",
        len(train_indices),
        "/",
        len(val_indices),
    )

    best_ndcg10 = -1.0
    best_state = None
    best_epoch = 0

    bad_epochs = 0

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        # epoch shuffle
        epoch_rng = random.Random(
            seed + epoch
        )

        epoch_rng.shuffle(
            train_indices
        )

        negative_rng = random.Random(
            seed
            + 1000 * epoch
        )

        epoch_loss = 0.0
        steps = 0

        for start in range(
            0,
            len(train_indices),
            batch_size,
        ):

            batch_indices = train_indices[
                start:
                start + batch_size
            ]

            (
                user_ids,
                candidate_items,
                rating_targets,
            ) = make_user_ranking_batch(
                batch_indices=batch_indices,
                data=data,
                history=history,
                num_negatives=
                    num_train_negatives,
                rng=negative_rng,
                device=device,
            )

            optimizer.zero_grad()

            logits, support = (
                model.score_user_candidates(
                    user_ids,
                    candidate_items,
                )
            )

            # Ranking target
            #
            # Candidate 0 is always positive

            labels = torch.zeros(
                logits.size(0),
                dtype=torch.long,
                device=device,
            )

            rank_loss = (
                F.cross_entropy(
                    logits,
                    labels,
                )
            )

            
            # Explicit rating loss

            positive_support = (
                support[:, 0]
            )

            rating_loss = (
                F.mse_loss(
                    positive_support,
                    rating_targets,
                )
            )

            loss = (
                rank_loss_weight
                *
                rank_loss
                +
                rating_loss_weight
                *
                rating_loss
            )

            loss.backward()

            optimizer.step()

            epoch_loss += (
                loss.item()
            )

            steps += 1

        # Validation

        metrics = evaluate_user_ranking(
            model=model,
            data=data,
            indices=val_indices,
            history=history,
            device=device,
            batch_size=256,
            num_negatives=
                num_val_negatives,
            seed=seed + 9999,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train={epoch_loss / steps:.6f} | "
            f"Rating MSE={metrics['rating_mse']:.6f} | "
            f"HR@10={metrics['hr10']:.6f} | "
            f"NDCG@10={metrics['ndcg10']:.6f}"
        )

        # Select by validation NDCG@10

        if (
            metrics["ndcg10"]
            >
            best_ndcg10
        ):

            best_ndcg10 = (
                metrics["ndcg10"]
            )

            best_epoch = epoch

            bad_epochs = 0

            best_state = {
                key:
                    value
                    .detach()
                    .cpu()
                    .clone()

                for key, value
                in model.state_dict().items()
            }

        else:

            bad_epochs += 1

        if bad_epochs >= patience:

            print(
                "Early stopping."
            )

            break

    # Restore best model

    if best_state is None:
        raise RuntimeError(
            "No valid Stage-A checkpoint was created."
        )

    model.load_state_dict(
        best_state
    )

    print()

    print(
        "Best user-preference epoch:",
        best_epoch,
    )

    print(
        "Best user validation NDCG@10:",
        f"{best_ndcg10:.6f}",
    )

    return model


# GROUP-LEVEL UTILITIES

def build_known_group_items(data):
    """
    Build group-item history using only groupRatingTrain.
    """

    history = {
        g: set()
        for g in data.group_members
    }

    for group_id, item_id in zip(
        data.group_train.entity_ids,
        data.group_train.item_ids,
    ):
        history.setdefault(
            group_id,
            set(),
        ).add(
            item_id
        )

    return history


def sample_group_negatives(
    group_id,
    positive_item,
    history,
    num_items,
    num_negatives,
    rng,
):

    """
    Sample negative candidate items for a group.

    Returns:
        negatives: List of sampled negative item IDs.
    """

    known_items = history.get(
        group_id,
        set(),
    )

    negatives = []
    used = set()

    while len(negatives) < num_negatives:

        item = rng.randrange(
            num_items
        )

        if (
            item != positive_item
            and item not in known_items
            and item not in used
        ):
            negatives.append(
                item
            )

            used.add(
                item
            )

    return negatives


def make_group_ranking_batch(
    batch_indices,
    data,
    padding_user_id,
    history,
    num_negatives,
    rng,
    device,
):

    """
    Prepare a batch of group-item examples for training.

    - Retrieve group IDs, positive items, and observed ratings.
    - Normalize ratings by dividing by 100.
    - Retrieve the members belonging to each group.
    - Pad smaller groups so all groups have the same tensor size.
    - Create a mask to distinguish real members from padding.
    - Sample negative candidate items for each group.
    - Place the positive item first, followed by negative items.
    - Convert the data into PyTorch tensors.

    Returns:
        member_ids: Padded group member IDs.
        member_mask: Mask identifying valid group members.
        candidates: Positive and negative candidate items.
        ratings: Normalized group ratings.
    """

    group_ids = [
        data.group_train.entity_ids[i]
        for i in batch_indices
    ]

    positive_items = [
        data.group_train.item_ids[i]
        for i in batch_indices
    ]

    ratings = [
        data.group_train.ratings[i]
        / 100.0
        for i in batch_indices
    ]

    # Build padded member tensors

    batch_members = [
        data.group_members[g]
        for g in group_ids
    ]

    max_group_size = max(
        len(members)
        for members in batch_members
    )

    member_ids = torch.full(
        (
            len(batch_indices),
            max_group_size,
        ),
        padding_user_id,
        dtype=torch.long,
    )

    member_mask = torch.zeros(
        (
            len(batch_indices),
            max_group_size,
        ),
        dtype=torch.float32,
    )

    for row, members in enumerate(
        batch_members
    ):

        size = len(
            members
        )

        member_ids[
            row,
            :size
        ] = torch.tensor(
            members,
            dtype=torch.long,
        )

        member_mask[
            row,
            :size
        ] = 1.0

    # Candidate items

    candidates = []

    for group_id, positive_item in zip(
        group_ids,
        positive_items,
    ):

        negatives = sample_group_negatives(
            group_id=group_id,
            positive_item=positive_item,
            history=history,
            num_items=data.num_items,
            num_negatives=num_negatives,
            rng=rng,
        )

        candidates.append(
            [positive_item]
            + negatives
        )

    return (
        member_ids.to(
            device
        ),

        member_mask.to(
            device
        ),

        torch.tensor(
            candidates,
            dtype=torch.long,
            device=device,
        ),

        torch.tensor(
            ratings,
            dtype=torch.float32,
            device=device,
        ),
    )


# GROUP QUALITY VALIDATION

@torch.no_grad()
def evaluate_group_quality(
    model,
    data,
    indices,
    history,
    device,
    batch_size=256,
    num_negatives=100,
    seed=12345,
):

    """
    Evaluate how well the model predicts group-item quality.

    - Prepare group members and candidate items for validation.
    - Predict quality scores for each group-item pair.
    - Determine the rank of the positive item.
    - Calculate Hit Rate@10 and NDCG@10.
    - Compare predicted quality with observed group ratings using MSE.

    Returns:
        A dictionary containing HR@10, NDCG@10, and rating MSE.
    """

    model.eval()

    rng = random.Random(
        seed
    )

    hr10_sum = 0.0
    ndcg10_sum = 0.0
    rating_error_sum = 0.0

    total = 0

    for start in range(
        0,
        len(indices),
        batch_size,
    ):

        batch_indices = indices[
            start:
            start + batch_size
        ]

        (
            member_ids,
            member_mask,
            candidate_items,
            rating_targets,
        ) = make_group_ranking_batch(
            batch_indices=batch_indices,
            data=data,
            padding_user_id=
                model.padding_user_id,
            history=history,
            num_negatives=num_negatives,
            rng=rng,
            device=device,
        )

        output = model(
            member_ids,
            member_mask,
            candidate_items,
        )

        logits = output[
            "quality_logits"
        ]

        positive_scores = (
            logits[:, 0:1]
        )

        ranks = (
            1
            +
            (
                logits[:, 1:]
                > positive_scores
            )
            .sum(dim=1)
        )

        hit10 = (
            ranks <= 10
        )

        hr10_sum += (
            hit10.float()
            .sum()
            .item()
        )

        ndcg_values = torch.where(
            hit10,

            1.0
            /
            torch.log2(
                ranks.float()
                + 1.0
            ),

            torch.zeros_like(
                ranks,
                dtype=torch.float32,
            ),
        )

        ndcg10_sum += (
            ndcg_values.sum().item()
        )

        positive_quality = (
            output[
                "group_quality"
            ][:, 0]
        )

        rating_error_sum += (
            F.mse_loss(
                positive_quality,
                rating_targets,
                reduction="sum",
            ).item()
        )

        total += len(
            batch_indices
        )

    return {
        "hr10":
            hr10_sum
            / max(total, 1),

        "ndcg10":
            ndcg10_sum
            / max(total, 1),

        "rating_mse":
            rating_error_sum
            / max(total, 1),
    }


# STAGE B:
# GROUP-QUALITY TRAINING

def train_group_quality(
    model,
    data,
    device,

    epochs=50,
    batch_size=512,
    learning_rate=1e-3,

    num_train_negatives=20,
    num_val_negatives=100,

    rank_loss_weight=1.0,
    rating_loss_weight=0.25,

    val_fraction=0.10,
    seed=42,
    patience=8,
):
    """
    Train the group representation and quality network (Stage B).

    - Freeze the user and item embeddings learned in Stage A.
    - Enable training of the group transformation and quality MLP.
    - Split group interactions into training and validation sets.
    - Prepare group members, masks, and candidate items.
    - Predict group-item quality scores.
    - Calculate ranking loss using cross-entropy.
    - Calculate rating prediction loss using MSE.
    - Combine both losses and update the group-quality parameters.
    - Evaluate performance after each epoch.
    - Stop early if validation NDCG@10 stops improving.
    - Restore the model parameters from the best epoch.

    Returns:
        model: Model with trained group-quality components.
    """

    # Freeze user and item embeddings

    # Freeze Stage-A user and item embeddings

    model.user_embedding.weight.requires_grad = False
    model.item_embedding.weight.requires_grad = False

    # Re-enable Stage-B group-quality parameters
    

    for parameter in model.group_transform.parameters():
        parameter.requires_grad = True

    for parameter in model.quality_mlp.parameters():
        parameter.requires_grad = True

    model = model.to(
        device
    )

    # Only group-quality parameters are optimized

    optimizer = torch.optim.Adam(
        list(
            model.group_transform.parameters()
        )
        +
        list(
            model.quality_mlp.parameters()
        ),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    (
        train_indices,
        val_indices,
    ) = split_indices(
        len(
            data.group_train.ratings
        ),
        val_fraction=val_fraction,
        seed=seed + 1,
    )

    history = build_known_group_items(
        data
    )

    print(
        "Group train / validation:",
        len(train_indices),
        "/",
        len(val_indices),
    )

    best_ndcg10 = -1.0
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        epoch_rng = random.Random(
            seed + epoch
        )

        epoch_rng.shuffle(
            train_indices
        )

        negative_rng = random.Random(
            seed
            + 2000 * epoch
        )

        epoch_loss = 0.0
        steps = 0

        for start in range(
            0,
            len(train_indices),
            batch_size,
        ):

            batch_indices = train_indices[
                start:
                start + batch_size
            ]

            (
                member_ids,
                member_mask,
                candidate_items,
                rating_targets,
            ) = make_group_ranking_batch(
                batch_indices=
                    batch_indices,
                data=data,
                padding_user_id=
                    model.padding_user_id,
                history=history,
                num_negatives=
                    num_train_negatives,
                rng=negative_rng,
                device=device,
            )

            optimizer.zero_grad()

            output = model(
                member_ids,
                member_mask,
                candidate_items,
            )

            logits = output[
                "quality_logits"
            ]

            labels = torch.zeros(
                logits.size(0),
                dtype=torch.long,
                device=device,
            )

            # Ranking objective

            rank_loss = (
                F.cross_entropy(
                    logits,
                    labels,
                )
            )

            # Explicit rating objective

            positive_quality = (
                output[
                    "group_quality"
                ][:, 0]
            )

            rating_loss = (
                F.mse_loss(
                    positive_quality,
                    rating_targets,
                )
            )

            loss = (
                rank_loss_weight
                *
                rank_loss
                +
                rating_loss_weight
                *
                rating_loss
            )

            loss.backward()

            optimizer.step()

            epoch_loss += (
                loss.item()
            )

            steps += 1

        metrics = evaluate_group_quality(
            model=model,
            data=data,
            indices=val_indices,
            history=history,
            device=device,
            batch_size=256,
            num_negatives=
                num_val_negatives,
            seed=seed + 9999,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train={epoch_loss / steps:.6f} | "
            f"Rating MSE={metrics['rating_mse']:.6f} | "
            f"HR@10={metrics['hr10']:.6f} | "
            f"NDCG@10={metrics['ndcg10']:.6f}"
        )

        if (
            metrics["ndcg10"]
            >
            best_ndcg10
        ):

            best_ndcg10 = (
                metrics["ndcg10"]
            )

            best_epoch = epoch
            bad_epochs = 0

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in model.state_dict().items()
            }

        else:

            bad_epochs += 1

        if bad_epochs >= patience:

            print(
                "Early stopping."
            )

            break

    if best_state is None:
        raise RuntimeError(
            "No valid Stage-B checkpoint was created."
        )

    model.load_state_dict(
        best_state
    )

    print()

    print(
        "Best group-quality epoch:",
        best_epoch,
    )

    print(
        "Best group validation NDCG@10:",
        f"{best_ndcg10:.6f}",
    )

    return model

