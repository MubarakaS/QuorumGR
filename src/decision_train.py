import math
import random

import torch
import torch.nn.functional as F

from train import (
    set_seed,
    split_indices,
    build_known_group_items,
    make_group_ranking_batch,
)


# Freeze the representation model

def freeze_representation_model(model):
    """
    Freeze ALL QuorumGR representation parameters.

    The following remain fixed:
        user embeddings
        item embeddings
        group representation
        quality network
    """

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()


# Validation of collective-decision dynamics

@torch.no_grad()
def evaluate_decision_rates(
    model,
    decision_layer,
    data,
    indices,
    history,
    device,
    batch_size=256,
    num_negatives=100,
    seed=12345,
):
    """
    Evaluate final commitment scores.

    The positive item is always candidate 0.
    """

    model.eval()
    decision_layer.eval()

    rng = random.Random(seed)

    hr5_sum = 0.0
    hr10_sum = 0.0
    hr20_sum = 0.0

    ndcg5_sum = 0.0
    ndcg10_sum = 0.0
    ndcg20_sum = 0.0

    total = 0

    for start in range(
        0,
        len(indices),
        batch_size,
    ):

        batch_indices = indices[
            start:start + batch_size
        ]

        (
            member_ids,
            member_mask,
            candidate_items,
            _,
        ) = make_group_ranking_batch(
            batch_indices=batch_indices,
            data=data,
            padding_user_id=model.padding_user_id,
            history=history,
            num_negatives=num_negatives,
            rng=rng,
            device=device,
        )

        # Frozen representation model

        representation = model(
            member_ids,
            member_mask,
            candidate_items,
        )

        # Collective-decision dynamics

        decision = decision_layer(
            member_support=
                representation["member_support"],

            member_mask=
                member_mask,

            collective_support=
                representation["collective_support"],

            group_quality=
                representation["group_quality"],
        )

        scores = decision[
            "final_score"
        ]

        # Positive item is candidate 0

        positive_score = (
            scores[:, 0:1]
        )

        ranks = (
            1
            +
            (
                scores[:, 1:]
                > positive_score
            )
            .sum(dim=1)
        )

        # ----------------------------------------------------
        # HR / NDCG
        # ----------------------------------------------------

        for k in [5, 10, 20]:

            hits = (
                ranks <= k
            )

            hit_count = (
                hits.float()
                .sum()
                .item()
            )

            ndcg_values = torch.where(
                hits,

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

            ndcg_count = (
                ndcg_values
                .sum()
                .item()
            )

            if k == 5:
                hr5_sum += hit_count
                ndcg5_sum += ndcg_count

            elif k == 10:
                hr10_sum += hit_count
                ndcg10_sum += ndcg_count

            elif k == 20:
                hr20_sum += hit_count
                ndcg20_sum += ndcg_count

        total += len(
            batch_indices
        )

    total = max(
        total,
        1,
    )

    return {
        "HR@5":
            hr5_sum / total,

        "HR@10":
            hr10_sum / total,

        "HR@20":
            hr20_sum / total,

        "NDCG@5":
            ndcg5_sum / total,

        "NDCG@10":
            ndcg10_sum / total,

        "NDCG@20":
            ndcg20_sum / total,
    }


# Train biological decision rates

def train_collective_decision_rates(
    model,
    decision_layer,
    data,
    device,

    epochs=100,
    batch_size=256,

    learning_rate=1e-2,

    num_train_negatives=20,
    num_val_negatives=100,

    temperature=0.10,

    val_fraction=0.10,
    seed=42,
    patience=12,
):
    """
    Train ONLY:

        discovery strength
        recruitment strength
        abandonment strength
        inhibition strength

    The representation model remains completely frozen.

    Objective:
        listwise ranking loss over final commitment scores.
    """

    set_seed(seed)

    model = model.to(
        device
    )

    decision_layer = decision_layer.to(
        device
    )

    # Freeze representation model

    freeze_representation_model(
        model
    )

    # Decision rates remain trainable

    for parameter in decision_layer.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.Adam(
        decision_layer.parameters(),
        lr=learning_rate,
    )

    # Same group train-validation split used previously

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

    # Uses groupRatingTrain only.
    history = build_known_group_items(
        data
    )

    print(
        "Decision train / validation:",
        len(train_indices),
        "/",
        len(val_indices),
    )

    # Best checkpoint

    best_ndcg10 = -1.0
    best_epoch = 0
    best_state = None

    bad_epochs = 0

    # Training

    for epoch in range(
        1,
        epochs + 1,
    ):

        decision_layer.train()
        model.eval()

        # Deterministic shuffle
        epoch_rng = random.Random(
            seed + epoch
        )

        epoch_rng.shuffle(
            train_indices
        )

        # Epoch-specific negative sampler
        negative_rng = random.Random(
            seed
            + 5000 * epoch
        )

        total_loss = 0.0
        steps = 0

        for start in range(
            0,
            len(train_indices),
            batch_size,
        ):

            batch_indices = train_indices[
                start:start + batch_size
            ]

            (
                member_ids,
                member_mask,
                candidate_items,
                _,
            ) = make_group_ranking_batch(
                batch_indices=batch_indices,
                data=data,
                padding_user_id=
                    model.padding_user_id,
                history=history,
                num_negatives=
                    num_train_negatives,
                rng=negative_rng,
                device=device,
            )

            # Frozen representation signals

            with torch.no_grad():

                representation = model(
                    member_ids,
                    member_mask,
                    candidate_items,
                )

            # Trainable collective-decision dynamics

            decision = decision_layer(
                member_support=
                    representation[
                        "member_support"
                    ],

                member_mask=
                    member_mask,

                collective_support=
                    representation[
                        "collective_support"
                    ],

                group_quality=
                    representation[
                        "group_quality"
                    ],
            )

            final_score = decision[
                "final_score"
            ]

            
            # Ranking loss
            
            # Positive candidate is always column 0.

            decision_logits = (
                final_score
                /
                temperature
            )

            labels = torch.zeros(
                decision_logits.size(0),
                dtype=torch.long,
                device=device,
            )

            loss = F.cross_entropy(
                decision_logits,
                labels,
            )

            optimizer.zero_grad()

            loss.backward()

            # Four parameters only, but clipping protects
            # against unstable rate updates.
            torch.nn.utils.clip_grad_norm_(
                decision_layer.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            total_loss += (
                loss.item()
            )

            steps += 1

        # Validation

        metrics = evaluate_decision_rates(
            model=model,
            decision_layer=decision_layer,
            data=data,
            indices=val_indices,
            history=history,
            device=device,
            batch_size=256,
            num_negatives=
                num_val_negatives,
            seed=seed + 9999,
        )

        (
            discovery,
            recruitment,
            abandonment,
            inhibition,
        ) = decision_layer.get_rates()

        print(
            f"Epoch {epoch:03d} | "
            f"Loss={total_loss / steps:.6f} | "
            f"HR@10={metrics['HR@10']:.6f} | "
            f"NDCG@10={metrics['NDCG@10']:.6f} | "
            f"D={discovery.item():.4f} | "
            f"R={recruitment.item():.4f} | "
            f"A={abandonment.item():.4f} | "
            f"I={inhibition.item():.4f}"
        )

        # Select by validation NDCG@10

        if (
            metrics["NDCG@10"]
            >
            best_ndcg10
        ):

            best_ndcg10 = (
                metrics["NDCG@10"]
            )

            best_epoch = epoch

            bad_epochs = 0

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in decision_layer
                .state_dict()
                .items()
            }

        else:

            bad_epochs += 1

        if bad_epochs >= patience:

            print(
                "Early stopping."
            )

            break

    # Restore best biological rates

    if best_state is None:

        raise RuntimeError(
            "No collective-decision checkpoint was created."
        )

    decision_layer.load_state_dict(
        best_state
    )

    # Final best rates

    (
        discovery,
        recruitment,
        abandonment,
        inhibition,
    ) = decision_layer.get_rates()

    print()
    print(
        "Best collective-decision epoch:",
        best_epoch,
    )

    print(
        "Best collective-decision NDCG@10:",
        f"{best_ndcg10:.6f}",
    )

    print()
    print("Best learned rates:")
    print(
        "Discovery:",
        f"{discovery.item():.6f}",
    )
    print(
        "Recruitment:",
        f"{recruitment.item():.6f}",
    )
    print(
        "Abandonment:",
        f"{abandonment.item():.6f}",
    )
    print(
        "Inhibition:",
        f"{inhibition.item():.6f}",
    )

    return decision_layer