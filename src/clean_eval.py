import math
import random

import torch


# Build known group-item interactions

def build_group_train_history(data):
    """
    Use ONLY groupRatingTrain.

    groupRatingTest is never inspected here.
    """

    history = {}

    for group_id, item_id in zip(
        data.group_train.entity_ids,
        data.group_train.item_ids,
    ):
        history.setdefault(
            group_id,
            set(),
        ).add(item_id)

    return history


# Create FIXED validation candidate sets

def build_fixed_validation_candidates(
    data,
    val_indices,
    num_negatives=100,
    seed=42,
):
    """
    Precompute one fixed candidate set for every validation row.

    Candidate 0 is always the positive item.

    Negatives:
        - are unique
        - are not the positive item
        - are not observed group-training items
        - never use groupRatingTest
    """

    rng = random.Random(seed)

    history = build_group_train_history(
        data
    )

    all_items = list(
        range(data.num_items)
    )

    fixed_candidates = []

    for idx in val_indices:

        group_id = (
            data.group_train.entity_ids[idx]
        )

        positive_item = (
            data.group_train.item_ids[idx]
        )

        forbidden = history.get(
            group_id,
            set(),
        )

        negatives = []
        used = set()

        while len(negatives) < num_negatives:

            item_id = rng.choice(
                all_items
            )

            if item_id == positive_item:
                continue

            if item_id in forbidden:
                continue

            if item_id in used:
                continue

            negatives.append(
                item_id
            )

            used.add(
                item_id
            )

        fixed_candidates.append(
            (
                idx,
                group_id,
                positive_item,
                [positive_item] + negatives,
            )
        )

    return fixed_candidates


# ============================================================
# Build padded group batch
# ============================================================

def make_group_batch(
    query_batch,
    data,
    padding_user_id,
    device,
):
    """
    query_batch contains:
        (idx, group_id, positive_item, candidate_items)
    """

    batch_size = len(
        query_batch
    )

    max_group_size = max(
        len(
            data.group_members[
                group_id
            ]
        )
        for _, group_id, _, _
        in query_batch
    )

    num_candidates = len(
        query_batch[0][3]
    )

    member_ids = torch.full(
        (
            batch_size,
            max_group_size,
        ),
        padding_user_id,
        dtype=torch.long,
        device=device,
    )

    member_mask = torch.zeros(
        (
            batch_size,
            max_group_size,
        ),
        dtype=torch.float32,
        device=device,
    )

    candidate_items = torch.empty(
        (
            batch_size,
            num_candidates,
        ),
        dtype=torch.long,
        device=device,
    )

    for row, (
        _,
        group_id,
        _,
        candidates,
    ) in enumerate(query_batch):

        members = (
            data.group_members[
                group_id
            ]
        )

        size = len(
            members
        )

        member_ids[
            row,
            :size
        ] = torch.tensor(
            members,
            dtype=torch.long,
            device=device,
        )

        member_mask[
            row,
            :size
        ] = 1.0

        candidate_items[
            row
        ] = torch.tensor(
            candidates,
            dtype=torch.long,
            device=device,
        )

    return (
        member_ids,
        member_mask,
        candidate_items,
    )


# ============================================================
# Metric accumulator
# ============================================================

def initialize_metrics(
    score_names,
    ks,
):
    results = {}

    for name in score_names:

        results[name] = {}

        for k in ks:

            results[name][
                f"HR@{k}"
            ] = 0.0

            results[name][
                f"NDCG@{k}"
            ] = 0.0

    return results


# Clean fixed-candidate evaluator

@torch.no_grad()
def evaluate_fixed_validation(
    model,
    decision_layer,
    data,
    fixed_candidates,
    device,
    batch_size=256,
    ks=(5, 10, 20),
):
    """
    Evaluate all components on EXACTLY the same
    precomputed candidate sets.

    Scores:
        quality
        support
        commitment
        quorum
    """

    model.eval()
    decision_layer.eval()

    score_names = [
        "quality",
        "support",
        "commitment",
        "quorum",
    ]

    results = initialize_metrics(
        score_names,
        ks,
    )

    total = 0

    for start in range(
        0,
        len(fixed_candidates),
        batch_size,
    ):

        query_batch = (
            fixed_candidates[
                start:
                start + batch_size
            ]
        )

        (
            member_ids,
            member_mask,
            candidate_items,
        ) = make_group_batch(
            query_batch,
            data,
            model.padding_user_id,
            device,
        )

        # Frozen representation model

        output = model(
            member_ids,
            member_mask,
            candidate_items,
        )

        # Collective decision layer

        decision = decision_layer(
            member_support=
                output[
                    "member_support"
                ],

            member_mask=
                member_mask,

            collective_support=
                output[
                    "collective_support"
                ],

            group_quality=
                output[
                    "group_quality"
                ],
        )

        scores = {
            "quality":
                output[
                    "group_quality"
                ],

            "support":
                output[
                    "collective_support"
                ],

            "commitment":
                decision[
                    "normalized_commitment"
                ],

            "quorum":
                decision[
                    "quorum_score"
                ],
        }

        # ----------------------------------------------------
        # Positive candidate is column 0
        # ----------------------------------------------------

        for name, score_matrix in scores.items():

            positive_score = (
                score_matrix[:, 0:1]
            )

            ranks = (
                1
                +
                (
                    score_matrix[:, 1:]
                    >
                    positive_score
                )
                .sum(dim=1)
            )

            for k in ks:

                hits = (
                    ranks <= k
                )

                results[name][
                    f"HR@{k}"
                ] += (
                    hits.float()
                    .sum()
                    .item()
                )

                ndcg = torch.where(
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

                results[name][
                    f"NDCG@{k}"
                ] += (
                    ndcg.sum().item()
                )

        total += len(
            query_batch
        )

    # --------------------------------------------------------
    # Average
    # --------------------------------------------------------

    for name in score_names:

        for metric in results[name]:

            results[name][metric] /= (
                total
            )

    return results

@torch.no_grad()
def evaluate_quorum_thresholds(
    model,
    decision_layer,
    data,
    fixed_candidates,
    device,
    thresholds=(
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
    ),
    batch_size=256,
    ks=(5, 10, 20),
):
    """
    Evaluate quorum-triggered stopping.

    For each group:

        - run the existing dynamics
        - inspect commitment after every step
        - stop at the FIRST step where the leading
          candidate reaches the quorum threshold
        - if no candidate reaches quorum, use the final step

    The selected commitment vector is then used
    for ranking.
    """

    model.eval()
    decision_layer.eval()

    results = {}

    for threshold in thresholds:

        results[threshold] = {
            f"HR@{k}": 0.0
            for k in ks
        }

        for k in ks:
            results[threshold][
                f"NDCG@{k}"
            ] = 0.0

        results[threshold][
            "mean_stop_step"
        ] = 0.0

        results[threshold][
            "quorum_rate"
        ] = 0.0

    total = 0

    for start in range(
        0,
        len(fixed_candidates),
        batch_size,
    ):

        query_batch = fixed_candidates[
            start:start + batch_size
        ]

        (
            member_ids,
            member_mask,
            candidate_items,
        ) = make_group_batch(
            query_batch,
            data,
            model.padding_user_id,
            device,
        )

        representation = model(
            member_ids,
            member_mask,
            candidate_items,
        )

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

        history = decision["history"]

        batch_n = member_ids.size(0)

        # --------------------------------------------
        # Test each threshold
        # --------------------------------------------

        for threshold in thresholds:

            # Default:
            # no quorum -> final state
            selected = history[-1].clone()

            stop_steps = torch.full(
                (batch_n,),
                len(history) - 1,
                dtype=torch.long,
                device=device,
            )

            reached = torch.zeros(
                batch_n,
                dtype=torch.bool,
                device=device,
            )

            # Skip step 0 because it is only
            # the initial commitment.
            for step in range(
                1,
                len(history),
            ):

                state = history[step]

                max_commitment = (
                    state.max(dim=1).values
                )

                newly_reached = (
                    (~reached)
                    &
                    (
                        max_commitment
                        >= threshold
                    )
                )

                if newly_reached.any():

                    selected[
                        newly_reached
                    ] = state[
                        newly_reached
                    ]

                    stop_steps[
                        newly_reached
                    ] = step

                    reached[
                        newly_reached
                    ] = True

            # ----------------------------------------
            # Ranking
            # ----------------------------------------

            positive_score = (
                selected[:, 0:1]
            )

            ranks = (
                1
                +
                (
                    selected[:, 1:]
                    >
                    positive_score
                )
                .sum(dim=1)
            )

            for k in ks:

                hits = (
                    ranks <= k
                )

                results[threshold][
                    f"HR@{k}"
                ] += (
                    hits.float()
                    .sum()
                    .item()
                )

                ndcg = torch.where(
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

                results[threshold][
                    f"NDCG@{k}"
                ] += (
                    ndcg.sum()
                    .item()
                )

            results[threshold][
                "mean_stop_step"
            ] += (
                stop_steps.float()
                .sum()
                .item()
            )

            results[threshold][
                "quorum_rate"
            ] += (
                reached.float()
                .sum()
                .item()
            )

        total += batch_n

    # ========================================================
    # Average
    # ========================================================

    for threshold in thresholds:

        for k in ks:

            results[threshold][
                f"HR@{k}"
            ] /= total

            results[threshold][
                f"NDCG@{k}"
            ] /= total

        results[threshold][
            "mean_stop_step"
        ] /= total

        results[threshold][
            "quorum_rate"
        ] /= total

    return results

@torch.no_grad()
def evaluate_quality_commitment_fusion(
    model,
    decision_layer,
    data,
    fixed_candidates,
    device,
    lambdas=(
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ),
    batch_size=256,
    ks=(5, 10, 20),
):

    model.eval()
    decision_layer.eval()

    results = {}

    for lam in lambdas:

        results[lam] = {}

        for k in ks:
            results[lam][f"HR@{k}"] = 0.0
            results[lam][f"NDCG@{k}"] = 0.0

    total = 0

    eps = 1e-8

    for start in range(
        0,
        len(fixed_candidates),
        batch_size,
    ):

        query_batch = fixed_candidates[
            start:start + batch_size
        ]

        (
            member_ids,
            member_mask,
            candidate_items,
        ) = make_group_batch(
            query_batch,
            data,
            model.padding_user_id,
            device,
        )

        output = model(
            member_ids,
            member_mask,
            candidate_items,
        )

        decision = decision_layer(
            member_support=
                output["member_support"],

            member_mask=
                member_mask,

            collective_support=
                output["collective_support"],

            group_quality=
                output["group_quality"],
        )

        quality = output[
            "group_quality"
        ]

        commitment = decision[
            "normalized_commitment"
        ]

        # ----------------------------------------------------
        # Query-wise standardization
        # ----------------------------------------------------

        quality_z = (
            quality
            -
            quality.mean(
                dim=1,
                keepdim=True,
            )
        ) / (
            quality.std(
                dim=1,
                keepdim=True,
            )
            + eps
        )

        commitment_z = (
            commitment
            -
            commitment.mean(
                dim=1,
                keepdim=True,
            )
        ) / (
            commitment.std(
                dim=1,
                keepdim=True,
            )
            + eps
        )

        # ----------------------------------------------------
        # Lambda sweep
        # ----------------------------------------------------

        for lam in lambdas:

            score = (
                lam * quality_z
                +
                (1.0 - lam)
                * commitment_z
            )

            positive_score = (
                score[:, 0:1]
            )

            ranks = (
                1
                +
                (
                    score[:, 1:]
                    >
                    positive_score
                )
                .sum(dim=1)
            )

            for k in ks:

                hits = (
                    ranks <= k
                )

                results[lam][
                    f"HR@{k}"
                ] += (
                    hits.float()
                    .sum()
                    .item()
                )

                ndcg = torch.where(
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

                results[lam][
                    f"NDCG@{k}"
                ] += (
                    ndcg.sum().item()
                )

        total += len(
            query_batch
        )

    for lam in lambdas:

        for metric in results[lam]:

            results[lam][metric] /= total

    return results

@torch.no_grad()
def compare_quality_vs_commitment(
    model,
    decision_layer,
    data,
    fixed_candidates,
    device,
    batch_size=256,
):

    model.eval()
    decision_layer.eval()

    total = 0

    commitment_better = 0
    quality_better = 0
    tied = 0

    quality_rank_sum = 0.0
    commitment_rank_sum = 0.0

    top10_rescue = 0
    top10_damage = 0

    rank_improvements = []

    for start in range(
        0,
        len(fixed_candidates),
        batch_size,
    ):

        query_batch = fixed_candidates[
            start:start + batch_size
        ]

        (
            member_ids,
            member_mask,
            candidate_items,
        ) = make_group_batch(
            query_batch,
            data,
            model.padding_user_id,
            device,
        )

        output = model(
            member_ids,
            member_mask,
            candidate_items,
        )

        decision = decision_layer(
            member_support=
                output["member_support"],

            member_mask=
                member_mask,

            collective_support=
                output["collective_support"],

            group_quality=
                output["group_quality"],
        )

        quality = output[
            "group_quality"
        ]

        commitment = decision[
            "normalized_commitment"
        ]

        # ----------------------------------------------------
        # Positive candidate is column 0
        # ----------------------------------------------------

        quality_positive = (
            quality[:, 0:1]
        )

        commitment_positive = (
            commitment[:, 0:1]
        )

        quality_rank = (
            1
            +
            (
                quality[:, 1:]
                >
                quality_positive
            )
            .sum(dim=1)
        )

        commitment_rank = (
            1
            +
            (
                commitment[:, 1:]
                >
                commitment_positive
            )
            .sum(dim=1)
        )

        # Positive difference means commitment improved rank
        rank_delta = (
            quality_rank
            -
            commitment_rank
        )

        commitment_better += (
            rank_delta > 0
        ).sum().item()

        quality_better += (
            rank_delta < 0
        ).sum().item()

        tied += (
            rank_delta == 0
        ).sum().item()

        quality_rank_sum += (
            quality_rank.float()
            .sum()
            .item()
        )

        commitment_rank_sum += (
            commitment_rank.float()
            .sum()
            .item()
        )

        # ----------------------------------------------------
        # Did commitment rescue a query into Top-10?
        # ----------------------------------------------------

        top10_rescue += (
            (quality_rank > 10)
            &
            (commitment_rank <= 10)
        ).sum().item()

        # ----------------------------------------------------
        # Did commitment push a Quality Top-10 item out?
        # ----------------------------------------------------

        top10_damage += (
            (quality_rank <= 10)
            &
            (commitment_rank > 10)
        ).sum().item()

        rank_improvements.append(
            rank_delta.cpu()
        )

        total += (
            quality.size(0)
        )

    rank_improvements = torch.cat(
        rank_improvements
    ).float()

    return {
        "total":
            total,

        "commitment_better":
            commitment_better,

        "quality_better":
            quality_better,

        "tied":
            tied,

        "commitment_better_pct":
            100.0
            * commitment_better
            / total,

        "quality_better_pct":
            100.0
            * quality_better
            / total,

        "tied_pct":
            100.0
            * tied
            / total,

        "mean_quality_rank":
            quality_rank_sum
            / total,

        "mean_commitment_rank":
            commitment_rank_sum
            / total,

        "mean_rank_change":
            rank_improvements
            .mean()
            .item(),

        "median_rank_change":
            rank_improvements
            .median()
            .item(),

        "top10_rescue":
            top10_rescue,

        "top10_damage":
            top10_damage,

        "top10_rescue_pct":
            100.0
            * top10_rescue
            / total,

        "top10_damage_pct":
            100.0
            * top10_damage
            / total,
    }

@torch.no_grad()
def evaluate_oracle_quality_commitment(
    model,
    decision_layer,
    data,
    fixed_candidates,
    device,
    batch_size=256,
    ks=(5, 10, 20),
):
    """
    Oracle diagnostic.

    For every validation query, choose whichever of
    quality or commitment gives the positive item the
    better rank.

    This is NOT a usable recommender.

    It measures the maximum possible benefit of a
    perfect quality-vs-commitment selector.
    """

    model.eval()
    decision_layer.eval()

    results = {
        "quality": {},
        "commitment": {},
        "oracle": {},
    }

    for name in results:
        for k in ks:
            results[name][f"HR@{k}"] = 0.0
            results[name][f"NDCG@{k}"] = 0.0

    total = 0

    oracle_uses_quality = 0
    oracle_uses_commitment = 0
    tied = 0

    for start in range(
        0,
        len(fixed_candidates),
        batch_size,
    ):

        query_batch = fixed_candidates[
            start:start + batch_size
        ]

        (
            member_ids,
            member_mask,
            candidate_items,
        ) = make_group_batch(
            query_batch,
            data,
            model.padding_user_id,
            device,
        )

        output = model(
            member_ids,
            member_mask,
            candidate_items,
        )

        decision = decision_layer(
            member_support=
                output["member_support"],

            member_mask=
                member_mask,

            collective_support=
                output["collective_support"],

            group_quality=
                output["group_quality"],
        )

        quality = output["group_quality"]

        commitment = decision[
            "normalized_commitment"
        ]

        # ----------------------------------------------------
        # Positive candidate is column 0
        # ----------------------------------------------------

        quality_rank = (
            1
            +
            (
                quality[:, 1:]
                >
                quality[:, 0:1]
            ).sum(dim=1)
        )

        commitment_rank = (
            1
            +
            (
                commitment[:, 1:]
                >
                commitment[:, 0:1]
            ).sum(dim=1)
        )

        # Oracle takes the better rank
        oracle_rank = torch.minimum(
            quality_rank,
            commitment_rank,
        )

        oracle_uses_quality += (
            quality_rank
            <
            commitment_rank
        ).sum().item()

        oracle_uses_commitment += (
            commitment_rank
            <
            quality_rank
        ).sum().item()

        tied += (
            quality_rank
            ==
            commitment_rank
        ).sum().item()

        rank_sets = {
            "quality": quality_rank,
            "commitment": commitment_rank,
            "oracle": oracle_rank,
        }

        for name, ranks in rank_sets.items():

            for k in ks:

                hits = (
                    ranks <= k
                )

                results[name][
                    f"HR@{k}"
                ] += (
                    hits.float()
                    .sum()
                    .item()
                )

                ndcg = torch.where(
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

                results[name][
                    f"NDCG@{k}"
                ] += (
                    ndcg.sum().item()
                )

        total += quality.size(0)

    for name in results:
        for metric in results[name]:
            results[name][metric] /= total

    selection = {
        "quality_better":
            oracle_uses_quality,

        "commitment_better":
            oracle_uses_commitment,

        "tied":
            tied,

        "quality_better_pct":
            100.0
            * oracle_uses_quality
            / total,

        "commitment_better_pct":
            100.0
            * oracle_uses_commitment
            / total,

        "tied_pct":
            100.0
            * tied
            / total,
    }

    return results, selection

@torch.no_grad()
def inspect_inhibition_strength(
    model,
    decision_layer,
    data,
    fixed_candidates,
    device,
    batch_size=256,
):

    model.eval()
    decision_layer.eval()

    inhibition_values = []
    abandonment_values = []
    ratios = []

    (
        discovery_strength,
        recruitment_strength,
        abandonment_strength,
        inhibition_strength,
    ) = decision_layer.get_rates()

    for start in range(
        0,
        len(fixed_candidates),
        batch_size,
    ):

        query_batch = fixed_candidates[
            start:start + batch_size
        ]

        (
            member_ids,
            member_mask,
            candidate_items,
        ) = make_group_batch(
            query_batch,
            data,
            model.padding_user_id,
            device,
        )

        output = model(
            member_ids,
            member_mask,
            candidate_items,
        )

        member_support = output[
            "member_support"
        ]

        group_quality = output[
            "group_quality"
        ]

        collective_support = output[
            "collective_support"
        ]

        inhibition_matrix = (
            decision_layer.compute_cross_inhibition(
                member_support,
                member_mask,
            )
        )

        eps = 1e-8

        support_distribution = (
            collective_support
            /
            (
                collective_support.sum(
                    dim=1,
                    keepdim=True,
                )
                + eps
            )
        )

        commitment = (
            decision_layer.initial_commitment
            *
            support_distribution
        )

        # Examine the penalties during every dynamic step
        for _ in range(
            decision_layer.num_steps
        ):

            inhibition_pressure = torch.einsum(
                "bji,bj->bi",
                inhibition_matrix,
                commitment,
            )

            cross_inhibition = (
                inhibition_strength
                *
                commitment
                *
                inhibition_pressure
            )

            abandonment = (
                abandonment_strength
                *
                (1.0 - group_quality)
                *
                commitment
            )

            valid = commitment > 0

            inhibition_values.append(
                cross_inhibition[
                    valid
                ].cpu()
            )

            abandonment_values.append(
                abandonment[
                    valid
                ].cpu()
            )

            ratio = (
                cross_inhibition
                /
                (
                    abandonment
                    + eps
                )
            )

            ratios.append(
                ratio[
                    valid
                ].cpu()
            )

            # Continue using same dynamics
            committed_mass = (
                commitment.sum(
                    dim=1,
                    keepdim=True,
                )
            )

            uncommitted = (
                1.0
                -
                committed_mass
            ).clamp(min=0.0)

            discovery = (
                discovery_strength
                *
                uncommitted
                *
                collective_support
            )

            recruitment = (
                recruitment_strength
                *
                uncommitted
                *
                group_quality
                *
                commitment
            )

            delta = (
                discovery
                +
                recruitment
                -
                abandonment
                -
                cross_inhibition
            )

            commitment = (
                commitment
                +
                decision_layer.dt
                *
                delta
            ).clamp(min=0.0)

            total_commitment = (
                commitment.sum(
                    dim=1,
                    keepdim=True,
                )
            )

            scale = torch.where(
                total_commitment > 1.0 - 1e-6,

                (1.0 - 1e-6)
                /
                (
                    total_commitment
                    + eps
                ),

                torch.ones_like(
                    total_commitment
                ),
            )

            commitment = (
                commitment
                *
                scale
            )

    inhibition_values = torch.cat(
        inhibition_values
    )

    abandonment_values = torch.cat(
        abandonment_values
    )

    ratios = torch.cat(
        ratios
    )

    print("CROSS-INHIBITION")
    print(
        "Mean:",
        f"{inhibition_values.mean():.6f}"
    )
    print(
        "Median:",
        f"{torch.quantile(inhibition_values, 0.50):.6f}"
    )
    print(
        "P90:",
        f"{torch.quantile(inhibition_values, 0.90):.6f}"
    )

    print()

    print("ABANDONMENT")
    print(
        "Mean:",
        f"{abandonment_values.mean():.6f}"
    )
    print(
        "Median:",
        f"{torch.quantile(abandonment_values, 0.50):.6f}"
    )
    print(
        "P90:",
        f"{torch.quantile(abandonment_values, 0.90):.6f}"
    )

    print()

    print("INHIBITION / ABANDONMENT RATIO")
    print(
        "Mean:",
        f"{ratios.mean():.4f}"
    )
    print(
        "Median:",
        f"{torch.quantile(ratios, 0.50):.4f}"
    )
    print(
        "P90:",
        f"{torch.quantile(ratios, 0.90):.4f}"
    )