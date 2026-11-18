import math

import torch


def build_official_group_test_candidates(
    data,
):
    """
    Construct the official CAMRa group test candidate sets.

    For every row in groupRatingTest:

        positive item
        +
        negatives from groupRatingNegative

    Duplicate test (group,item) pairs reuse the same
    official negative set.
    """

    test_candidates = []

    missing = 0

    for idx, (
        group_id,
        positive_item,
    ) in enumerate(
        zip(
            data.group_test.entity_ids,
            data.group_test.item_ids,
        )
    ):

        key = (
            group_id,
            positive_item,
        )

        negatives = (
            data.group_negatives.get(
                key
            )
        )

        if negatives is None:
            missing += 1
            continue

        candidates = (
            [positive_item]
            +
            list(negatives)
        )

        test_candidates.append(
            (
                idx,
                group_id,
                positive_item,
                candidates,
            )
        )

    print(
        "Official test rows:",
        len(data.group_test.ratings)
    )

    print(
        "Evaluable rows:",
        len(test_candidates)
    )

    print(
        "Missing negative sets:",
        missing
    )

    lengths = [
        len(row[3])
        for row in test_candidates
    ]

    print(
        "Candidate-set sizes:",
        sorted(set(lengths))
    )

    return test_candidates


def make_test_group_batch(
    query_batch,
    data,
    padding_user_id,
    device,
):

    batch_size = len(
        query_batch
    )

    max_group_size = max(
        len(
            data.group_members[group_id]
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
    ) in enumerate(
        query_batch
    ):

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


@torch.no_grad()
def evaluate_official_group_test(
    model,
    decision_layer,
    data,
    test_candidates,
    device,
    batch_size=256,
    ks=(5, 10, 20),
):

    model.eval()
    decision_layer.eval()

    names = [
        "quality",
        "support",
        "commitment",
    ]

    results = {
        name: {
            f"{metric}@{k}": 0.0
            for k in ks
            for metric in [
                "HR",
                "NDCG",
            ]
        }
        for name in names
    }

    total = 0

    # Candidate sizes should normally be equal.
    # Group by size so evaluation remains safe
    # even if CAMRa contains a different-sized set.
    by_size = {}

    for query in test_candidates:

        size = len(
            query[3]
        )

        by_size.setdefault(
            size,
            []
        ).append(query)

    for candidate_size, queries in (
        by_size.items()
    ):

        for start in range(
            0,
            len(queries),
            batch_size,
        ):

            query_batch = queries[
                start:
                start + batch_size
            ]

            (
                member_ids,
                member_mask,
                candidate_items,
            ) = make_test_group_batch(
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
            }

            for name, score_matrix in (
                scores.items()
            ):

                positive = (
                    score_matrix[:, 0:1]
                )

                ranks = (
                    1
                    +
                    (
                        score_matrix[:, 1:]
                        >
                        positive
                    ).sum(dim=1)
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

    for name in names:

        for metric in results[name]:

            results[name][metric] /= (
                total
            )

    return results

@torch.no_grad()
def evaluate_official_group_test_tie_aware(
    model,
    decision_layer,
    data,
    test_candidates,
    device,
    batch_size=256,
    ks=(5, 10, 20),
):

    model.eval()
    decision_layer.eval()

    names = [
        "quality",
        "support",
        "commitment",
    ]

    results = {
        name: {
            f"{metric}@{k}": 0.0
            for k in ks
            for metric in [
                "HR",
                "NDCG",
            ]
        }
        for name in names
    }

    total = 0

    by_size = {}

    for query in test_candidates:

        size = len(
            query[3]
        )

        by_size.setdefault(
            size,
            []
        ).append(query)

    for candidate_size, queries in (
        by_size.items()
    ):

        for start in range(
            0,
            len(queries),
            batch_size,
        ):

            query_batch = queries[
                start:
                start + batch_size
            ]

            (
                member_ids,
                member_mask,
                candidate_items,
            ) = make_test_group_batch(
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
            }

            for name, score_matrix in (
                scores.items()
            ):

                positive = (
                    score_matrix[:, 0:1]
                )

                # Number of candidates strictly
                # better than the positive.
                greater = (
                    score_matrix[:, 1:]
                    >
                    positive
                ).sum(dim=1)

                # Number of negatives exactly tied
                # with the positive.
                tied_negatives = (
                    score_matrix[:, 1:]
                    ==
                    positive
                ).sum(dim=1)

                # First possible rank of positive.
                first_rank = (
                    greater + 1
                )

                # Size of the tied block,
                # including the positive item.
                tie_size = (
                    tied_negatives + 1
                )

                # Last possible rank of positive.
                last_rank = (
                    first_rank
                    +
                    tie_size
                    - 1
                )

                for k in ks:

                    # Expected HR@K

                    positions_inside_k = (
                        torch.minimum(
                            last_rank,
                            torch.full_like(
                                last_rank,
                                k,
                            )
                        )
                        -
                        first_rank
                        +
                        1
                    ).clamp(
                        min=0
                    )

                    expected_hr = (
                        positions_inside_k.float()
                        /
                        tie_size.float()
                    )

                    results[name][
                        f"HR@{k}"
                    ] += (
                        expected_hr
                        .sum()
                        .item()
                    )

                    # ============================
                    # Expected NDCG@K
                    # ============================

                    expected_ndcg = (
                        torch.zeros(
                            score_matrix.size(0),
                            dtype=torch.float32,
                            device=device,
                        )
                    )

                    for row in range(
                        score_matrix.size(0)
                    ):

                        a = int(
                            first_rank[
                                row
                            ].item()
                        )

                        b = int(
                            last_rank[
                                row
                            ].item()
                        )

                        m = int(
                            tie_size[
                                row
                            ].item()
                        )

                        upper = min(
                            b,
                            k,
                        )

                        if a <= upper:

                            ranks = (
                                torch.arange(
                                    a,
                                    upper + 1,
                                    dtype=torch.float32,
                                    device=device,
                                )
                            )

                            discounts = (
                                1.0
                                /
                                torch.log2(
                                    ranks
                                    +
                                    1.0
                                )
                            )

                            expected_ndcg[
                                row
                            ] = (
                                discounts.sum()
                                /
                                float(m)
                            )

                    results[name][
                        f"NDCG@{k}"
                    ] += (
                        expected_ndcg
                        .sum()
                        .item()
                    )

            total += len(
                query_batch
            )

    for name in names:

        for metric in results[name]:

            results[name][
                metric
            ] /= total

    return results