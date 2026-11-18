from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple


# Data containers

@dataclass
class RatingData:
    """
    A simple container for rating information.

    Each rating is represented across three parallel lists:
        - entity_ids[i]: the user or group who gave the rating
        - item_ids[i]: the item being rated
        - ratings[i]: the numeric rating value

    All three lists must have the same length.
    """
    entity_ids: List[int]
    item_ids: List[int]
    ratings: List[float]


@dataclass
class CAMRaData:
    """
    A complete representation of the CAMRa2011 dataset.

    Contains:
        - group membership information
        - group train/test ratings
        - group negative candidate items
        - user train/test ratings
        - user negative candidate items
        - dataset dimensions (number of users, groups, items)

    This is the final structured dataset returned by load_camra().
    """
    group_members: Dict[int, List[int]]

    group_train: RatingData
    group_test: RatingData
    group_negatives: Dict[Tuple[int, int], List[int]]

    user_train: RatingData
    user_test: RatingData
    user_negatives: Dict[Tuple[int, int], List[int]]

    num_users: int
    num_groups: int
    num_items: int



# Group membership

def load_group_members(file_path):
    """
    Loads group membership information from groupMember.txt.

    Returns:
        dict: {group_id: [user_id, user_id, ...]}
    """
    group_members = {}

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid groupMember format "
                    f"at line {line_number}: {line}"
                )

            group_id = int(
                parts[0]
            )

            members = [
                int(x)
                for x in parts[1].split(",")
            ]

            group_members[
                group_id
            ] = members

    return group_members


# Rating files

def load_rating_file(file_path):
    """
    Loads rating data from a rating file (train or test).

    Returns:
        RatingData: containing lists of entity_ids, item_ids, ratings.
    """
    entity_ids = []
    item_ids = []
    ratings = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 3:
                raise ValueError(
                    f"Invalid rating format "
                    f"at line {line_number}: {line}"
                )

            entity_id = int(
                parts[0]
            )

            item_id = int(
                parts[1]
            )

            rating = float(
                parts[2]
            )

            entity_ids.append(
                entity_id
            )

            item_ids.append(
                item_id
            )

            ratings.append(
                rating
            )

    return RatingData(
        entity_ids=entity_ids,
        item_ids=item_ids,
        ratings=ratings,
    )


# Negative candidate files
#

def load_negative_file(file_path):
    """
    Loads negative candidate items for each positive rating.

    Returns:
        dict: {(entity_id, positive_item): [negative_item, ...]}
    """
    negatives = {}

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                raise ValueError(
                    f"Invalid negative file format "
                    f"at line {line_number}: {line}"
                )

            pair = parts[0]

            if not (
                pair.startswith("(")
                and pair.endswith(")")
            ):
                raise ValueError(
                    f"Invalid pair at line "
                    f"{line_number}: {pair}"
                )

            pair = pair[1:-1]

            pair_parts = pair.split(",")

            if len(pair_parts) != 2:
                raise ValueError(
                    f"Invalid pair at line "
                    f"{line_number}: {pair}"
                )

            entity_id = int(
                pair_parts[0]
            )

            positive_item = int(
                pair_parts[1]
            )

            negative_items = [
                int(x)
                for x in parts[1:]
            ]

            negatives[
                (
                    entity_id,
                    positive_item,
                )
            ] = negative_items

    return negatives


# Infer dataset dimensions


def infer_dimensions(
    group_members,
    group_train,
    group_test,
    group_negatives,
    user_train,
    user_test,
    user_negatives,
):
    """
    Computes dataset dimensions by scanning all IDs appearing anywhere.

    Returns:
        (num_users, num_groups, num_items)
    """

    user_ids = set()
    group_ids = set()
    item_ids = set()

    
    # Group membership
    

    for group_id, members in group_members.items():

        group_ids.add(
            group_id
        )

        for user_id in members:
            user_ids.add(
                user_id
            )

    # User train/test

    for user_id, item_id in zip(
        user_train.entity_ids,
        user_train.item_ids,
    ):
        user_ids.add(
            user_id
        )

        item_ids.add(
            item_id
        )

    for user_id, item_id in zip(
        user_test.entity_ids,
        user_test.item_ids,
    ):
        user_ids.add(
            user_id
        )

        item_ids.add(
            item_id
        )

    # Group train/test

    for group_id, item_id in zip(
        group_train.entity_ids,
        group_train.item_ids,
    ):
        group_ids.add(
            group_id
        )

        item_ids.add(
            item_id
        )

    for group_id, item_id in zip(
        group_test.entity_ids,
        group_test.item_ids,
    ):
        group_ids.add(
            group_id
        )

        item_ids.add(
            item_id
        )

    # User negatives

    for (
        user_id,
        positive_item,
    ), negative_items in user_negatives.items():

        user_ids.add(
            user_id
        )

        item_ids.add(
            positive_item
        )

        item_ids.update(
            negative_items
        )

    # Group negatives

    for (
        group_id,
        positive_item,
    ), negative_items in group_negatives.items():

        group_ids.add(
            group_id
        )

        item_ids.add(
            positive_item
        )

        item_ids.update(
            negative_items
        )

    # Raw IDs are preserved, so embedding size = max ID + 1.
    num_users = (
        max(user_ids)
        + 1
    )

    num_groups = (
        max(group_ids)
        + 1
    )

    num_items = (
        max(item_ids)
        + 1
    )

    return (
        num_users,
        num_groups,
        num_items,
    )


# Complete CAMRa loader

def load_camra(data_dir):

    """
    Loads the entire CAMRa2011 dataset from a directory.

    Steps:
        1. Load group membership
        2. Load group train/test ratings
        3. Load group negative candidates
        4. Load user train/test ratings
        5. Load user negative candidates
        6. Infer dataset dimensions
        7. Return a CAMRaData object

    Returns:
        CAMRaData: fully populated dataset structure.
    """


    data_dir = Path(
        data_dir
    )

    group_members = load_group_members(
        data_dir / "groupMember.txt"
    )

    group_train = load_rating_file(
        data_dir / "groupRatingTrain.txt"
    )

    group_test = load_rating_file(
        data_dir / "groupRatingTest.txt"
    )

    group_negatives = load_negative_file(
        data_dir / "groupRatingNegative.txt"
    )

    user_train = load_rating_file(
        data_dir / "userRatingTrain.txt"
    )

    user_test = load_rating_file(
        data_dir / "userRatingTest.txt"
    )

    user_negatives = load_negative_file(
        data_dir / "userRatingNegative.txt"
    )

    (
        num_users,
        num_groups,
        num_items,
    ) = infer_dimensions(
        group_members,
        group_train,
        group_test,
        group_negatives,
        user_train,
        user_test,
        user_negatives,
    )

    return CAMRaData(
        group_members=group_members,

        group_train=group_train,
        group_test=group_test,
        group_negatives=group_negatives,

        user_train=user_train,
        user_test=user_test,
        user_negatives=user_negatives,

        num_users=num_users,
        num_groups=num_groups,
        num_items=num_items,
    )


# Dataset sanity checks

def validate_camra(data):
    """
    Prints dataset statistics and checks for missing negative candidate sets.

    Ensures:
        - Every group test rating has a corresponding negative set
        - Every user test rating has a corresponding negative set

    This helps verify dataset completeness and correctness.
    """


    print("=" * 60)

    print(
        "CAMRa2011 DATASET"
    )

    print("=" * 60)

    print(
        "Number of users:",
        data.num_users,
    )

    print(
        "Number of groups:",
        data.num_groups,
    )

    print(
        "Number of items:",
        data.num_items,
    )

    print()

    print(
        "User train ratings:",
        len(
            data.user_train.ratings
        ),
    )

    print(
        "User test ratings:",
        len(
            data.user_test.ratings
        ),
    )

    print(
        "Group train ratings:",
        len(
            data.group_train.ratings
        ),
    )

    print(
        "Group test ratings:",
        len(
            data.group_test.ratings
        ),
    )

    print()

    print(
        "User negative sets:",
        len(
            data.user_negatives
        ),
    )

    print(
        "Group negative sets:",
        len(
            data.group_negatives
        ),
    )

    # Verify group-test negative sets
    #

    missing_group_negatives = 0

    for group_id, item_id in zip(
        data.group_test.entity_ids,
        data.group_test.item_ids,
    ):

        key = (
            group_id,
            item_id,
        )

        if key not in data.group_negatives:
            missing_group_negatives += 1

    print()

    print(
        "Group-test positives missing "
        "negative candidates:",
        missing_group_negatives,
    )

    # Verify user-test negative sets

    missing_user_negatives = 0

    for user_id, item_id in zip(
        data.user_test.entity_ids,
        data.user_test.item_ids,
    ):

        key = (
            user_id,
            item_id,
        )

        if key not in data.user_negatives:
            missing_user_negatives += 1

    print(
        "User-test positives missing "
        "negative candidates:",
        missing_user_negatives,
    )

    print("=" * 60)


# ============================================================
# Run directly
# ============================================================

if __name__ == "__main__":

    # Locate the CAMRa2011 dataset inside the project folder
    DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "CAMRa2011"

    data = load_camra(
        DATA_DIR
    )

    validate_camra(
        data
    )