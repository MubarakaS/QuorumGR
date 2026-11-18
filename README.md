# QuorumGR: Bio-Inspired Collective Decision-Making for Group Recommendation

## Overview


QuorumGR is a group recommendation framework integrating individual preference learning, group-quality prediction, and bio-inspired collective decision-making.

Candidate commitment evolves through four mechanisms:

- **Discovery:** Allocates uncommitted decision mass according to collective member support.
- **Recruitment:** Reinforces commitment based on predicted group quality.
- **Abandonment:** Reduces commitment to lower-quality candidates.
- **Cross-inhibition:** Models competition among alternatives using conflicting member preferences.

A disagreement-adaptive quorum mechanism then produces candidate recommendation scores.



## Installation

Python 3.10 or later is recommended. Clone the repository and install dependencies:

```bash
git clone https://github.com/YOUR-USERNAME/QuorumGR.git
cd QuorumGR
pip install -r requirements.txt
```

Replace `YOUR-USERNAME` with the GitHub account that hosts the repository.

## Datasets

The datasets are not redistributed here. Research repositories containing dataset files include:

- [CAMRa2011 — Attentive Group Recommendation](https://github.com/LianHaiMiao/Attentive-Group-Recommendation/tree/master/data/CAMRa2011)
- [Mafengwo — ConsRec](https://github.com/FDUDSDE/WWW2023ConsRec/tree/master/data/Mafengwo)

For the included notebook, put the seven CAMRa2011 files in this folder:

```text
data/CAMRa2011/
├── groupMember.txt
├── groupRatingTrain.txt
├── groupRatingTest.txt
├── groupRatingNegative.txt
├── userRatingTrain.txt
├── userRatingTest.txt
└── userRatingNegative.txt
```

See `data/README.md` for the expected files. The supplied notebook runs **CAMRa2011 only**; Mafengwo preprocessing and execution are not included in this package.

## Run the notebook

From the repository root:

```bash
jupyter notebook notebooks/Run_QuorumGR_CAMRa_With_Final_Evaluation.ipynb
```



### Running QuorumGR

The notebook contains the complete CAMRa2011 workflow:

1. Load the CAMRa2011 dataset and initialize the model.
2. Train individual preferences (Stage A).
3. Train group-quality predictions (Stage B).
4. Train the collective decision layer.
5. Evaluate recommendations using fixed validation candidates.
6. Save trained checkpoints and experimental results.

For evaluation without retraining, initialize the model, recreate the validation candidates, and use Section 19 to load compatible checkpoints.

Set `LOAD_SAVED_WEIGHTS = True` in Section 19.1 when the trained checkpoint files are available.

The repository does not include pretrained checkpoints. Users running the project for the first time must train the model before loading saved weights.

Avoid running all notebook cells when reusing existing checkpoints, as earlier training and saving cells may overwrite them.



## Scoring and evaluation

The implementation exposes distinct scoring signals:

- `collective_support`: Mean member support for an item.
- `group_quality`: Predicted group-item quality.
- `commitment`: Commitment after the final collective decision iteration.
- `trajectory_commitment`: Average commitment throughout the simulated trajectory.
- `quorum_score`: Final commitment adjusted by the adaptive quorum gate.

The CAMRa notebook evaluates fixed validation candidates and official CAMRa test candidates, including tie-aware evaluation. **The included official-test functions do not yet directly evaluate `trajectory_commitment` or `quorum_score`.** Use an explicit evaluation extension for those scores if they are reported in the paper. Do not tune hyperparameters on the official test set.

## Reproducibility

This repository provides the core QuorumGR code and a CAMRa2011 execution workflow. To reproduce particular manuscript results, match the experimental configuration, random seeds, checkpoints, candidate construction, and selected ranking score. The provided notebook defaults are a starting configuration and have not been verified against every reported paper table.

## Citation

If you use this code, please cite the accompanying QuorumGR paper. Add its full bibliographic information here when available.
