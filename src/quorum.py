import torch
import torch.nn as nn
import torch.nn.functional as F


class QuorumDecisionLayer(nn.Module):
    """
    Bio-inspired collective decision layer for QuorumGR.
    Implements the collective decision-making mechanism for QuorumGR.
    Simulates how candidate commitment evolves through discovery,
    recruitment, abandonment, and cross-inhibition.
    
    Applies an adaptive quorum mechanism that accounts for member
    disagreement when computing final recommendation scores.

    Components
    ----------
    1. Member-evidence discovery
    2. Quality-sensitive recruitment
    3. Quality-sensitive abandonment
    4. Preference-conditioned cross-inhibition
    5. Disagreement-adaptive quorum

    """

    def __init__(
        self,
        num_steps=5,
        dt=0.20,
        initial_commitment=0.10,

        discovery_strength=0.20,
        recruitment_strength=1.00,
        abandonment_strength=0.20,
        inhibition_strength=0.50,

        quorum_scale=2.0,
        quorum_temperature=0.25,
    ):

        """
        Initializes the collective decision layer, including the
        simulation settings, learnable decision rates, and quorum
        parameters.
        """
        super().__init__()

        self.num_steps = num_steps
        self.dt = dt
        self.initial_commitment = initial_commitment

        # Trainable biological rates

        self.raw_discovery = nn.Parameter(
            torch.tensor(
                self._inverse_softplus(
                    discovery_strength
                ),
                dtype=torch.float32,
            )
        )

        self.raw_recruitment = nn.Parameter(
            torch.tensor(
                self._inverse_softplus(
                    recruitment_strength
                ),
                dtype=torch.float32,
            )
        )

        self.raw_abandonment = nn.Parameter(
            torch.tensor(
                self._inverse_softplus(
                    abandonment_strength
                ),
                dtype=torch.float32,
            )
        )

        self.raw_inhibition = nn.Parameter(
            torch.tensor(
                self._inverse_softplus(
                    inhibition_strength
                ),
                dtype=torch.float32,
            )
        )

        # Quorum parameters
       

        self.quorum_scale = quorum_scale
        self.quorum_temperature = quorum_temperature

    
    # Utilities

    @staticmethod
    def _inverse_softplus(value):
        """
        Converts a desired positive rate into its raw parameter value
        so that applying softplus recovers the intended initial rate.
        """
        # Use float64 and expm1 to initialize very small rates safely.
        value = torch.as_tensor(value, dtype=torch.float64)
        if torch.any(value <= 0):
            raise ValueError("Decision rate must be positive")
        return (value + torch.log(-torch.expm1(-value))).item()

    # Positive biological rates

    def get_rates(self):
        """
        Applies softplus to obtain positive discovery, recruitment,
        abandonment, and inhibition rates used in the decision process.
        """
        discovery = F.softplus(
            self.raw_discovery
        )

        recruitment = F.softplus(
            self.raw_recruitment
        )

        abandonment = F.softplus(
            self.raw_abandonment
        )

        inhibition = F.softplus(
            self.raw_inhibition
        )

        return (
            discovery,
            recruitment,
            abandonment,
            inhibition,
        )

    # Member disagreement

    def compute_member_disagreement(
        self,
        member_support,
        member_mask,
        collective_support,
    ):
        """
        Measures how much group members disagree about each candidate
        using the variance of their predicted support.

        Returns the original and normalized disagreement scores.
        """
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

        mean_support = (
            collective_support
            .unsqueeze(1)
        )

        squared_difference = (
            member_support-mean_support).pow(2)

        squared_difference = (squared_difference*mask)

        disagreement = (squared_difference.sum(dim=1)/group_size)

        normalized_disagreement = (4.0*disagreement).clamp(
            min=0.0,
            max=1.0,
        )

        return (
            disagreement,
            normalized_disagreement,
        )

    
    # Preference-conditioned cross-inhibition

    def compute_cross_inhibition(
        self,
        member_support,
        member_mask,
    ):
        """
        Computes competition between candidate items based on
        members' support for one candidate and opposition to another.

        Returns a pairwise inhibition matrix, excluding self-inhibition.
        """

        mask = (
            member_mask
            .unsqueeze(-1)
        )

        support = (
            member_support
            *
            mask
        )

        opposition = (
            (1.0 - member_support)
            *
            mask
        )

        group_size = (
            member_mask
            .sum(
                dim=1,
                keepdim=True,
            )
            .clamp(min=1.0)
        )

        inhibition_matrix = torch.einsum(
            "bgj,bgi->bji",
            support,
            opposition,
        )

        inhibition_matrix = (
            inhibition_matrix
            /
            group_size.unsqueeze(-1)
        )

        # No self-inhibition.
        num_candidates = (
            inhibition_matrix.size(1)
        )

        eye = torch.eye(
            num_candidates,
            device=inhibition_matrix.device,
            dtype=inhibition_matrix.dtype,
        ).unsqueeze(0)

        inhibition_matrix = (
            inhibition_matrix
            *
            (1.0 - eye)
        )

        return inhibition_matrix

    # Adaptive quorum

    def apply_adaptive_quorum(
        self,
        commitment,
        normalized_disagreement,
    ):
        """
        Applies a disagreement-dependent quorum threshold to
        normalized candidate commitment.

        Candidates with greater member disagreement require stronger
        relative commitment to achieve a high quorum score.
        """
        eps = 1e-8

        num_candidates = (
            commitment.size(1)
        )

        # Normalize commitment

        normalized_commitment = (
            commitment
            /
            (
                commitment.sum(
                    dim=1,
                    keepdim=True,
                )
                +
                eps
            )
        )

        # Commitment relative to uniform candidate share
        # Uniform baseline becomes 1.

        relative_commitment = (
            normalized_commitment
            *
            float(num_candidates)
        )

        # Adaptive quorum threshold

        quorum_threshold = (
            1.0
            +
            self.quorum_scale
            *
            normalized_disagreement
        )

        # Soft quorum attainment

        quorum_gate = torch.sigmoid(
            (
                relative_commitment
                -
                quorum_threshold
            )
            /
            self.quorum_temperature
        )

        # Final adaptive quorum score

        quorum_score = (
            normalized_commitment
            *
            quorum_gate
        )

        return (
            normalized_commitment,
            relative_commitment,
            quorum_threshold,
            quorum_gate,
            quorum_score,
        )

    # Forward collective-decision dynamics

    def forward(
        self,
        member_support,
        member_mask,
        collective_support,
        group_quality,
    ):
        """
        Simulates the collective decision-making process over multiple
        iterations using discovery, recruitment, abandonment, and
        cross-inhibition.

        Updates candidate commitment, records the decision trajectory,
        and applies the adaptive quorum mechanism to produce final
        recommendation scores and decision diagnostics.
        """


        (
            discovery_strength,
            recruitment_strength,
            abandonment_strength,
            inhibition_strength,
        ) = self.get_rates()

        eps = 1e-8

        # Member disagreement

        (
            disagreement,
            normalized_disagreement,
        ) = self.compute_member_disagreement(
            member_support,
            member_mask,
            collective_support,
        )

        # Cross-inhibition

        inhibition_matrix = (
            self.compute_cross_inhibition(
                member_support,
                member_mask,
            )
        )

        # Initial commitment
        #
        # Member support determines discovery distribution.

        support_distribution = (
            collective_support
            /
            (
                collective_support.sum(
                    dim=1,
                    keepdim=True,
                )
                +
                eps
            )
        )

        commitment = (
            self.initial_commitment
            *
            support_distribution
        )

        history = [
            commitment
        ]

        # Collective decision iterations

        for _ in range(
            self.num_steps
        ):

            committed_mass = (
                commitment
                .sum(
                    dim=1,
                    keepdim=True,
                )
            )

            uncommitted = (
                1.0
                -
                committed_mass
            ).clamp(
                min=0.0,
            )

            # Discovery

            discovery = (
                discovery_strength
                *
                uncommitted
                *
                collective_support
            )

            # Quality-sensitive recruitment

            recruitment = (
                recruitment_strength
                *
                uncommitted
                *
                group_quality
                *
                commitment
            )

            # Quality-sensitive abandonment

            abandonment = (
                abandonment_strength
                *
                (1.0 - group_quality)
                *
                commitment
            )

            # Preference-conditioned cross-inhibition

            inhibition_pressure = (
                torch.einsum(
                    "bji,bj->bi",
                    inhibition_matrix,
                    commitment,
                )
            )

            cross_inhibition = (
                inhibition_strength
                *
                commitment
                *
                inhibition_pressure
            )

            # State update

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
                self.dt
                *
                delta
            )

            commitment = torch.clamp(
                commitment,
                min=0.0,
            )

            # Keep committed mass below one

            total_commitment = (
                commitment.sum(
                    dim=1,
                    keepdim=True,
                )
            )

            max_mass = (
                1.0 - 1e-6
            )

            scale = torch.where(
                total_commitment
                >
                max_mass,

                max_mass
                /
                (
                    total_commitment
                    +
                    eps
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

            history.append(
                commitment
            )
        
        # Preserve collective evidence accumulated throughout
        # the complete decision process

        history_tensor = torch.stack(
            history,
            dim=0,
        )

        trajectory_commitment = (
            history_tensor.mean(
                dim=0
            )
        )

        # Adaptive quorum

        (
            normalized_commitment,
            relative_commitment,
            quorum_threshold,
            quorum_gate,
            quorum_score,
        ) = self.apply_adaptive_quorum(
            commitment,
            normalized_disagreement,
        )

        # Return
        

        return {
            # Final QuorumGR ranking score
            "final_score":
                quorum_score,

            # Before quorum
            "commitment":
                commitment,

            "normalized_commitment":
                normalized_commitment,
                
            "trajectory_commitment": trajectory_commitment,
            # Quorum quantities
            "relative_commitment":
                relative_commitment,

            "disagreement":
                disagreement,

            "normalized_disagreement":
                normalized_disagreement,

            "quorum_threshold":
                quorum_threshold,

            "quorum_gate":
                quorum_gate,

            "quorum_score":
                quorum_score,

            # Cross-inhibition
            "inhibition_matrix":
                inhibition_matrix,

            # Dynamics history
            "history":
                history,

            # Learned biological rates
            "discovery_strength":
                discovery_strength,

            "recruitment_strength":
                recruitment_strength,

            "abandonment_strength":
                abandonment_strength,

            "inhibition_strength":
                inhibition_strength,
        }