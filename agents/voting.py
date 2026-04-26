"""
Voting Engine for the Head Agent Council.

Aggregates votes from RiskAgent, FraudAgent, PolicyAgent.
Detects dissent and logs it to logs/dissent_log.json.
Returns a VoteResult which is a recommendation to the final
weighted decision engine, not the final decision.
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from agents.council.risk_agent import HeadAgentVote
from agents.sub_agents.risk.credit_analyst import SubAgentReport
from env.action import LoanAction


class DissentLog(BaseModel):
    """Record of a dissenting vote in the council."""

    episode: int
    dissenting_agent: str
    minority_decision: str
    majority_decision: str
    dissent_reason: str
    timestamp: str


class VoteResult(BaseModel):
    """
    Result of the council voting process.
    This is a recommendation to the final weighted engine, not the
    final decision.
    """

    council_recommendation: LoanAction
    vote_breakdown: Dict[str, str]
    agent_reasoning: Dict[str, str] = Field(default_factory=dict)
    agent_confidence: Dict[str, float] = Field(default_factory=dict)
    was_unanimous: bool
    dissent_log: Optional[DissentLog] = None
    confidence_average: float
    all_sub_reports: List[SubAgentReport] = Field(default_factory=list)


class VotingEngine:
    """
    Aggregates head agent votes using equal-weight majority.
    Detects and logs dissent. Never overwrites existing logs.
    """

    DISSENT_LOG_PATH = "logs/dissent_log.json"

    def _sanitize_reasoning(self, text: str, max_length: int = 250) -> str:
        """
        Sanitize LLM reasoning to prevent indirect prompt injection.
        Caps length and strips out instruction-like keywords.
        """
        if not text:
            return ""
        
        # 1. Cap length
        sanitized = text[:max_length]
        
        # 2. Strip out common injection keywords (case-insensitive)
        injection_patterns = [
            r"ignore previous instructions",
            r"system:",
            r"user:",
            r"assistant:",
            r"respond only",
            r"instruction:",
            r"<\/?applicant_data>",
            r"<\/?sub_agent_reports>",
            r"<\/?dissent_history>",
            r"<\/?identity_guidance>",
        ]
        for pattern in injection_patterns:
            sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)
            
        return sanitized.strip()

    def run_vote(
        self,
        votes: List[HeadAgentVote],
        episode: int,
    ) -> VoteResult:
        """
        Count votes with equal weight (1.0 each).
        Majority determines council_recommendation.
        3-way tie: PolicyAgent breaks tie.
        """
        # Build vote breakdown and reasoning
        vote_breakdown: Dict[str, str] = {}
        agent_reasoning: Dict[str, str] = {}
        agent_confidence: Dict[str, float] = {}
        for v in votes:
            vote_breakdown[v.agent_name] = v.decision.value
            agent_reasoning[v.agent_name] = v.reasoning
            agent_confidence[v.agent_name] = v.confidence

        # Count votes
        tally: Dict[str, int] = {}
        for v in votes:
            action = v.decision.value
            tally[action] = tally.get(action, 0) + 1

        # Determine majority
        max_count = max(tally.values())
        winners = [a for a, c in tally.items() if c == max_count]

        if len(winners) == 1:
            majority_action = winners[0]
        else:
            # Tie: PolicyAgent breaks it
            policy_vote = None
            for v in votes:
                if v.agent_name == "PolicyAgent":
                    policy_vote = v.decision.value
                    break
            majority_action = policy_vote if policy_vote else winners[0]

        council_recommendation = LoanAction(majority_action)

        # Check unanimity
        all_same = all(v.decision.value == majority_action for v in votes)
        was_unanimous = all_same

        # Calculate confidence average
        conf_sum = sum(v.confidence for v in votes)
        confidence_average = conf_sum / max(len(votes), 1)

        # Detect dissent
        dissent_log_entry: Optional[DissentLog] = None
        if not was_unanimous:
            dissent_entries: List[DissentLog] = []
            for v in votes:
                if v.decision.value != majority_action:
                    dissent_entries.append(
                        DissentLog(
                            episode=episode,
                            dissenting_agent=v.agent_name,
                            minority_decision=v.decision.value,
                            majority_decision=majority_action,
                            dissent_reason=self._sanitize_reasoning(v.reasoning),
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        )
                    )

            if dissent_entries:
                # Keep UI behavior stable by exposing first dissenter in VoteResult.
                dissent_log_entry = dissent_entries[0]
                for entry in dissent_entries:
                    self._append_dissent(entry)

        return VoteResult(
            council_recommendation=council_recommendation,
            vote_breakdown=vote_breakdown,
            agent_reasoning=agent_reasoning,
            agent_confidence=agent_confidence,
            was_unanimous=was_unanimous,
            dissent_log=dissent_log_entry,
            confidence_average=confidence_average,
            all_sub_reports=[],
        )

    def _append_dissent(self, entry: DissentLog) -> None:
        """
        Append dissent entry to logs/dissent_log.json.
        Never overwrites — always appends.
        Creates file with [] if it doesn't exist.
        """
        os.makedirs(os.path.dirname(self.DISSENT_LOG_PATH), exist_ok=True)

        if not os.path.exists(self.DISSENT_LOG_PATH):
            with open(self.DISSENT_LOG_PATH, "w") as f:
                json.dump([], f)

        with open(self.DISSENT_LOG_PATH, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []

        logs.append(entry.model_dump())

        with open(self.DISSENT_LOG_PATH, "w") as f:
            json.dump(logs, f, indent=2)
