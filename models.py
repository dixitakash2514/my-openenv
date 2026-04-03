# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Supply Chain Retail Environment.

Three tasks: shelf_restock (easy), delivery_routing (medium), demand_surge (hard).
"""

from typing import Any, Dict, List

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SupplyChainAction(Action):
    """Action for the Supply Chain environment — the agent's decision."""

    decision: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task-specific decision as a JSON-like dict",
    )
    reasoning: str = Field(
        default="",
        description="Agent's explanation for its decision",
    )


class SupplyChainObservation(Observation):
    """Observation from the Supply Chain environment."""

    task_name: str = Field(default="", description="Current task identifier")
    scenario_text: str = Field(
        default="", description="Human-readable scenario for the LLM agent"
    )
    scenario_data: Dict[str, Any] = Field(
        default_factory=dict, description="Structured scenario data"
    )
    score_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Per-criterion scores after grading"
    )
    feedback: str = Field(
        default="", description="Grader feedback text after step"
    )
