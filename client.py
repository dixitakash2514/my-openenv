# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Supply Chain Retail Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SupplyChainAction, SupplyChainObservation


class SupplyChainEnv(
    EnvClient[SupplyChainAction, SupplyChainObservation, State]
):
    """
    Client for the Supply Chain Retail Environment.

    Example:
        >>> env = await SupplyChainEnv.from_docker_image("my_env-env:latest")
        >>> result = await env.reset(task_name="shelf_restock", seed=42)
        >>> print(result.observation.scenario_text)
        >>> result = await env.step(SupplyChainAction(decision={...}))
        >>> print(result.reward)
    """

    def _step_payload(self, action: SupplyChainAction) -> Dict:
        return {
            "decision": action.decision,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SupplyChainObservation]:
        obs_data = payload.get("observation", {})
        observation = SupplyChainObservation(
            task_name=obs_data.get("task_name", ""),
            step_number=obs_data.get("step_number", 0),
            total_steps=obs_data.get("total_steps", 1),
            scenario_text=obs_data.get("scenario_text", ""),
            scenario_data=obs_data.get("scenario_data", {}),
            score_breakdown=obs_data.get("score_breakdown", {}),
            feedback=obs_data.get("feedback", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
