from __future__ import annotations

import copy
import functools
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass
class EpisodeResult:
    metadata: dict[str, Any]
    activation: np.ndarray | None


@functools.lru_cache(maxsize=1)
def _load_official_example(vima_root: str | Path):
    example_path = Path(vima_root) / "scripts" / "example.py"
    if not example_path.exists():
        raise FileNotFoundError(f"official VIMA example not found: {example_path}")
    if not hasattr(np, "bool"):
        np.bool = np.bool_  # type: ignore[attr-defined]
    spec = importlib.util.spec_from_file_location("official_vima_example", example_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {example_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_policy(checkpoint: str | Path, device: str):
    import torch
    from vima.policy import VIMAPolicy

    try:
        checkpoint_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    policy = VIMAPolicy(**checkpoint_payload["cfg"])
    policy.load_state_dict(
        {
            key.replace("policy.", "", 1): value
            for key, value in checkpoint_payload["state_dict"].items()
        },
        strict=True,
    )
    policy.to(device)
    policy.eval()
    return policy


def _task_kwargs(config: Mapping[str, Any], cell: Mapping[str, str]) -> dict[str, Any]:
    from vima_bench import PARTITION_TO_SPECS

    benchmark = config["benchmark"]
    kwargs = copy.deepcopy(
        PARTITION_TO_SPECS["test"][benchmark["partition"]][benchmark["task"]]
    )
    if benchmark["task"] == "visual_manipulation":
        kwargs.update(
            {
                "possible_dragged_obj": [cell["target_shape"]],
                "possible_dragged_obj_texture": [cell["target_texture"]],
                "possible_base_obj": [cell["receptacle_shape"]],
                "possible_base_obj_texture": [cell["receptacle_texture"]],
            }
        )
    elif benchmark["task"] == "manipulate_old_neighbor":
        # Preserve the official partition generator exactly. Cells are obtained by outcome-blind
        # seed stratification, not by narrowing the task's object or texture support.
        pass
    else:
        raise ValueError(f"unsupported benchmark task: {benchmark['task']}")
    return kwargs


def _neighbor_direction(task) -> str:
    delta = (
        task.sampled_neighbor_idx[0] - task.sampled_dragged_obj_idx[0],
        task.sampled_neighbor_idx[1] - task.sampled_dragged_obj_idx[1],
    )
    lookup = {(-1, 0): "north", (1, 0): "south", (0, -1): "west", (0, 1): "east"}
    try:
        return lookup[delta]
    except KeyError as error:
        raise RuntimeError(f"invalid neighbor displacement: {delta}") from error


def _actual_factors(env, benchmark_task: str) -> dict[str, str]:
    task = env.unwrapped.task
    if benchmark_task == "visual_manipulation":
        target = task.placeholders["dragged_obj_1"]
        receptacle = task.placeholders["base_obj"]
        return {
            "target_shape": str(target.name),
            "target_texture": str(target.color.name),
            "receptacle_shape": str(receptacle.name),
            "receptacle_texture": str(receptacle.color.name),
        }
    if benchmark_task == "manipulate_old_neighbor":
        target = task.placeholders["dragged_obj"]
        receptacle = task.placeholders["base_obj"]
        return {
            "target_shape": str(target.name),
            "receptacle_shape": str(receptacle.name),
            "direction": _neighbor_direction(task),
        }
    raise ValueError(f"unsupported benchmark task: {benchmark_task}")


def _layout_controls(env, benchmark_task: str) -> dict[str, float | str]:
    import pybullet as bullet

    task = env.unwrapped.task
    target_key = "dragged_obj_1" if benchmark_task == "visual_manipulation" else "dragged_obj"
    target_id = task.placeholders[target_key].obj_id
    receptacle_id = task.placeholders["base_obj"].obj_id
    target_position = np.asarray(
        bullet.getBasePositionAndOrientation(
            target_id, physicsClientId=env.unwrapped.client_id
        )[0][:2],
        dtype=np.float64,
    )
    receptacle_position = np.asarray(
        bullet.getBasePositionAndOrientation(
            receptacle_id, physicsClientId=env.unwrapped.client_id
        )[0][:2],
        dtype=np.float64,
    )
    controls: dict[str, float | str] = {
        "target_receptacle_xy_distance": float(
            np.linalg.norm(target_position - receptacle_position)
        ),
        "target_x": float(target_position[0]),
        "target_y": float(target_position[1]),
        "receptacle_x": float(receptacle_position[0]),
        "receptacle_y": float(receptacle_position[1]),
    }
    if benchmark_task == "manipulate_old_neighbor":
        neighbor_id = task._neighbor_obj[0]
        neighbor_position = np.asarray(
            bullet.getBasePositionAndOrientation(
                neighbor_id, physicsClientId=env.unwrapped.client_id
            )[0][:2],
            dtype=np.float64,
        )
        controls.update(
            {
                "target_neighbor_xy_distance": float(
                    np.linalg.norm(target_position - neighbor_position)
                ),
                "neighbor_x": float(neighbor_position[0]),
                "neighbor_y": float(neighbor_position[1]),
                "target_texture_nuisance": str(task.placeholders[target_key].color.name),
                "receptacle_texture_nuisance": str(
                    task.placeholders["base_obj"].color.name
                ),
            }
        )
    return controls


def _distribution_confidence(distributions: Mapping[str, Any]) -> float:
    maxima = []
    for distribution in distributions.values():
        components = getattr(distribution, "_dists", [distribution])
        for component in components:
            maxima.append(float(component.probs.max(dim=-1).values.mean().item()))
    return float(np.mean(maxima))


def _pad_history(example, cache: dict[str, list[Any]], device: str):
    import torch

    max_objects = max(token.shape[0] for token in cache["obs_tokens"])
    observation_tokens = []
    observation_masks = []
    for token, mask in zip(cache["obs_tokens"], cache["obs_masks"]):
        required = max_objects - token.shape[0]
        observation_tokens.append(
            example.any_concat(
                [
                    token,
                    torch.zeros(
                        required,
                        token.shape[1],
                        device=device,
                        dtype=token.dtype,
                    ),
                ],
                dim=0,
            )
        )
        observation_masks.append(
            example.any_concat(
                [
                    mask,
                    torch.zeros(required, device=device, dtype=mask.dtype),
                ],
                dim=0,
            )
        )
    obs_tokens = example.any_stack(
        [example.any_stack(observation_tokens, dim=0)], dim=0
    ).transpose(0, 1)
    obs_masks = example.any_stack(
        [example.any_stack(observation_masks, dim=0)], dim=0
    ).transpose(0, 1)
    if not cache["action_tokens"]:
        action_tokens = None
    else:
        action_tokens = example.any_stack(
            [example.any_stack(cache["action_tokens"], dim=0)], dim=0
        ).transpose(0, 1)
    return obs_tokens, obs_masks, action_tokens


def _decode_environment_action(policy, distributions, meta_info, device: str):
    import torch

    actions = {key: distribution.mode() for key, distribution in distributions.items()}
    action_token = policy.forward_action_token(actions).squeeze(0)[0]
    actions = policy._de_discretize_actions(actions)
    low = torch.as_tensor(
        np.asarray([meta_info["action_bounds"]["low"]]),
        dtype=torch.float32,
        device=device,
    )
    high = torch.as_tensor(
        np.asarray([meta_info["action_bounds"]["high"]]),
        dtype=torch.float32,
        device=device,
    )
    for key in ("pose0_position", "pose1_position"):
        actions[key] = torch.clamp(actions[key] * (high - low) + low, min=low, max=high)
    for key in ("pose0_rotation", "pose1_rotation"):
        actions[key] = torch.clamp(actions[key] * 2 - 1, min=-1, max=1)
    numpy_actions = {key: value.detach().cpu().numpy()[0, 0] for key, value in actions.items()}
    return numpy_actions, action_token


def run_episode(
    *,
    policy,
    config: Mapping[str, Any],
    cell: Mapping[str, str],
    seed: int,
    device: str,
    vima_root: str | Path,
    closed_loop: bool,
) -> EpisodeResult:
    """Run one fixed VIMA-Bench episode without substituting failed seeds."""

    import torch
    from gym.wrappers import TimeLimit
    from vima_bench import make

    example = _load_official_example(vima_root)
    benchmark = config["benchmark"]
    env = make(
        benchmark["task"],
        modalities=["segm", "rgb"],
        task_kwargs=_task_kwargs(config, cell),
        seed=int(seed),
        render_prompt=False,
        display_debug_window=False,
        hide_arm_rgb=bool(benchmark["hide_arm_rgb"]),
    )
    env = TimeLimit(
        env,
        max_episode_steps=env.task.oracle_max_steps + int(benchmark["bonus_steps"]),
    )
    try:
        obs = env.reset()
        actual = _actual_factors(env, benchmark["task"])
        expected = {factor: str(cell[factor]) for factor in actual}
        if actual != expected:
            raise RuntimeError(f"factor mismatch: actual={actual}, expected={expected}")
        meta_info = env.unwrapped.meta_info
        if int(meta_info["seed"]) != int(seed):
            raise RuntimeError(
                f"seed substitution detected: actual={meta_info['seed']} requested={seed}"
            )
        layout = _layout_controls(env, benchmark["task"])

        prompt_types, words, images = example.prepare_prompt(
            prompt=env.unwrapped.prompt,
            prompt_assets=env.unwrapped.prompt_assets,
            views=["front", "top"],
        )
        words = words.to(device)
        images = images.to_torch_tensor(device=device)
        with torch.inference_mode():
            prompt_tokens, prompt_masks = policy.forward_prompt_assembly(
                (prompt_types, words, images)
            )

        cache: dict[str, list[Any]] = {
            "obs_tokens": [],
            "obs_masks": [],
            "action_tokens": [],
        }
        initial_activation = None
        initial_confidence = None
        success = False
        terminal = False
        steps = 0
        info: dict[str, Any] = {"success": False, "failure": False}
        while True:
            obs["ee"] = np.asarray(obs["ee"])
            prepared = example.add_batch_dim(obs)
            prepared = example.prepare_obs(
                obs=prepared, rgb_dict=None, meta=meta_info
            ).to_torch_tensor(device=device)
            with torch.inference_mode():
                obs_token, obs_mask = policy.forward_obs_token(prepared)
            cache["obs_tokens"].append(obs_token.squeeze(0)[0])
            cache["obs_masks"].append(obs_mask.squeeze(0)[0])
            obs_tokens, obs_masks, action_tokens = _pad_history(
                example, cache, device
            )
            with torch.inference_mode():
                predicted = policy.forward(
                    obs_token=obs_tokens,
                    action_token=action_tokens,
                    prompt_token=prompt_tokens,
                    prompt_token_mask=prompt_masks,
                    obs_mask=obs_masks,
                )
                action_facing = predicted[-1].unsqueeze(0)
                distributions = policy.forward_action_decoder(action_facing)
            if initial_activation is None:
                initial_activation = (
                    action_facing[0, 0].detach().cpu().to(torch.float32).numpy()
                )
                initial_confidence = _distribution_confidence(distributions)
            if not closed_loop:
                break
            action, action_token = _decode_environment_action(
                policy, distributions, meta_info, device
            )
            cache["action_tokens"].append(action_token)
            obs, _, terminal, info = env.step(action)
            steps += 1
            if terminal:
                success = bool(info["success"])
                break
        if initial_activation is None or initial_confidence is None:
            raise AssertionError("episode did not produce an initial action representation")
        if not np.all(np.isfinite(initial_activation)) or not math.isfinite(initial_confidence):
            raise RuntimeError("non-finite activation or action confidence")
        metadata = {
            "cell_id": str(cell["cell_id"]),
            **actual,
            "seed": int(seed),
            "closed_loop": bool(closed_loop),
            "valid": True,
            "terminal": bool(terminal),
            "success": bool(success),
            "failure": bool(info.get("failure", False)),
            "steps": int(steps),
            "initial_action_confidence": float(initial_confidence),
            **layout,
        }
        return EpisodeResult(metadata=metadata, activation=initial_activation)
    finally:
        env.close()
