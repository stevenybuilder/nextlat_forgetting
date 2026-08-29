from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def _simulation(env: Any) -> Any | None:
    current = env
    visited: set[int] = set()
    for _ in range(8):
        if id(current) in visited:
            break
        visited.add(id(current))
        sim = getattr(current, "sim", None)
        if sim is not None:
            return sim
        next_env = getattr(current, "env", None)
        if next_env is None:
            next_env = getattr(current, "unwrapped", None)
        if next_env is None or next_env is current:
            break
        current = next_env
    return None


def _body_name(model: Any, body_id: int) -> str:
    try:
        return str(model.body_id2name(int(body_id)))
    except Exception:
        try:
            return str(model.body_names[int(body_id)])
        except Exception:
            return ""


def _root_body_id(model: Any, body_id: int) -> int:
    current = int(body_id)
    try:
        parent = int(model.body_parentid[current])
    except Exception:
        return current
    while parent > 0:
        current = parent
        try:
            parent = int(model.body_parentid[current])
        except Exception:
            break
    return current


def gripper_contact_root_names(env: Any) -> set[str]:
    """Return root-body instance names currently touching a gripper body.

    The measurement follows LIBERO-CF's public touch metric at commit
    8460457bfca6e0ef2e856bc104e2c60b023ef2a7, implemented locally so the
    experiment can validate and test it without importing an evaluation script.
    """

    sim = _simulation(env)
    if sim is None or getattr(sim, "model", None) is None or getattr(sim, "data", None) is None:
        return set()
    model, data = sim.model, sim.data

    def is_gripper(name: str) -> bool:
        lowered = name.lower()
        return any(token in lowered for token in ("gripper", "finger", "hand", "panda_hand"))

    names: set[str] = set()
    for contact_index in range(int(getattr(data, "ncon", 0))):
        try:
            contact = data.contact[contact_index]
            body_1 = int(model.geom_bodyid[int(contact.geom1)])
            body_2 = int(model.geom_bodyid[int(contact.geom2)])
            name_1 = _body_name(model, body_1)
            name_2 = _body_name(model, body_2)
            if is_gripper(name_1) and name_2:
                names.add(_body_name(model, _root_body_id(model, body_2)))
            elif is_gripper(name_2) and name_1:
                names.add(_body_name(model, _root_body_id(model, body_1)))
        except Exception:
            continue
    return {name for name in names if name}


def resolve_instance_root_body_name(env: Any, instance_name: str) -> str | None:
    """Resolve a LIBERO BDDL instance token to its MuJoCo root body name."""

    sim = _simulation(env)
    if sim is None or getattr(sim, "model", None) is None:
        return None
    model = sim.model
    try:
        model.body_name2id(instance_name)
        return instance_name
    except Exception:
        pass

    matches: list[int] = []
    for body_id in range(int(getattr(model, "nbody", 0))):
        if _body_name(model, body_id).startswith(instance_name):
            matches.append(body_id)
    if not matches:
        return None
    root_counts = Counter(_root_body_id(model, body_id) for body_id in matches)
    root_body, _ = root_counts.most_common(1)[0]
    resolved = _body_name(model, root_body)
    return resolved or None


def touched_instances(env: Any, instance_names: Iterable[str]) -> set[str]:
    contacts = gripper_contact_root_names(env)
    touched: set[str] = set()
    for instance_name in instance_names:
        resolved = resolve_instance_root_body_name(env, instance_name)
        if resolved is not None and resolved in contacts:
            touched.add(instance_name)
    return touched

