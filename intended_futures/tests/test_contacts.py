from types import SimpleNamespace

from intended_futures.contacts import (
    gripper_contact_root_names,
    resolve_instance_root_body_name,
    touched_instances,
)


class _FakeModel:
    body_names = ["world", "robot0_right_finger", "cookies_1", "cookies_1_visual"]
    body_parentid = [0, 0, 0, 2]
    geom_bodyid = [1, 3]
    nbody = 4

    def body_id2name(self, body_id):
        return self.body_names[body_id]

    def body_name2id(self, name):
        if name not in self.body_names:
            raise KeyError(name)
        return self.body_names.index(name)


def _environment():
    contact = SimpleNamespace(geom1=0, geom2=1)
    sim = SimpleNamespace(
        model=_FakeModel(),
        data=SimpleNamespace(ncon=1, contact=[contact]),
    )
    return SimpleNamespace(sim=sim)


def test_contact_metric_returns_object_root():
    assert gripper_contact_root_names(_environment()) == {"cookies_1"}


def test_instance_resolution_and_touch_identity():
    env = _environment()
    assert resolve_instance_root_body_name(env, "cookies_1") == "cookies_1"
    assert touched_instances(env, ["cookies_1", "ramekin_1"]) == {"cookies_1"}

