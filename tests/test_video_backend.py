"""Offscreen capture lifecycle contracts; no claim of GPU/RTX verification."""

import sys
import types

import numpy as np
import pytest

from isaac_drone.sim.isaaclab.camera import IsaacCameraRig

PLAY_SETTING = "/app/player/playSimulations"


@pytest.fixture
def kit(monkeypatch):
    state = types.SimpleNamespace(
        updates=0,
        physics_steps=0,
        forwards=0,
        failure=False,
        settings={PLAY_SETTING: True},
        products=[],
        annotators=[],
        removed=[],
    )

    def module(name, **attributes):
        value = types.ModuleType(name)
        value.__path__ = []
        value.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, value)
        if "." in name:
            parent, key = name.rsplit(".", 1)
            setattr(sys.modules[parent], key, value)
        return value

    class Product:
        def __init__(self, path, size):
            self.path, self.size = path, size
            self.paused, self.destroyed = False, 0
            # Replicator 1.13.36 exposes the render switch on the nested
            # Hydra texture; the product has no pause()/resume() methods.
            self.hydra_texture = types.SimpleNamespace(set_updates_enabled=self._set_updates_enabled)
            state.products.append(self)

        def _set_updates_enabled(self, enabled):
            self.paused = not enabled

        def destroy(self):
            self.destroyed += 1

    class Annotator:
        def __init__(self, *args, **kwargs):
            self.detached = 0
            self.buffer = np.zeros((2, 4, 4), dtype=np.uint8)
            state.annotators.append(self)

        def attach(self, products):
            self.product = products[0]

        def detach(self):
            self.detached += 1

        def get_data(self):
            # Simulate delayed first annotator output and reused CPU buffers.
            if state.updates == 1:
                return np.array([], dtype=np.uint8)
            self.buffer[:] = state.updates if state.updates > 2 else 0
            return {"data": self.buffer}

    class Matrix:
        def SetLookAt(self, *values):
            self.pose = values
            return self

        def GetInverse(self):
            return self

    class Camera:
        @staticmethod
        def Define(stage, path):
            return Camera()

        def __getattr__(self, name):
            return lambda value: None

        def AddTransformOp(self):
            return types.SimpleNamespace(Set=lambda value: None)

    def update():
        state.updates += 1
        if state.settings.get(PLAY_SETTING, True):
            state.physics_steps += 1
        assert all(not product.paused for product in state.products)
        if state.failure:
            raise RuntimeError("renderer failed")

    def forward():
        state.forwards += 1

    settings = types.SimpleNamespace(
        get=state.settings.get,
        set_bool=state.settings.__setitem__,
        destroy_item=lambda name: state.settings.pop(name, None),
    )
    stage = types.SimpleNamespace(RemovePrim=state.removed.append)
    module("carb")
    module("carb.settings", get_settings=lambda: settings)
    module("omni")
    module("omni.kit")
    module("omni.kit.app", get_app=lambda: types.SimpleNamespace(update=update, is_running=lambda: True))
    module("omni.replicator")
    module(
        "omni.replicator.core",
        create=types.SimpleNamespace(render_product=Product),
        AnnotatorRegistry=types.SimpleNamespace(get_annotator=Annotator),
    )
    module("isaaclab")
    module("isaaclab.sim")
    module("isaaclab.sim.utils")
    module("isaaclab.sim.utils.stage", get_current_stage=lambda: stage)
    module(
        "pxr",
        Gf=types.SimpleNamespace(Matrix4d=Matrix, Vec3d=lambda *v: v, Vec2f=lambda *v: v),
        UsdGeom=types.SimpleNamespace(Camera=Camera, Xform=Camera),
    )
    state.sim = types.SimpleNamespace(forward=forward)
    return state


def views():
    return {
        "follow": {"eye": [2, -2, 2], "target": [0, 0, 0]},
        "top": {"eye": [0, 0, 5], "target": [0, 0, 0], "up": [0, 1, 0]},
    }


def test_all_views_capture_one_instant_without_stepping_and_own_their_buffers(kit):
    rig = IsaacCameraRig(kit.sim, 4, 2, views())
    assert kit.updates == 3  # Empty and black startup frames were discarded.
    assert kit.physics_steps == 0
    poses = views()
    poses["follow"]["target"] = [1, 0, 0]
    first = rig.read(poses)
    assert kit.updates == 4 and kit.forwards == 4
    assert all(frame.shape == (2, 4, 3) and np.all(frame == 4) for frame in first.values())
    assert all(product.paused for product in kit.products)
    rig.read(poses)
    assert all(np.all(frame == 4) for frame in first.values())
    assert kit.physics_steps == 0
    rig.close()
    rig.close()
    assert all(product.destroyed == 1 for product in kit.products)
    assert all(annotator.detached == 1 for annotator in kit.annotators)
    assert len(kit.removed) == 1
    assert kit.removed[0].startswith("/World/RecordingCameras_")


@pytest.mark.parametrize("previous", [True, False, None])
def test_render_failure_restores_setting_and_pauses_all_products(kit, previous):
    rig = IsaacCameraRig(kit.sim, 4, 2, views())
    if previous is None:
        kit.settings.pop(PLAY_SETTING)
    else:
        kit.settings[PLAY_SETTING] = previous
    kit.failure = True
    with pytest.raises(RuntimeError, match="renderer failed"):
        rig.read(views())
    assert kit.settings.get(PLAY_SETTING) is previous
    assert all(product.paused for product in kit.products)
    rig.close()


def test_constructor_failure_releases_partially_initialized_cameras(kit):
    kit.failure = True
    with pytest.raises(RuntimeError, match="renderer failed"):
        IsaacCameraRig(kit.sim, 4, 2, views())
    assert all(product.destroyed == 1 for product in kit.products)
    assert all(annotator.detached == 1 for annotator in kit.annotators)
    assert len(kit.removed) == 1
    assert kit.settings[PLAY_SETTING] is True


def test_vertical_camera_requires_nonparallel_up_vector(kit):
    poses = views()
    poses["top"].pop("up")
    with pytest.raises(ValueError, match="nonparallel up"):
        IsaacCameraRig(kit.sim, 4, 2, poses)
    assert kit.updates == 0
    assert all(product.destroyed == 1 for product in kit.products)


def test_warmup_fails_instead_of_recording_blank_frames(kit, monkeypatch):
    monkeypatch.setattr(IsaacCameraRig, "WARMUP_UPDATES", 2)
    with pytest.raises(RuntimeError, match="empty or black frames"):
        IsaacCameraRig(kit.sim, 4, 2, views())
    assert all(product.destroyed == 1 for product in kit.products)
    assert kit.physics_steps == 0
