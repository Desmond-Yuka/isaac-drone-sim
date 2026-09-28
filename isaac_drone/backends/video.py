"""Optional, synchronized offscreen cameras for Isaac Lab 3.0 / Isaac Sim 6.1.

Importing this module does not import Kit or Replicator. Construct the rig only
after AppLauncher(enable_cameras=True) and simulation reset. The cameras have
their own render products, so neither a viewport nor WebRTC is required.
"""
from __future__ import annotations

import contextlib
import uuid

import numpy as np


class IsaacCameraRig:
    """Capture named RGB views at one physical instant, without stepping physics.

    Poses contain world-space ``eye`` and ``target``, with optional ``up``
    (default world Z). Use world Y as up for a vertical overhead camera.
    Horizontal aperture is 36 mm and focal length 24 mm: horizontal FOV 73.74°.
    Vertical aperture preserves the requested image aspect ratio.
    """

    HORIZONTAL_APERTURE_MM = 36.0
    FOCAL_LENGTH_MM = 24.0
    WARMUP_UPDATES = 120

    def __init__(self, sim, width: int, height: int, views: dict[str, dict]):
        if isinstance(width, bool) or isinstance(height, bool) or int(width) != width or int(height) != height:
            raise ValueError("Camera dimensions must be positive integers")
        if width <= 0 or height <= 0 or not views:
            raise ValueError("Camera dimensions and views must be nonempty")
        self._sim = sim
        self.width, self.height = int(width), int(height)
        self._stage = None
        self._root_path = f"/World/RecordingCameras_{uuid.uuid4().hex}"
        self._cameras = {}
        self._products = {}
        self._annotators = {}
        self._closed = False

        # Simulator-only imports remain inside the explicitly enabled path.
        import carb.settings
        import omni.kit.app
        import omni.replicator.core as rep
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Gf, UsdGeom

        self._Gf = Gf
        self._settings = carb.settings.get_settings()
        self._app = omni.kit.app.get_app()
        self._stage = get_current_stage()
        try:
            UsdGeom.Xform.Define(self._stage, self._root_path)
            for index, name in enumerate(views):
                # User-facing names are not interpreted as USD paths.
                path = f"{self._root_path}/Camera_{index}"
                camera = UsdGeom.Camera.Define(self._stage, path)
                camera.CreateFocalLengthAttr(self.FOCAL_LENGTH_MM)
                camera.CreateHorizontalApertureAttr(self.HORIZONTAL_APERTURE_MM)
                camera.CreateVerticalApertureAttr(self.HORIZONTAL_APERTURE_MM * self.height / self.width)
                camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10000.0))
                self._cameras[name] = camera.AddTransformOp()
                product = rep.create.render_product(path, (self.width, self.height))
                self._products[name] = product
                annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._annotators[name] = annotator
                annotator.attach([product])
            self._set_poses(views)
            # RTX may initially publish empty or black buffers. Warm it up at
            # the unchanged initial physical state, without encoding fake frames.
            for attempt in range(self.WARMUP_UPDATES):
                frames = self._capture(allow_empty=True)
                if attempt >= 2 and all(frame is not None and np.any(frame) for frame in frames.values()):
                    break
            else:
                raise RuntimeError(
                    "Recording cameras produced empty or black frames during warmup. "
                    "Check --enable_cameras, scene lighting, camera poses and RTX availability."
                )
        except BaseException:
            self.close()
            raise

    def _set_poses(self, poses):
        if set(poses) != set(self._cameras):
            raise ValueError("Camera poses must contain exactly the configured views")
        transforms = {}
        for name, pose in poses.items():
            eye = np.asarray(pose["eye"], dtype=float)
            target = np.asarray(pose["target"], dtype=float)
            up = np.asarray(pose.get("up", (0.0, 0.0, 1.0)), dtype=float)
            if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in (eye, target, up)):
                raise ValueError(f"Camera {name!r} needs finite 3D eye, target and up vectors")
            direction = target - eye
            if np.linalg.norm(direction) <= 1e-10 or np.linalg.norm(np.cross(direction, up)) <= 1e-10:
                raise ValueError(f"Camera {name!r} needs distinct eye/target and a nonparallel up vector")
            # USD cameras look along local -Z with +Y up. Invert the Gf view
            # matrix to author the camera's world transform directly.
            transforms[name] = self._Gf.Matrix4d().SetLookAt(
                self._Gf.Vec3d(*map(float, eye)),
                self._Gf.Vec3d(*map(float, target)),
                self._Gf.Vec3d(*map(float, up)),
            ).GetInverse()
        for name, transform in transforms.items():
            self._cameras[name].Set(transform)

    def _capture(self, *, allow_empty=False):
        if self._closed:
            raise RuntimeError("Recording camera rig is closed")
        if not self._app.is_running():
            raise RuntimeError("Isaac Sim stopped before the recording frame was captured")
        try:
            for product in self._products.values():
                product.hydra_texture.set_updates_enabled(True)
            # SimulationContext.render() alone does not pump Kit in Lab 3.0
            # when --visualizer none is selected. forward() explicitly syncs
            # PhysX poses into Fabric before the shared offscreen render.
            self._sim.forward()
            setting = "/app/player/playSimulations"
            previous = self._settings.get(setting)
            self._settings.set_bool(setting, False)
            try:
                self._app.update()
            finally:
                if previous is None:
                    self._settings.destroy_item(setting)
                else:
                    self._settings.set_bool(setting, bool(previous))
            frames = {}
            for name, annotator in self._annotators.items():
                raw = annotator.get_data()
                if isinstance(raw, dict):
                    raw = raw.get("data", [])
                array = np.asarray(raw)
                if array.size == 0 and allow_empty:
                    frames[name] = None
                    continue
                if array.dtype != np.uint8 or array.size not in (
                    self.height * self.width * 3, self.height * self.width * 4,
                ):
                    raise RuntimeError(f"Camera {name!r} returned an invalid RGB frame: {array.shape}, {array.dtype}")
                if array.ndim == 1:
                    array = array.reshape(self.height, self.width, -1)
                if array.ndim != 3 or array.shape[:2] != (self.height, self.width):
                    raise RuntimeError(f"Camera {name!r} returned an unexpected image shape: {array.shape}")
                # Copy before the next render can overwrite annotator memory.
                frames[name] = np.array(array[:, :, :3], dtype=np.uint8, order="C", copy=True)
            return frames
        finally:
            # UI/stream redraws between video samples must not render every
            # recording camera. Re-enable updates only at the next capture.
            for product in self._products.values():
                with contextlib.suppress(Exception):
                    product.hydra_texture.set_updates_enabled(False)

    def read(self, poses: dict[str, dict]) -> dict[str, np.ndarray]:
        """Update all cameras, then synchronously capture all requested views."""
        if self._closed:
            raise RuntimeError("Recording camera rig is closed")
        self._set_poses(poses)
        return self._capture()

    def close(self):
        """Release only this rig's resources; safe after partial initialization."""
        if self._closed:
            return
        self._closed = True
        for annotator in self._annotators.values():
            with contextlib.suppress(Exception):
                annotator.detach()
        self._annotators.clear()
        for product in self._products.values():
            with contextlib.suppress(Exception):
                product.destroy()
        self._products.clear()
        self._cameras.clear()
        if self._stage is not None:
            with contextlib.suppress(Exception):
                self._stage.RemovePrim(self._root_path)
