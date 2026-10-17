import argparse
from pathlib import Path
import numpy as np
import trimesh
import pyrender
import imageio


def normalize_mesh(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values()]
        )

    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)

    scale = np.max(mesh.extents)
    if scale > 0:
        mesh.apply_scale(1.0 / scale)

    return mesh


def look_at(camera_position, target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0])):
    """
    生成标准 look-at 相机位姿矩阵
    """
    forward = target - camera_position
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)

    true_up = np.cross(right, forward)
    true_up = true_up / (np.linalg.norm(true_up) + 1e-8)

    pose = np.eye(4)
    pose[0:3, 0] = right
    pose[0:3, 1] = true_up
    pose[0:3, 2] = -forward
    pose[0:3, 3] = camera_position
    return pose


def render_views(obj_path: Path, output_dir: Path, num_views: int = 12, image_size: int = 512):
    loaded = trimesh.load(obj_path, force="scene")

    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            print(f"[跳过] 空模型: {obj_path}")
            return
        mesh = trimesh.util.concatenate([g for g in loaded.geometry.values()])
    else:
        mesh = loaded

    mesh = normalize_mesh(mesh)

    scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.8, 0.8, 0.8])
    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(render_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

    light1 = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    light2 = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light3 = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)

    renderer = pyrender.OffscreenRenderer(image_size, image_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    radius = 2.5

    for i in range(num_views):
        theta = 2.0 * np.pi * i / num_views
        height = 0.6

        cam_x = radius * np.cos(theta)
        cam_y = height
        cam_z = radius * np.sin(theta)

        cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float32)
        cam_pose = look_at(cam_pos)

        cam_node = scene.add(camera, pose=cam_pose)
        light_node1 = scene.add(light1, pose=cam_pose)

        light_pose2 = np.eye(4)
        light_pose2[:3, 3] = [-2.0, 2.0, 2.0]
        light_node2 = scene.add(light2, pose=light_pose2)

        light_pose3 = np.eye(4)
        light_pose3[:3, 3] = [2.0, -1.0, -2.0]
        light_node3 = scene.add(light3, pose=light_pose3)

        color, depth = renderer.render(scene)

        save_path = output_dir / f"view_{i + 1:02d}.png"
        imageio.imwrite(save_path, color)

        scene.remove_node(cam_node)
        scene.remove_node(light_node1)
        scene.remove_node(light_node2)
        scene.remove_node(light_node3)

    renderer.delete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=12)
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)

    obj_files = sorted(models_dir.glob("*.obj"), key=lambda x: int(x.stem))

    for obj_path in obj_files:
        save_dir = output_dir / obj_path.stem
        print(f"Rendering {obj_path.name} -> {save_dir}")
        try:
            render_views(obj_path, save_dir, num_views=args.num_views)
        except Exception as e:
            print(f"[失败] {obj_path.name}: {e}")