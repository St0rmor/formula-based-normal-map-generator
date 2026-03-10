from __future__ import annotations
from pathlib import Path
from PIL import Image
import numpy as np


total_width_px = 1024.0
total_height_px = 512.0
strength = 20.0



def height_function(x_norm: np.ndarray, y_norm: np.ndarray) -> np.ndarray:

    cone_width_px = 65.0
    cylinder_width_px = 894.0

    body_radius_px = 256.0   # 512 / 2
    tip_radius_px = 136.0    # 272 / 2

    cone_w = cone_width_px / total_width_px
    cyl_w = cylinder_width_px / total_width_px

    # Region boundaries in normalized x
    x_left_cone_end = cone_w
    x_cylinder_end = cone_w + cyl_w

    # Radius profile along x
    r = np.zeros_like(x_norm, dtype=np.float32)

    # Left truncated cone: tip -> body
    left_mask = x_norm < x_left_cone_end
    if np.any(left_mask):
        t = x_norm[left_mask] / x_left_cone_end
        r[left_mask] = tip_radius_px + t * (body_radius_px - tip_radius_px)

    # Constant radius cylinder
    center_mask = (x_norm >= x_left_cone_end) & (x_norm < x_cylinder_end)
    r[center_mask] = body_radius_px

    # Right truncated cone: body -> tip
    right_mask = x_norm >= x_cylinder_end
    if np.any(right_mask):
        t = (x_norm[right_mask] - x_cylinder_end) / (1.0 - x_cylinder_end)
        r[right_mask] = body_radius_px + t * (tip_radius_px - body_radius_px)

    # Convert y to centered pixel coordinate
    y_px = y_norm * total_height_px
    y_centered = y_px - total_height_px * 0.5

    # Surface height: z = sqrt(r(x)^2 - y^2)
    inside = r**2 - y_centered**2
    mask = inside > 0

    inside = np.clip(inside, 0.0, None)
    h = np.sqrt(inside)

    h /= body_radius_px

    return h.astype(np.float32), mask


def compute_normal_map(height_map: np.ndarray, strength: float = 1.0) -> np.ndarray:

    dh_dy, dh_dx = np.gradient(height_map)

    nx = -dh_dx * strength
    ny = -dh_dy * strength
    nz = np.ones_like(height_map, dtype=np.float32)

    normals = np.stack((nx, ny, nz), axis=-1)

    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    lengths = np.maximum(lengths, 1e-8)
    normals /= lengths

    normal_map = ((normals + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    return normal_map


def save_grayscale_image(height_map, mask, output_path):
    img = (height_map * 255).astype(np.uint8)
    alpha = np.where(mask, 255, 0).astype(np.uint8)

    rgba = np.dstack((img, img, img, alpha))
    Image.fromarray(rgba, mode="RGBA").save(output_path)


def save_rgb_image(image_array: np.ndarray, mask: np.ndarray, output_path: Path) -> None:
    alpha = np.where(mask, 255, 0).astype(np.uint8)
    rgba = np.dstack((image_array, alpha))
    Image.fromarray(rgba, mode="RGBA").save(output_path)


def main() -> None:
    width = int(total_width_px)
    height = int(total_height_px)
    
    height_output_path = Path("height.png")
    normal_output_path = Path("normal.png")

    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x, y)

    height_map, mask = height_function(x_grid, y_grid)
    normal_map = compute_normal_map(height_map, strength=strength)

    save_grayscale_image(height_map, mask, height_output_path)
    save_rgb_image(normal_map, mask, normal_output_path)

    print(f"Saved height map to: {height_output_path.resolve()}")
    print(f"Saved normal map to: {normal_output_path.resolve()}")


if __name__ == "__main__":
    main()