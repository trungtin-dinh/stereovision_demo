import tempfile
from typing import Optional, Tuple
import re
import cv2
import gradio as gr
import numpy as np
import plotly.graph_objects as go


MAX_DISPLAY_SIZE = 1200
MAX_ORB_MATCHES = 1200
DEFAULT_POINT_COUNT = 40000
DEFAULT_LEFT_URL = "https://vision.middlebury.edu/stereo/data/scenes2021/data/artroom1/im0.png"
DEFAULT_RIGHT_URL = "https://vision.middlebury.edu/stereo/data/scenes2021/data/artroom1/im1.png"
LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]

with open("documentation_fr.md", "r", encoding="utf-8") as f:
    DOCUMENTATION_fr = f.read()

with open("documentation_en.md", "r", encoding="utf-8") as f:
    DOCUMENTATION_en = f.read()


def split_markdown_by_h2(markdown_text: str) -> dict[str, str]:
    sections = {}
    parts = re.split(r"(?m)^##\s+", markdown_text.strip())

    for part in parts:
        part = part.strip()
        if not part:
            continue

        lines = part.splitlines()
        title = lines[0].strip()

        if title.lower() in {"table des matières", "table of contents"}:
            continue

        sections[title] = "## " + part

    return sections


DOC_FR_SECTIONS = split_markdown_by_h2(DOCUMENTATION_fr)
DOC_EN_SECTIONS = split_markdown_by_h2(DOCUMENTATION_en)

DOC_FR_TITLES = list(DOC_FR_SECTIONS.keys())
DOC_EN_TITLES = list(DOC_EN_SECTIONS.keys())


def load_doc_fr_section(title: str) -> str:
    return DOC_FR_SECTIONS[title]


def load_doc_en_section(title: str) -> str:
    return DOC_EN_SECTIONS[title]


def ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Missing image.")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        pass
    else:
        raise ValueError("Unsupported image format.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def resize_for_speed(image: np.ndarray, max_size: int = MAX_DISPLAY_SIZE) -> np.ndarray:
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_size:
        return image

    scale = max_size / float(longest_side)
    new_width = max(32, int(round(width * scale)))
    new_height = max(32, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def make_same_size(left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    if left_image.shape[:2] == right_image.shape[:2]:
        return left_image, right_image, False

    height, width = left_image.shape[:2]
    resized_right = cv2.resize(right_image, (width, height), interpolation=cv2.INTER_AREA)
    return left_image, resized_right, True


def detect_orb_matches(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    max_matches: int = MAX_ORB_MATCHES,
) -> Tuple[np.ndarray, np.ndarray, list, list, list]:
    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
    keypoints_left, descriptors_left = orb.detectAndCompute(left_gray, None)
    keypoints_right, descriptors_right = orb.detectAndCompute(right_gray, None)

    if descriptors_left is None or descriptors_right is None:
        raise ValueError("Not enough visual features were detected.")
    if len(keypoints_left) < 8 or len(keypoints_right) < 8:
        raise ValueError("Not enough visual features were detected.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_left, descriptors_right)
    matches = sorted(matches, key=lambda match: match.distance)
    matches = matches[:max_matches]

    if len(matches) < 20:
        raise ValueError("Not enough feature correspondences were found between the two images.")

    points_left = np.float32([keypoints_left[m.queryIdx].pt for m in matches])
    points_right = np.float32([keypoints_right[m.trainIdx].pt for m in matches])
    return points_left, points_right, matches, keypoints_left, keypoints_right


def estimate_rectified_status(
    points_left: np.ndarray,
    points_right: np.ndarray,
    vertical_threshold_px: float,
) -> Tuple[bool, float]:
    vertical_residuals = np.abs(points_left[:, 1] - points_right[:, 1])
    median_vertical_residual = float(np.median(vertical_residuals))
    is_rectified = median_vertical_residual <= float(vertical_threshold_px)
    return is_rectified, median_vertical_residual


def draw_matches(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    keypoints_left,
    keypoints_right,
    matches,
    inlier_mask=None,
) -> np.ndarray:
    if inlier_mask is None:
        shown_matches = matches[:80]
    else:
        shown_matches = [m for m, keep in zip(matches, inlier_mask.ravel().tolist()) if keep]
        if len(shown_matches) == 0:
            shown_matches = matches[:80]
        else:
            shown_matches = shown_matches[:80]

    visual = cv2.drawMatches(
        left_bgr,
        keypoints_left,
        right_bgr,
        keypoints_right,
        shown_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return cv2.cvtColor(visual, cv2.COLOR_BGR2RGB)


def compute_fundamental_matrix(points_left: np.ndarray, points_right: np.ndarray):
    fundamental_matrix, inlier_mask = cv2.findFundamentalMat(
        points_left,
        points_right,
        cv2.FM_RANSAC,
        1.0,
        0.99,
    )

    if fundamental_matrix is None or inlier_mask is None:
        raise ValueError("Failed to estimate the fundamental matrix.")

    inlier_mask = inlier_mask.astype(bool).ravel()
    inlier_points_left = points_left[inlier_mask]
    inlier_points_right = points_right[inlier_mask]

    if len(inlier_points_left) < 12:
        raise ValueError("Not enough RANSAC inliers were found.")

    return fundamental_matrix, inlier_mask, inlier_points_left, inlier_points_right


def rectify_uncalibrated(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    fundamental_matrix: np.ndarray,
    inlier_points_left: np.ndarray,
    inlier_points_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = left_rgb.shape[:2]
    ok, homography_left, homography_right = cv2.stereoRectifyUncalibrated(
        inlier_points_left,
        inlier_points_right,
        fundamental_matrix,
        imgSize=(width, height),
    )

    if not ok:
        raise ValueError("Failed to rectify the image pair.")

    rectified_left = cv2.warpPerspective(left_rgb, homography_left, (width, height))
    rectified_right = cv2.warpPerspective(right_rgb, homography_right, (width, height))
    return rectified_left, rectified_right


def validate_positive_float(value, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"{name} must be a valid number.") from exc

    if not np.isfinite(value) or value <= 0:
        raise gr.Error(f"{name} must be strictly greater than 0.")

    return value


def validate_positive_int(value, name: str) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"{name} must be an integer.") from exc

    if value <= 0:
        raise gr.Error(f"{name} must be strictly greater than 0.")

    return value


def validate_nonnegative_int(value, name: str) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"{name} must be an integer.") from exc

    if value < 0:
        raise gr.Error(f"{name} must be greater than or equal to 0.")

    return value


def validate_integer_range(value, name: str, minimum: int, maximum: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"{name} must be an integer.") from exc

    if value < minimum or value > maximum:
        raise gr.Error(f"{name} must be between {minimum} and {maximum}.")

    return value


def validate_disparity_parameters(
    image_width: int,
    image_height: int,
    num_disparities,
    block_size,
    uniqueness_ratio,
    speckle_window_size,
    speckle_range,
):
    max_num_disparities = ((image_width - 1) // 16) * 16
    if max_num_disparities < 16:
        raise gr.Error("The processed image width is too small for StereoSGBM.")

    num_disparities = int(num_disparities)
    if num_disparities < 16:
        raise gr.Error("Num Disparities must be at least 16.")
    if num_disparities > max_num_disparities:
        raise gr.Error(
            f"Num Disparities must be less than the processed image width. "
            f"For this image, the maximum allowed value is {max_num_disparities}."
        )
    if num_disparities % 16 != 0:
        raise gr.Error("Num Disparities must be a multiple of 16.")

    max_block_size = min(image_width, image_height)
    block_size = int(block_size)
    if block_size < 3:
        raise gr.Error("Block Size must be at least 3.")
    if block_size > max_block_size:
        raise gr.Error(
            f"Block Size cannot be greater than the processed image size. "
            f"For this image, the maximum allowed value is {max_block_size}."
        )
    if block_size % 2 == 0:
        raise gr.Error("Block Size must be an odd integer.")

    uniqueness_ratio = validate_integer_range(uniqueness_ratio, "Uniqueness Ratio", 0, 100)
    speckle_window_size = validate_nonnegative_int(speckle_window_size, "Speckle Window Size")
    speckle_range = validate_nonnegative_int(speckle_range, "Speckle Range")

    return (
        num_disparities,
        block_size,
        uniqueness_ratio,
        speckle_window_size,
        speckle_range,
    )


def compute_disparity(
    left_rectified_gray: np.ndarray,
    right_rectified_gray: np.ndarray,
    num_disparities: int,
    block_size: int,
    uniqueness_ratio: int,
    speckle_window_size: int,
    speckle_range: int,
) -> np.ndarray:
    p1 = 8 * block_size * block_size
    p2 = 32 * block_size * block_size

    stereo_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=p1,
        P2=p2,
        disp12MaxDiff=1,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disparity = stereo_matcher.compute(left_rectified_gray, right_rectified_gray).astype(np.float32) / 16.0
    disparity[disparity <= 0] = np.nan
    return disparity


def colorize_map(values: np.ndarray, invert: bool = False) -> np.ndarray:
    valid_mask = np.isfinite(values)
    normalized = np.zeros(values.shape, dtype=np.uint8)

    if not np.any(valid_mask):
        color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    clipped = values.copy()
    low = np.nanpercentile(clipped, 2)
    high = np.nanpercentile(clipped, 98)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        high = low + 1.0

    clipped = np.clip(clipped, low, high)
    clipped = 255.0 * (clipped - low) / (high - low)
    clipped = np.nan_to_num(clipped, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)

    if invert:
        clipped = 255 - clipped

    normalized[valid_mask] = clipped[valid_mask]
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid_mask] = 0
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def compute_depth_map(disparity: np.ndarray, focal_length_px: float, baseline_m: float) -> np.ndarray:
    return (focal_length_px * baseline_m) / disparity


def build_point_cloud(
    disparity: np.ndarray,
    color_rgb: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
    max_points: int,
):
    valid_mask = np.isfinite(disparity) & (disparity > 0)
    if np.count_nonzero(valid_mask) < 100:
        return None, None

    height, width = disparity.shape
    ys, xs = np.where(valid_mask)
    disparity_values = disparity[ys, xs]

    z = (focal_length_px * baseline_m) / disparity_values
    cx = width / 2.0
    cy = height / 2.0
    x = -(xs - cx) * z / focal_length_px
    y = -(ys - cy) * z / focal_length_px

    points = np.stack([x, y, z], axis=1)
    colors = color_rgb[ys, xs]

    keep = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    points = points[keep]
    colors = colors[keep]

    if len(points) == 0:
        return None, None

    z_low = float(np.percentile(points[:, 2], 2))
    z_high = float(np.percentile(points[:, 2], 98))
    keep = (points[:, 2] >= z_low) & (points[:, 2] <= z_high)
    points = points[keep]
    colors = colors[keep]

    if len(points) == 0:
        return None, None

    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = points[indices]
        colors = colors[indices]

    return points, colors


def make_point_cloud_plot(
    color_rgb: np.ndarray,
    disparity: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
    max_points: int,
):
    points, colors = build_point_cloud(
        disparity,
        color_rgb,
        focal_length_px,
        baseline_m,
        max_points=max_points,
    )

    if points is None:
        figure = go.Figure()
        figure.update_layout(
            title="Point Cloud Unavailable",
            scene=dict(xaxis_title="X", yaxis_title="Depth", zaxis_title="Z"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return figure

    plot_x = -points[:, 0]
    plot_y = points[:, 2]
    plot_z = points[:, 1]

    rgb_strings = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in colors]

    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=plot_x,
                y=plot_y,
                z=plot_z,
                mode="markers",
                marker=dict(size=2, color=rgb_strings, opacity=0.8),
            )
        ]
    )

    figure.update_layout(
        title="3D Point Cloud",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Depth",
            zaxis_title="Z",
            aspectmode="manual",
            aspectratio=dict(x=1.6, y=1.2, z=1.0),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0.0, y=-2.8, z=0.05),
                projection=dict(type="orthographic"),
            ),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return figure


def write_ply(
    ply_path: str,
    color_rgb: np.ndarray,
    disparity: np.ndarray,
    focal_length_px: float,
    baseline_m: float,
    max_points: int,
) -> Optional[str]:
    points, colors = build_point_cloud(
        disparity,
        color_rgb,
        focal_length_px,
        baseline_m,
        max_points=max_points,
    )
    if points is None:
        return None

    with open(ply_path, "w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            handle.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

    return ply_path


def format_depth_summary(depth_map: np.ndarray) -> str:
    valid_depth = depth_map[np.isfinite(depth_map)]
    if valid_depth.size == 0:
        return "Approx. depth: unavailable"

    depth_low = float(np.nanpercentile(valid_depth, 2))
    depth_median = float(np.nanmedian(valid_depth))
    depth_high = float(np.nanpercentile(valid_depth, 98))
    return f"Approx. depth (2%, median, 98%): {depth_low:.3f}, {depth_median:.3f}, {depth_high:.3f} m"


def run_stereo_demo(
    left_image: np.ndarray,
    right_image: np.ndarray,
    focal_length_px,
    baseline_m,
    auto_resize: bool,
    rectification_mode: str,
    auto_rectification_threshold_px,
    num_disparities,
    block_size,
    uniqueness_ratio,
    speckle_window_size,
    speckle_range,
    point_count,
):
    if left_image is None or right_image is None:
        raise gr.Error("Please upload exactly two images.")

    focal_length_px = validate_positive_float(focal_length_px, "Focal Length")
    baseline_m = validate_positive_float(baseline_m, "Baseline")
    auto_rectification_threshold_px = validate_positive_float(
        auto_rectification_threshold_px,
        "Auto Rectification Threshold",
    )
    point_count = validate_positive_int(point_count, "Point Count")

    left_rgb = ensure_uint8_rgb(left_image)
    right_rgb = ensure_uint8_rgb(right_image)

    if auto_resize:
        left_rgb = resize_for_speed(left_rgb)
        right_rgb = resize_for_speed(right_rgb)

    left_rgb, right_rgb, _ = make_same_size(left_rgb, right_rgb)

    image_height, image_width = left_rgb.shape[:2]
    (
        num_disparities,
        block_size,
        uniqueness_ratio,
        speckle_window_size,
        speckle_range,
    ) = validate_disparity_parameters(
        image_width=image_width,
        image_height=image_height,
        num_disparities=num_disparities,
        block_size=block_size,
        uniqueness_ratio=uniqueness_ratio,
        speckle_window_size=speckle_window_size,
        speckle_range=speckle_range,
    )

    left_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
    right_bgr = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    points_left, points_right, matches, keypoints_left, keypoints_right = detect_orb_matches(
        left_gray,
        right_gray,
    )
    is_rectified, _ = estimate_rectified_status(
        points_left,
        points_right,
        auto_rectification_threshold_px,
    )

    fundamental_matrix, inlier_mask, inlier_points_left, inlier_points_right = compute_fundamental_matrix(
        points_left,
        points_right,
    )
    ransac_inliers = len(inlier_points_left)

    matches_view = draw_matches(
        left_bgr,
        right_bgr,
        keypoints_left,
        keypoints_right,
        matches,
        inlier_mask,
    )

    if rectification_mode == "Already Rectified":
        apply_rectification = False
    elif rectification_mode == "Estimate And Rectify":
        apply_rectification = True
    else:
        apply_rectification = not is_rectified

    if apply_rectification:
        used_left_rgb, used_right_rgb = rectify_uncalibrated(
            left_rgb,
            right_rgb,
            fundamental_matrix,
            inlier_points_left,
            inlier_points_right,
        )
    else:
        used_left_rgb, used_right_rgb = left_rgb, right_rgb

    used_left_gray = cv2.cvtColor(used_left_rgb, cv2.COLOR_RGB2GRAY)
    used_right_gray = cv2.cvtColor(used_right_rgb, cv2.COLOR_RGB2GRAY)

    disparity = compute_disparity(
        used_left_gray,
        used_right_gray,
        num_disparities=num_disparities,
        block_size=block_size,
        uniqueness_ratio=uniqueness_ratio,
        speckle_window_size=speckle_window_size,
        speckle_range=speckle_range,
    )

    valid_disparity = disparity[np.isfinite(disparity)]
    if valid_disparity.size == 0:
        raise gr.Error("Disparity could not be computed correctly for this image pair.")

    point_cloud_figure = make_point_cloud_plot(
        used_left_rgb,
        disparity,
        focal_length_px,
        baseline_m,
        max_points=point_count,
    )
    depth_map = compute_depth_map(disparity, focal_length_px, baseline_m)
    depth_view = colorize_map(depth_map, invert=True)
    disparity_view = colorize_map(disparity, invert=False)

    summary = "\n".join(
        [
            "## Results",
            f"- RANSAC inliers: {ransac_inliers}",
            f"- {format_depth_summary(depth_map)}",
        ]
    )

    ply_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
    ply_file.close()
    ply_path = write_ply(
        ply_file.name,
        used_left_rgb,
        disparity,
        focal_length_px,
        baseline_m,
        max_points=point_count,
    )

    return (
        summary,
        point_cloud_figure,
        depth_view,
        disparity_view,
        matches_view,
        used_left_rgb,
        used_right_rgb,
        ply_path,
    )

with gr.Blocks(title="Noise Filtering Demo") as demo:
    with gr.Tab("App"):

        #gr.Markdown(
        #    """
        ### Test Stereo Image Sources

        #- [Middlebury Stereo Datasets](https://vision.middlebury.edu/stereo/data/)
        #- [KITTI Stereo 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
        #- [ETH3D Stereo Benchmark](https://www.eth3d.net/)
        #    """
        #)

        with gr.Row():
            left_image = gr.Image(label="Left Image", type="numpy", value=DEFAULT_LEFT_URL)
            right_image = gr.Image(label="Right Image", type="numpy", value=DEFAULT_RIGHT_URL)

        with gr.Accordion("Settings", open=True):
            with gr.Row():
                focal_length_px = gr.Number(label="Focal Length (Pixels)", value=1200.0, precision=3)
                baseline_m = gr.Number(label="Baseline (Meters)", value=0.10, precision=6)
                auto_resize = gr.Checkbox(label="Automatically Resize For Speed", value=True)

            rectification_mode = gr.Radio(
                label="Rectification Mode",
                choices=["Auto", "Already Rectified", "Estimate And Rectify"],
                value="Auto",
            )

            with gr.Row():
                auto_rectification_threshold_px = gr.Number(
                    label="Auto Rectification Threshold (Pixels)",
                    value=2.0,
                    precision=2,
                )
                point_count = gr.Number(label="Point Count", value=DEFAULT_POINT_COUNT, precision=0)

            with gr.Row():
                num_disparities = gr.Number(label="Num Disparities", value=256, precision=0)
                block_size = gr.Number(label="Block Size", value=3, precision=0)

            with gr.Row():
                uniqueness_ratio = gr.Number(label="Uniqueness Ratio", value=5, precision=0)
                speckle_window_size = gr.Number(label="Speckle Window Size", value=20, precision=0)
                speckle_range = gr.Number(label="Speckle Range", value=4, precision=0)

        run_button = gr.Button("Run Reconstruction")
        results_markdown = gr.Markdown()

        with gr.Tab("3D Point Cloud"):
            point_cloud_output = gr.Plot(label="Interactive Point Cloud")

        with gr.Tab("Depth And Disparity"):
            with gr.Row():
                depth_output = gr.Image(label="Depth Map")
                disparity_output = gr.Image(label="Disparity Map")

        with gr.Tab("Matches"):
            matches_output = gr.Image(label="Orb Matches And Ransac Inliers")

        with gr.Tab("Rectified Views"):
            with gr.Row():
                used_left_output = gr.Image(label="Used Left View")
                used_right_output = gr.Image(label="Used Right View")

        with gr.Tab("3D Export"):
            ply_output = gr.File(label="Point Cloud (.ply)")

        run_button.click(
            fn=run_stereo_demo,
            inputs=[
                left_image,
                right_image,
                focal_length_px,
                baseline_m,
                auto_resize,
                rectification_mode,
                auto_rectification_threshold_px,
                num_disparities,
                block_size,
                uniqueness_ratio,
                speckle_window_size,
                speckle_range,
                point_count,
            ],
            outputs=[
                results_markdown,
                point_cloud_output,
                depth_output,
                disparity_output,
                matches_output,
                used_left_output,
                used_right_output,
                ply_output,
            ],
        )

    with gr.Tab("Documentation FR"):
        with gr.Row():
            with gr.Column(scale=1):
                doc_fr_buttons = []
                for title in DOC_FR_TITLES:
                    btn = gr.Button(title)
                    doc_fr_buttons.append((btn, title))

            with gr.Column(scale=3):
                doc_fr_view = gr.Markdown(
                    value=load_doc_fr_section(DOC_FR_TITLES[0]),
                    latex_delimiters=LATEX_DELIMITERS
                )

        for btn, title in doc_fr_buttons:
            btn.click(
                lambda t=title: load_doc_fr_section(t),
                inputs=None,
                outputs=doc_fr_view,
            )

    with gr.Tab("Documentation EN"):
        with gr.Row():
            with gr.Column(scale=1):
                doc_en_buttons = []
                for title in DOC_EN_TITLES:
                    btn = gr.Button(title)
                    doc_en_buttons.append((btn, title))

            with gr.Column(scale=3):
                doc_en_view = gr.Markdown(
                    value=load_doc_en_section(DOC_EN_TITLES[0]),
                    latex_delimiters=LATEX_DELIMITERS
                )

        for btn, title in doc_en_buttons:
            btn.click(
                lambda t=title: load_doc_en_section(t),
                inputs=None,
                outputs=doc_en_view,
            )


if __name__ == "__main__":
    demo.launch()