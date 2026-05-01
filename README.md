---
title: Stereovision Demo
emoji: 👀
colorFrom: purple
colorTo: gray
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: mit
short_description: Stereo vision demo
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Stereovision Demo

This repository contains an interactive stereo vision mini app based on classical computer vision methods.

The app reconstructs depth information from a pair of stereo images. It detects and matches visual features, estimates epipolar geometry, rectifies the stereo pair when needed, computes a dense disparity map with Semi-Global Block Matching, converts disparity into depth, and generates a colored 3D point cloud.

A Streamlit deployment is available here:

https://stereo-vision.streamlit.app/

## Main features

- Load a pair of stereo images or use the default Middlebury stereo example.
- Automatically resize large images for faster processing.
- Detect and match ORB features between the left and right images.
- Estimate the fundamental matrix with RANSAC.
- Display feature matches and inlier correspondences.
- Detect whether the image pair is already rectified.
- Apply uncalibrated stereo rectification when needed.
- Compute a dense disparity map using OpenCV StereoSGBM.
- Convert disparity into an approximate depth map.
- Generate an interactive colored 3D point cloud.
- Export the reconstructed point cloud as a PLY file.
- Read the English and French documentation tabs.

## Method overview

The app follows a classical stereo reconstruction pipeline:

```text
Stereo image pair
        |
        v
ORB feature detection and matching
        |
        v
RANSAC estimation of the fundamental matrix
        |
        v
Rectification check or uncalibrated rectification
        |
        v
Dense disparity estimation with StereoSGBM
        |
        v
Depth map computation
        |
        v
3D back-projection and colored point-cloud generation
```

Each step is exposed through visual outputs so that the user can inspect the intermediate results instead of treating stereo vision as a black box.

## Stereo vision geometry

A stereo system estimates depth by comparing two images of the same scene taken from different viewpoints.

After rectification, corresponding points lie on the same horizontal image row. The horizontal shift between corresponding pixels is called disparity.

The core relationship is:

```text
Z = f * b / d
```

where `Z` is the depth, `f` is the focal length in pixels, `b` is the stereo baseline, and `d` is the disparity.

Large disparities correspond to nearby objects. Small disparities correspond to distant objects.

If the true focal length and baseline are unknown, the app still produces meaningful relative depth, but the metric scale may not be physically accurate.

## ORB feature matching

The app uses ORB, which combines FAST keypoint detection with BRIEF binary descriptors.

ORB is fast, lightweight, and well suited for an online educational demo. Feature descriptors are matched using Hamming distance, and cross-check matching keeps only mutually consistent correspondences.

These correspondences are used to estimate the epipolar geometry of the stereo pair.

## Fundamental matrix and RANSAC

The fundamental matrix encodes the epipolar constraint between the two images.

Because raw feature matches may contain outliers, the app uses RANSAC to estimate the fundamental matrix robustly. RANSAC keeps only geometrically consistent matches, called inliers, and discards inconsistent correspondences.

The number of inliers gives a useful indication of whether the stereo pair contains enough reliable visual information for reconstruction.

## Rectification

Dense disparity estimation is easier after stereo rectification, because corresponding pixels are expected to lie on the same horizontal scanline.

The app evaluates the median vertical residual between matched inlier points. If the residual is below the selected threshold, the pair is treated as already rectified. Otherwise, the app applies uncalibrated rectification using the estimated fundamental matrix.

This makes the app usable both for already-rectified stereo datasets and for more general image pairs.

## Dense disparity with StereoSGBM

The dense disparity map is computed with OpenCV StereoSGBM.

StereoSGBM searches for the best horizontal displacement for each pixel and regularises the result to avoid excessive noise. It is more robust than simple local block matching, especially in weakly textured regions.

The interface exposes important SGBM parameters such as number of disparities, block size, uniqueness ratio, speckle window size, and speckle range.

These parameters control the trade-off between detail preservation, smoothness, robustness, and computation time.

## Depth map and 3D point cloud

Once a valid disparity map is obtained, the app converts disparity into approximate depth using the stereo depth relation.

Invalid or non-positive disparities are ignored. Extreme depth outliers are removed with percentile-based filtering. The remaining valid points are back-projected into 3D and colored with the corresponding RGB values from the left image.

The resulting point cloud can be visualised interactively and exported as a PLY file.

## Outputs

The app produces several visual and numerical outputs:

- feature matches,
- rectified left image,
- rectified right image,
- disparity map,
- depth map,
- interactive 3D point cloud,
- PLY point-cloud export,
- reconstruction summary.

These outputs help diagnose whether a reconstruction is geometrically meaningful.

## Repository structure

```text
.
├── app.py                 # Gradio / Hugging Face Space entry point
├── app_sl.py              # Streamlit version of the app
├── documentation_en.md    # English documentation
├── documentation_fr.md    # French documentation
├── requirements.txt       # Python dependencies
├── LICENSE.txt            # License file
└── README.md              # Repository and Hugging Face Space description
```

## Installation

Clone the repository:

```bash
git clone https://github.com/trungtin-dinh/stereovision_demo.git
cd stereovision_demo
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

If needed, install the main dependencies manually:

```bash
pip install gradio streamlit numpy opencv-python-headless plotly
```

## Run the Gradio app

```bash
python app.py
```

The local interface will usually be available at:

```text
http://127.0.0.1:7860
```

## Run the Streamlit app

```bash
streamlit run app_sl.py
```

The local interface will usually be available at:

```text
http://localhost:8501
```

## Hugging Face Space notes

The YAML block at the top of this README is used by Hugging Face Spaces.

The current metadata launches the Gradio version:

```yaml
sdk: gradio
app_file: app.py
```

If you want Hugging Face to launch the Streamlit version instead, update the metadata to:

```yaml
sdk: streamlit
app_file: app_sl.py
```

In that case, make sure `streamlit` is included in `requirements.txt`.

## Documentation

The repository includes two Markdown documentation files:

- `documentation_en.md` for the English documentation.
- `documentation_fr.md` for the French documentation.

These files explain stereo vision geometry, projective camera models, baseline and disparity, ORB feature detection, Hamming-distance matching, epipolar geometry, the fundamental matrix, RANSAC, uncalibrated rectification, Semi-Global Block Matching, depth-map computation, 3D back-projection, and point-cloud export.

## Notes and limitations

This app is intended as an educational demonstration of classical stereo vision.

The quality of the reconstruction depends strongly on the input image pair. Good results require sufficient texture, enough reliable feature matches, reasonable viewpoint overlap, and a stereo geometry compatible with rectification.

The depth map is only metrically meaningful when the focal length and stereo baseline are known. Otherwise, the output should be interpreted mainly as relative depth.

## License

This project is released under the MIT License.

## Author

Developed by Trung-Tin Dinh as part of a portfolio of interactive signal, audio, image, and computer vision mini apps.
