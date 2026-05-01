## Table of Contents

2. [The Stereo Vision Geometry](#1-the-stereo-vision-geometry)
   - 2.1 [Projective Camera Model](#11-projective-camera-model)
   - 2.2 [The Stereo Baseline and Disparity](#12-the-stereo-baseline-and-disparity)
3. [Feature Detection and Matching with ORB](#2-feature-detection-and-matching-with-orb)
   - 3.1 [ORB: Oriented FAST and Rotated BRIEF](#21-orb-oriented-fast-and-rotated-brief)
   - 3.2 [Scale Space and Multi-scale Detection](#22-scale-space-and-multi-scale-detection)
   - 3.3 [Matching by Hamming Distance](#23-matching-by-hamming-distance)
4. [Epipolar Geometry and the Fundamental Matrix](#3-epipolar-geometry-and-the-fundamental-matrix)
   - 4.1 [Epipolar Constraint](#31-epipolar-constraint)
   - 4.2 [8-Point Algorithm (Normalized)](#32-8-point-algorithm-normalized)
   - 4.3 [RANSAC for Robust Estimation](#33-ransac-for-robust-estimation)
5. [Uncalibrated Stereo Rectification](#4-uncalibrated-stereo-rectification)
   - 5.1 [Purpose of Rectification](#41-purpose-of-rectification)
   - 5.2 [Hartley's Algorithm](#42-hartleys-algorithm)
   - 5.3 [Automatic Rectification Detection](#43-automatic-rectification-detection)
6. [Semi-Global Block Matching (SGBM)](#5-semi-global-block-matching-sgbm)
   - 6.1 [The Disparity Estimation Problem](#51-the-disparity-estimation-problem)
   - 6.2 [Block Matching Cost](#52-block-matching-cost)
   - 6.3 [Semi-Global Matching Energy](#53-semi-global-matching-energy)
   - 6.4 [Dynamic Programming Along Paths](#54-dynamic-programming-along-paths)
   - 6.5 [Post-Processing: Uniqueness and Speckle Filtering](#55-post-processing-uniqueness-and-speckle-filtering)
7. [Depth Map and 3D Reconstruction](#6-depth-map-and-3d-reconstruction)
   - 7.1 [From Disparity to Depth](#61-from-disparity-to-depth)
   - 7.2 [3D Back-Projection](#62-3d-back-projection)
   - 7.3 [Outlier Removal and Subsampling](#63-outlier-removal-and-subsampling)
8. [Pipeline Summary](#7-pipeline-summary)
9. [Parameters Reference](#8-parameters-reference)
10. [References](#9-references)

## 1. The Stereo Vision Geometry

### 1.1 Projective Camera Model

A camera maps a 3D point $\mathbf{P} = (X, Y, Z)^\top$ in the world to a 2D pixel $\mathbf{p} = (u, v)^\top$ through the **perspective projection** (pinhole model):

$$u = f \frac{X}{Z} + c_x, \qquad v = f \frac{Y}{Z} + c_y$$

where $f$ is the focal length (in pixels) and $(c_x, c_y)$ is the principal point (usually the image center). In homogeneous coordinates this is written compactly as:

$$\lambda \begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = \mathbf{K} \begin{pmatrix} X \\ Y \\ Z \end{pmatrix}, \qquad \mathbf{K} = \begin{pmatrix} f & 0 & c_x \\ 0 & f & c_y \\ 0 & 0 & 1 \end{pmatrix}$$

The matrix $\mathbf{K}$ is the **intrinsic calibration matrix**. The scalar $\lambda = Z$ is the perspective depth, introduced to make the projection linear in homogeneous coordinates.

### 1.2 The Stereo Baseline and Disparity

In a **rectified stereo rig**, the two cameras share the same focal length $f$ and their optical axes are parallel. The right camera is displaced by a **baseline** $b$ along the $X$-axis. Under these conditions, a 3D point $\mathbf{P} = (X, Y, Z)^\top$ projects to pixel $(u_L, v)$ in the left image and $(u_R, v)$ in the right image — critically, the $v$-coordinates are **identical** after rectification.

The **disparity** is defined as:

$$d = u_L - u_R$$

Combining the projection equations of both cameras gives the fundamental **depth-disparity relation**:

$$Z = \frac{f \cdot b}{d}$$

This inverse relationship is central to stereo vision: large disparities correspond to nearby objects, small disparities to distant ones. Once $d$ is known for every pixel, the full 3D position follows by back-projection:

$$X = \frac{(u_L - c_x) \cdot Z}{f}, \qquad Y = \frac{(v - c_y) \cdot Z}{f}$$

---

## 2. Feature Detection and Matching with ORB

Before computing a dense disparity map, the pipeline must either verify that the image pair is already rectified, or estimate the rectifying transforms. Both tasks require finding **point correspondences** — pairs of pixels $(\mathbf{p}_L, \mathbf{p}_R)$ that depict the same 3D scene point — between the two images.

### 2.1 ORB: Oriented FAST and Rotated BRIEF

ORB (Rublee et al., 2011) is a binary feature descriptor designed to be fast, memory-efficient, and robust to viewpoint and illumination changes. It combines two building blocks:

**FAST keypoint detector.** FAST (Features from Accelerated Segment Test) declares a pixel $p$ a corner if there exists a contiguous arc of at least 9 pixels (out of 16 on a Bresenham circle of radius 3) that are all either brighter than $I_p + t$ or darker than $I_p - t$, where $t$ is a contrast threshold. FAST is $O(1)$ per pixel thanks to a learned decision tree that early-exits the arc test after checking only 3–4 pixels on average.

ORB adds **orientation** to each FAST keypoint using the **intensity centroid** method. The image moments of the patch $\mathcal{P}$ centered at the keypoint are:

$$m_{pq} = \sum_{(x,y) \in \mathcal{P}} x^p \, y^q \, I(x, y)$$

The centroid is $\mathbf{C} = (m_{10}/m_{00},\; m_{01}/m_{00})$, and the keypoint orientation is the angle of the vector from the geometric center to the intensity centroid:

$$\theta = \text{atan2}(m_{01},\; m_{10})$$

**BRIEF descriptor.** BRIEF (Binary Robust Independent Elementary Features) encodes a patch as a binary string of length $n$ (typically $n = 256$ bits). Each bit is the result of an intensity comparison between two patch locations $(x_i, y_i)$ and $(x_i', y_i')$ sampled according to a learned pattern:

$$\tau(I;\, x, x') = \begin{cases} 1 & \text{if } I(x) < I(x') \\ 0 & \text{otherwise} \end{cases}$$

$$\mathbf{f}_n(I) = \sum_{1 \le i \le n} 2^{i-1} \, \tau(I;\, x_i,\, x_i')$$

ORB **rotates** the sampling pairs by $\theta$ before computing BRIEF, making the descriptor rotation-invariant (rBRIEF). It also applies a greedy de-correlation step to the sampling pattern to maximize per-bit variance and minimize inter-bit correlation, ensuring maximum information content per bit.

### 2.2 Scale Space and Multi-scale Detection

ORB builds an **image pyramid** with `nlevels` levels and a scale factor `scaleFactor` $s > 1$ between consecutive levels. The image at level $k$ is a downsampled version of the original at resolution $s^{-k}$. Features detected at level $k$ correspond to structures of characteristic spatial size $\sim s^k$ pixels in the original image. The pyramid enables detection of features at multiple scales using a fixed-size FAST kernel, providing **scale invariance** without changing the detector.

The `nfeatures` budget is distributed across pyramid levels, with more features allocated to finer levels.

### 2.3 Matching by Hamming Distance

Since ORB descriptors are binary vectors in $\{0,1\}^n$, the natural dissimilarity measure between two descriptors $\mathbf{d}_1$ and $\mathbf{d}_2$ is the **Hamming distance**:

$$d_H(\mathbf{d}_1, \mathbf{d}_2) = \|\mathbf{d}_1 \oplus \mathbf{d}_2\|_1 = \text{popcount}(\mathbf{d}_1 \oplus \mathbf{d}_2)$$

where $\oplus$ denotes bitwise XOR and popcount counts set bits. For 256-bit descriptors this reduces to 4 XOR + 4 popcount operations on 64-bit hardware, making it extremely fast.

The **brute-force matcher with cross-check** retains only match pairs $(i, j)$ where feature $i$ in the left image has feature $j$ as its nearest neighbor in the right image **and** feature $j$ has feature $i$ as its nearest neighbor in the left image. This mutual consistency constraint eliminates a large fraction of false matches before any geometric verification.

---

## 3. Epipolar Geometry and the Fundamental Matrix

### 3.1 Epipolar Constraint

Given a point $\mathbf{p}_L = (u_L, v_L)^\top$ in the left image, the corresponding point $\mathbf{p}_R$ in the right image does not lie anywhere on the image plane — it must lie on a specific line called the **epipolar line**, which is the projection of the optical ray through $\mathbf{p}_L$ onto the right image. This constraint is encoded by the **fundamental matrix** $\mathbf{F} \in \mathbb{R}^{3 \times 3}$:

$$\tilde{\mathbf{p}}_R^\top \, \mathbf{F} \, \tilde{\mathbf{p}}_L = 0$$

where $\tilde{\mathbf{p}} = (u, v, 1)^\top$ denotes homogeneous pixel coordinates. The matrix $\mathbf{F}$ is rank-2 and has 7 degrees of freedom (9 entries, up to an overall scale factor, minus the rank-2 constraint).

The line $\ell_R = \mathbf{F} \tilde{\mathbf{p}}_L$ is the epipolar line in the right image corresponding to $\mathbf{p}_L$. Symmetrically, $\ell_L = \mathbf{F}^\top \tilde{\mathbf{p}}_R$ is the epipolar line in the left image corresponding to $\mathbf{p}_R$. After **stereo rectification**, all epipolar lines become horizontal rows, reducing the correspondence search from a 2D problem to a 1D scan.

### 3.2 8-Point Algorithm (Normalized)

The fundamental matrix is estimated from point correspondences. Expanding the epipolar constraint with $\tilde{\mathbf{p}}_L = (x, y, 1)^\top$ and $\tilde{\mathbf{p}}_R = (x', y', 1)^\top$ yields one linear equation in the 9 entries of $\mathbf{F}$:

$$x'x\,F_{11} + x'y\,F_{12} + x'\,F_{13} + y'x\,F_{21} + y'y\,F_{22} + y'\,F_{23} + x\,F_{31} + y\,F_{32} + F_{33} = 0$$

Stacking $n \geq 8$ such equations gives the homogeneous linear system $\mathbf{A} \mathbf{f} = \mathbf{0}$, where $\mathbf{f} = \text{vec}(\mathbf{F}) \in \mathbb{R}^9$. The solution is the right singular vector of $\mathbf{A}$ corresponding to its smallest singular value, found via SVD. The rank-2 constraint is then enforced by a second SVD: writing $\mathbf{F} = \mathbf{U} \,\text{diag}(\sigma_1, \sigma_2, \sigma_3)\, \mathbf{V}^\top$ with $\sigma_1 \geq \sigma_2 \geq \sigma_3$, the nearest rank-2 matrix is:

$$\hat{\mathbf{F}} = \mathbf{U} \,\text{diag}(\sigma_1, \sigma_2, 0)\, \mathbf{V}^\top$$

**Normalization** (Hartley, 1997) is critical for numerical stability. Before solving, each image's point set is independently transformed: translated to its centroid, then isotropically scaled so that the RMS distance of points from the centroid equals $\sqrt{2}$. The normalization transforms $\mathbf{T}_L$ and $\mathbf{T}_R$ are applied to the data, the fundamental matrix is solved in normalized coordinates, then de-normalized: $\mathbf{F} = \mathbf{T}_R^\top \hat{\mathbf{F}}_{\text{norm}} \mathbf{T}_L$.

### 3.3 RANSAC for Robust Estimation

Real matches contain **outliers** — incorrectly matched pairs due to repetitive textures, occlusion, or descriptor ambiguity. Ordinary least squares is highly sensitive to outliers. **RANSAC** (Random Sample Consensus, Fischler & Bolles 1981) addresses this with the following iterative scheme:

1. Randomly sample a **minimal set** of correspondences (8 for $\mathbf{F}$).
2. Fit the model — estimate $\mathbf{F}$ using the normalized 8-point algorithm.
3. Count **inliers**: all pairs $(\mathbf{p}_L, \mathbf{p}_R)$ for which the **Sampson distance** is below a threshold $\varepsilon$:

$$d_S(\mathbf{p}_L, \mathbf{p}_R, \mathbf{F}) = \frac{(\tilde{\mathbf{p}}_R^\top \mathbf{F} \tilde{\mathbf{p}}_L)^2}{(\mathbf{F}\tilde{\mathbf{p}}_L)_1^2 + (\mathbf{F}\tilde{\mathbf{p}}_L)_2^2 + (\mathbf{F}^\top\tilde{\mathbf{p}}_R)_1^2 + (\mathbf{F}^\top\tilde{\mathbf{p}}_R)_2^2}$$

The Sampson distance is a first-order approximation to the reprojection error that is cheaper to compute than the exact geometric distance. It measures how far the point pair is from satisfying the epipolar constraint, normalized by the local gradient of that constraint.

4. Repeat for $N$ iterations; keep the hypothesis with the most inliers.
5. Re-estimate $\mathbf{F}$ from **all** inliers of the best hypothesis using the normalized 8-point algorithm.

The number of iterations $N$ required to guarantee, with probability $p$, that at least one minimal sample contains no outliers, given an inlier rate $\rho$, is:

$$N = \frac{\log(1 - p)}{\log(1 - \rho^8)}$$

For $p = 0.99$ and a typical inlier rate $\rho = 0.5$, this gives $N \approx 1177$ iterations. This app uses threshold $\varepsilon = 1.0$ pixel (Sampson distance) and $p = 0.99$.

---

## 4. Uncalibrated Stereo Rectification

### 4.1 Purpose of Rectification

Dense disparity estimation is only tractable under the assumption that corresponding pixels lie on the same **horizontal scanline** in both images. This requires that all epipolar lines be horizontal, which is the definition of a **rectified** stereo pair. Rectification is implemented as a pair of planar homographies $\mathbf{H}_L, \mathbf{H}_R \in \mathbb{R}^{3 \times 3}$ that warp the two images such that $\ell_R = \mathbf{F}\tilde{\mathbf{p}}_L$ corresponds to a row of constant $v$.

When the cameras' intrinsic parameters are unknown — the **uncalibrated** case — rectification can still be performed from the fundamental matrix $\mathbf{F}$ alone, using the algorithm of Hartley (1999).

### 4.2 Hartley's Algorithm

The **epipoles** $\mathbf{e}_L$ and $\mathbf{e}_R$ (the projections of each camera center onto the other image) are the null vectors of $\mathbf{F}^\top$ and $\mathbf{F}$ respectively:

$$\mathbf{F}\,\mathbf{e}_L = \mathbf{0}, \qquad \mathbf{F}^\top \mathbf{e}_R = \mathbf{0}$$

Rectification requires mapping the epipoles to the point at infinity along the $x$-axis, $(1, 0, 0)^\top$ in homogeneous coordinates, which makes all epipolar lines horizontal. The construction proceeds in three steps:

**Step 1 — Translation.** Both images are translated so the image center maps to the origin.

**Step 2 — Right homography $\mathbf{H}_R$.** A rotation $\mathbf{R}$ is chosen to send $\mathbf{e}_R$ onto the $x$-axis (i.e., to $(f_e, 0, 1)^\top$ for some $f_e$). Then a projective transform $\mathbf{G}$ maps this finite point to the ideal point $(1, 0, 0)^\top$:

$$\mathbf{G} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ -1/f_e & 0 & 1 \end{pmatrix}$$

giving $\mathbf{H}_R = \mathbf{G}\mathbf{R}$ (up to re-centering).

**Step 3 — Left homography $\mathbf{H}_L$.** Given $\mathbf{H}_R$, the goal is to find $\mathbf{H}_L$ such that the vertical displacement between corresponding warped points is minimized. The matching homography is decomposed as $\mathbf{H}_L = \mathbf{H}_A \mathbf{H}_R \mathbf{M}$, where $\mathbf{M}$ is a transformation derived from $\mathbf{F}$ and $\mathbf{H}_A$ is an affine correction. The coefficients of $\mathbf{H}_A$ are found by **linear least squares** over all inlier correspondences, minimizing the sum of squared vertical residuals after warping by $\mathbf{H}_R$ on the right and $\mathbf{H}_L$ on the left.

After warping with $(\mathbf{H}_L, \mathbf{H}_R)$, any corresponding pair satisfies $v_L' \approx v_R'$, enabling pure horizontal 1D search for disparity.

### 4.3 Automatic Rectification Detection

To decide whether to apply rectification, the app evaluates the **median absolute vertical residual** over the RANSAC inlier correspondences:

$$r_v = \text{median}\bigl(\{|v_{L,i} - v_{R,i}|\}_{i=1}^{N_{\text{in}}}\bigr)$$

If $r_v \leq \delta$ (user-defined, default $\delta = 2$ pixels), the pair is considered already rectified and no homography is applied. The use of the median (rather than the mean) makes this test robust to the few remaining geometric outliers even among RANSAC inliers.

---

## 5. Semi-Global Block Matching (SGBM)

### 5.1 The Disparity Estimation Problem

Given two rectified grayscale images $I_L(u, v)$ and $I_R(u, v)$ of size $W \times H$, the goal is to assign a disparity value $d(u, v) \in \{0, 1, \ldots, D_{\max}\}$ to every pixel of the left image, such that $I_L(u, v) \approx I_R(u - d, v)$.

This is a **dense labeling problem** on a $W \times H \times D_{\max}$ search space. Naive winner-takes-all matching (choose independently for each pixel the $d$ that minimizes a local cost) produces noisy maps with many errors on textureless regions and near occlusion boundaries. Semi-Global Matching (SGBM, Hirschmüller 2005, 2008) solves a regularized version of this problem efficiently.

### 5.2 Block Matching Cost

For each pixel $(u, v)$ and candidate disparity $d$, the **local matching cost** is computed over a square block of side $b_s$ centered at $(u, v)$:

$$C_{\text{BM}}(u, v, d) = \sum_{(\Delta u, \Delta v) \in \mathcal{B}} \bigl|I_L(u + \Delta u,\; v + \Delta v) - I_R(u + \Delta u - d,\; v + \Delta v)\bigr|$$

where $\mathcal{B} = \{-\lfloor b_s/2 \rfloor, \ldots, \lfloor b_s/2 \rfloor\}^2$. This is the **Sum of Absolute Differences (SAD)** over the matching window. The block size $b_s$ controls a fundamental trade-off: small blocks preserve sharp depth boundaries but are sensitive to noise; large blocks are more stable but over-smooth disparity transitions near object edges.

In practice, OpenCV's SGBM pre-filters each image with a truncated horizontal gradient (clamped to `preFilterCap = 31`) before computing SAD, improving robustness to global illumination differences between the two cameras.

### 5.3 Semi-Global Matching Energy

Rather than minimizing $C_{\text{BM}}$ independently at each pixel, SGM minimizes a global energy over the entire disparity map $\mathbf{d}$:

$$E(\mathbf{d}) = \sum_{(u,v)} C_{\text{BM}}(u,v,d(u,v)) \;+\; \sum_{(u,v)} \sum_{(u',v') \in \mathcal{N}(u,v)} P\bigl(d(u,v),\, d(u',v')\bigr)$$

where $\mathcal{N}(u,v)$ is the set of 4- or 8-connected neighbors, and the pairwise penalty $P$ is:

$$P(d, d') = \begin{cases} 0 & \text{if } d = d' \\ P_1 & \text{if } |d - d'| = 1 \\ P_2 & \text{if } |d - d'| > 1 \end{cases}$$

$P_1$ penalizes small disparity increments, modeling gently sloping surfaces. $P_2 > P_1$ penalizes large jumps, allowing sharp depth discontinuities at object boundaries but discouraging them in smooth regions. In this app: $P_1 = 8 b_s^2$ and $P_2 = 32 b_s^2$, so the ratio $P_2/P_1 = 4$ is fixed regardless of block size.

Exact minimization of $E(\mathbf{d})$ over a 2D grid is NP-hard (it is a submodular MRF, but with the full 8-connected graph and $D_{\max}$ labels, exact inference is intractable in real-time). SGM approximates it via the following scheme.

### 5.4 Dynamic Programming Along Paths

SGM decomposes the 2D problem into a set of **1D dynamic programs** along $r$ directions (straight-line paths through the image). For a path going in direction $\mathbf{r} = (r_u, r_v)$, the **path cost** is computed recursively from one end of the path to the other:

$$L_\mathbf{r}(u,v,d) = C_{\text{BM}}(u,v,d) + \min \begin{cases} L_\mathbf{r}(u - r_u,\, v - r_v,\; d) \\ L_\mathbf{r}(u - r_u,\, v - r_v,\; d - 1) + P_1 \\ L_\mathbf{r}(u - r_u,\, v - r_v,\; d + 1) + P_1 \\ \displaystyle\min_{d''} L_\mathbf{r}(u - r_u,\, v - r_v,\; d'') + P_2 \end{cases} \;-\; \min_{d''} L_\mathbf{r}(u - r_u,\, v - r_v,\; d'')$$

The subtraction of the global minimum at the previous step prevents numerical overflow along long paths without changing the argmin. The **aggregated cost** sums over all $|\mathcal{R}|$ directions:

$$S(u,v,d) = \sum_{\mathbf{r} \in \mathcal{R}} L_\mathbf{r}(u,v,d)$$

The final disparity is selected as:

$$\hat{d}(u,v) = \arg\min_{d} \, S(u,v,d)$$

This app uses OpenCV's `STEREO_SGBM_MODE_SGBM_3WAY`, which aggregates over 3 directions, offering a good speed/quality trade-off. The raw integer output of OpenCV is stored with **4 fractional bits** (fixed-point representation), so the true disparity in pixels is obtained by dividing by 16.0.

### 5.5 Post-Processing: Uniqueness and Speckle Filtering

**Uniqueness ratio.** A disparity assignment $\hat{d}(u,v)$ is accepted only if the cost at the second-best disparity $d^*_2$ (outside a $\pm 1$ neighborhood of $\hat{d}$) exceeds the best cost by at least a relative margin:

$$S(u,v,d^*_2) > S(u,v,\hat{d}) \cdot \left(1 + \frac{\text{ratio}}{100}\right)$$

This rejects pixels whose cost volume is flat — typically on textureless regions (walls, sky) where the matching is ambiguous. Rejected pixels are marked as invalid.

**Speckle filter.** After disparity assignment, connected components of the disparity map are analyzed. Any component satisfying **both** conditions — (a) its area is smaller than `speckleWindowSize` pixels and (b) the range of disparity values within it exceeds `speckleRange` — is considered a spurious noise blob and marked as invalid ($d \to \text{NaN}$). This removes isolated artifacts without affecting genuine depth discontinuities whose connected regions are large.

---

## 6. Depth Map and 3D Reconstruction

### 6.1 From Disparity to Depth

Given the fundamental stereo relation $Z = f \cdot b / d$, the **depth map** is computed pixel-wise by applying this formula to the validated disparity map. Pixels where $d \leq 0$ or $d = \text{NaN}$ (invalid from speckle filtering or uniqueness rejection) yield $Z = \text{NaN}$ and are excluded from all subsequent processing.

For visualization, both the disparity and depth maps are rendered with the **Turbo colormap** after a robust percentile normalization: the display range is clipped to the $[2\%, 98\%]$ percentile interval of valid values, avoiding saturation from extreme outliers. The depth map additionally inverts the intensity mapping so that near objects (small $Z$) appear in warm colors (red/yellow) and far objects in cool colors (blue/violet), matching common depth visualization conventions.

### 6.2 3D Back-Projection

Each valid pixel $(u, v)$ with disparity $d$ is lifted to 3D coordinates by inverting the projection equations of both cameras simultaneously. Using the principal point approximated as the image center $(c_x, c_y) = (W/2, H/2)$:

$$\begin{pmatrix} X \\ Y \\ Z \end{pmatrix} = \begin{pmatrix} (u - c_x) \cdot b \;/\; d \\ (v - c_y) \cdot b \;/\; d \\ f \cdot b \;/\; d \end{pmatrix}$$

Each 3D point inherits the RGB color of the corresponding left-image pixel. The resulting colored point cloud is a discrete sampling of the visible scene surfaces.

### 6.3 Outlier Removal and Subsampling

Despite the speckle filter, the raw point cloud may still contain extreme depth outliers due to half-occluded pixels or boundary effects. A **percentile depth clamp** along the $Z$-axis (2nd–98th percentile) removes the most extreme outliers before visualization or export, ensuring a consistent view scale.

If the number of valid 3D points exceeds the user-defined `point_count`, a **uniform index subsampling** strategy (selecting evenly spaced indices from the sorted point list) reduces the cloud to the target size while preserving the global spatial distribution of the scene. This avoids the density bias that random subsampling can introduce near high-texture regions.

---

## 7. Pipeline Summary

The full reconstruction pipeline, from raw image pair to 3D point cloud, proceeds as follows:

$$\text{Image pair} \;\longrightarrow\; \text{ORB matching} \;\longrightarrow\; \text{RANSAC + } \mathbf{F} \;\longrightarrow\; \text{Rectification} \;\longrightarrow\; \text{SGBM} \;\longrightarrow\; \text{Depth} \;\longrightarrow\; \text{3D back-projection}$$

Each stage has explicit failure conditions that generate informative errors (insufficient features, too few RANSAC inliers, degenerate image dimensions for SGBM), ensuring the app degrades gracefully on challenging input rather than silently producing wrong results.

---

## 8. Parameters Reference

| Parameter | Symbol | Role | Typical range |
|---|---|---|---|
| **Focal Length** (px) | $f$ | Camera intrinsic; scales $Z$ linearly | 500 – 4000 px |
| **Baseline** (m) | $b$ | Camera separation; scales $Z$ linearly | 0.05 – 1.0 m |
| **Rectification threshold** (px) | $\delta$ | Max median vertical residual to skip rectification | 1 – 5 px |
| **Number of Disparities** | $D_{\max}$ | Search range $[0, D_{\max}]$; multiple of 16 | 64 – 512 |
| **Block Size** | $b_s$ | SAD window side length; odd integer | 3 – 15 |
| **Uniqueness Ratio** | — | Cost margin for acceptance (%) | 5 – 15 |
| **Speckle Window Size** | — | Min connected component area to keep | 0 – 300 |
| **Speckle Range** | — | Max disparity variation within a kept component | 1 – 16 |

**Note on $Z$ scaling.** The reconstructed depth is proportional to $f \cdot b$. If the true focal length and baseline are unknown, relative depth structure is still valid, but metric distances will be off by a constant scale factor. For benchmark datasets such as Middlebury, KITTI or ETH3D, ground-truth calibration values are provided with the data.

---

## 9. References

- **Hartley, R. & Zisserman, A.**, *Multiple View Geometry in Computer Vision*, Cambridge University Press, 2nd ed., 2004.
- **Hirschmüller, H.**, *Accurate and efficient stereo processing by semi-global matching and mutual information*, CVPR, 2005.
- **Hirschmüller, H.**, *Stereo processing by semiglobal matching and mutual information*, IEEE Transactions on Pattern Analysis and Machine Intelligence, 30(2):328–341, 2008.
- **Rublee, E., Rabaud, V., Konolige, K. & Bradski, G.**, *ORB: An efficient alternative to SIFT or SURF*, ICCV, pp. 2564–2571, 2011.
- **Fischler, M. A. & Bolles, R. C.**, *Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography*, Communications of the ACM, 24(6):381–395, 1981.
- **Hartley, R.**, *In defense of the eight-point algorithm*, IEEE Transactions on Pattern Analysis and Machine Intelligence, 19(6):580–593, 1997.
- **Calonder, M., Lepetit, V., Strecha, C. & Fua, P.**, *BRIEF: Binary Robust Independent Elementary Features*, ECCV, 2010.
