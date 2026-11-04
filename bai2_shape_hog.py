import os
import sys
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SHAPE_CLASSES = ["circle", "square", "triangle"]
SHAPE_FEATURE_COLS = ["Area_A", "Perimeter_P", "Width_W", "Height_H", "AspectRatio_AR", "Circularity_C", "EdgeDensity_ED"]


def _draw_single_shape(
    cls_name: str,
    rng: np.random.RandomState,
    img_size: int = 128,
    hard_case: bool = False,
) -> np.ndarray:
    """Render a synthetic image of circle, square, or triangle with random pose, scale, and contrast."""
    light_bg = bool(rng.randint(0, 2))
    if light_bg:
        bg_val = int(rng.randint(205, 248))
        fg_val = int(rng.randint(20, 75))
    else:
        bg_val = int(rng.randint(15, 60))
        fg_val = int(rng.randint(185, 245))
    #Tao nhung vat co do tuong phan nen ngau nhien 50% se co do sang (205-248), do toi (20,75) 50% con lai se co do sang tu (185 -245), do toi (28 -75)
    img = np.full((img_size, img_size, 3), bg_val, dtype=np.uint8)
    cx = int(rng.randint(44, img_size - 44))#random goc quay va kich thuoc cua cac hinh the
    cy = int(rng.randint(44, img_size - 44))
    radius = int(rng.randint(20, 40))
    angle_deg = float(rng.uniform(0, 360))
    angle_rad = math.radians(angle_deg)
    color = (fg_val, fg_val, fg_val)

    if cls_name == "circle":
        if hard_case:
            # Slightly polygonal / compressed ellipse-like circle
            ax1 = radius
            ax2 = max(12, int(radius * rng.uniform(0.72, 0.85)))
            cv2.ellipse(img, (cx, cy), (ax1, ax2), angle_deg, 0, 360, color, -1)
        else:
            ax1 = radius
            ax2 = max(15, int(radius * rng.uniform(0.92, 1.08)))
            cv2.ellipse(img, (cx, cy), (ax1, ax2), angle_deg, 0, 360, color, -1)
    #Cach ve hinh vuong (su dung ma tran)
    elif cls_name == "square":
        sx = radius * rng.uniform(0.9, 1.1)#Tinh nua chieu rong x
        sy = radius * (rng.uniform(0.75, 1.25) if hard_case else rng.uniform(0.9, 1.1))#Tinh nua chieu cao y
        corners = np.array([[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy]], dtype=np.float64)#suy ra 4 dinh toa do tu x va y
        rot_m = np.array(
            [[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad), math.cos(angle_rad)]]
        ) #su dung ma tran xoay tinh theo angle_rad
        pts = (corners @ rot_m.T) + np.array([cx, cy]) #Nhan ma tran de xoay toan bo 4 dinh quanh goc toa do roi cong them (cx,cy)
        pts_i = np.round(pts).astype(np.int32) #Lam tron toa do neu can thiet
        cv2.fillPoly(img, [pts_i], color)#fill mau vao da giac duoc tao boi 4 dinh ma chung ta vuaw tao ra
        if hard_case:#Su dung cai nay khi muon tao mot anh de lam kho mo hinh
            # Heavy smoothing rounds the corners of the square toward a circle
            img = cv2.GaussianBlur(img, (15, 15), 5.0)
    #cach ve hinh tam giac
    elif cls_name == "triangle":
        angles = [angle_rad + i * (2 * math.pi / 3) for i in range(3)]#goc duoc tao boi cac 
        if hard_case:
            # Perturb vertex angles and radii
            angles[1] += rng.uniform(-0.35, 0.35)
        pts = []
        for a in angles:
            r_v = radius * (rng.uniform(0.8, 1.2) if hard_case else rng.uniform(0.95, 1.05))
            pts.append([cx + r_v * math.cos(a), cy + r_v * math.sin(a)])
        pts_i = np.round(np.array(pts)).astype(np.int32)
        cv2.fillPoly(img, [pts_i], color)
        if hard_case:
            img = cv2.GaussianBlur(img, (11, 11), 3.5)

    # Add mild sensor noise
    noise_sigma = 14.0 if hard_case else 4.0
    noise = rng.normal(0, noise_sigma, img.shape)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return img


def generate_shape_dataset(
    data_dir: str,
    samples_per_class: int = 150,
    unseen_per_class: int = 4,
    img_size: int = 128,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Generate synthetic images for circle, square, and triangle using OpenCV."""
    rng = np.random.RandomState(seed)
    main_samples: list[dict] = []
    unseen_samples: list[dict] = []

    for cls_name in SHAPE_CLASSES:
        cls_dir = os.path.join(data_dir, "main", cls_name)
        unseen_dir = os.path.join(data_dir, "unseen", cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        os.makedirs(unseen_dir, exist_ok=True)

        for i in range(samples_per_class):
            hard_case = (i % 9 == 0)
            img = _draw_single_shape(cls_name, rng, img_size=img_size, hard_case=hard_case)
            fpath = os.path.join(cls_dir, f"{cls_name}_{i:03d}.png")
            cv2.imwrite(fpath, img)
            main_samples.append({"path": fpath, "label": cls_name, "hard_case": hard_case})

        for j in range(unseen_per_class):
            img = _draw_single_shape(cls_name, rng, img_size=img_size, hard_case=(j == unseen_per_class - 1))
            fpath = os.path.join(unseen_dir, f"{cls_name}_unseen_{j:02d}.png")
            cv2.imwrite(fpath, img)
            unseen_samples.append({"path": fpath, "label": cls_name})

    return main_samples, unseen_samples


def segment_and_find_contour(img_bgr: np.ndarray) -> dict:
    """Convert to Grayscale, GaussianBlur, Otsu Threshold (auto-polarity), Canny Edge, and Contour."""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Input image is None or empty.")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr.copy()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Check border pixels: if background is white (>127), invert mask so object is foreground (255)
    border_pixels = np.concatenate([thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]])
    if np.median(border_pixels) > 127:
        thresh = cv2.bitwise_not(thresh)

    edges = cv2.Canny(thresh, 50, 150)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_contour = None
    if contours:
        best_contour = max(contours, key=cv2.contourArea)

    contour_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if best_contour is not None:
        cv2.drawContours(contour_vis, [best_contour], -1, (0, 220, 60), 2)
        x, y, w, h = cv2.boundingRect(best_contour)
        cv2.rectangle(contour_vis, (x, y), (x + w, y + h), (255, 80, 0), 1)

    return {
        "gray": gray,
        "blur": blur,
        "thresh": thresh,
        "edges": edges,
        "contour": best_contour,
        "contour_vis": contour_vis,
    }


def extract_shape_features(seg_result: dict) -> np.ndarray:
    """Compute 7 shape descriptors: [A, P, W, H, AR=W/H, C=4*pi*A/P^2, ED=edge_pixels/roi_pixels]."""
    cnt = seg_result["contour"]
    edges = seg_result["edges"]
    if cnt is None or len(cnt) < 3:
        return np.zeros(7, dtype=np.float64)

    area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    x, y, w, h = cv2.boundingRect(cnt)
    w_f = float(w)
    h_f = float(h)

    ar = w_f / max(h_f, 1.0)
    circ = (4.0 * math.pi * area) / max(perim * perim, 1e-6)

    roi_edges = edges[y : y + h, x : x + w]
    edge_pixels = float(np.count_nonzero(roi_edges))
    roi_area = max(w_f * h_f, 1.0)
    ed = edge_pixels / roi_area

    return np.array([area, perim, w_f, h_f, ar, circ, ed], dtype=np.float64)


def extract_hog_features_and_vis(
    img_gray: np.ndarray, resize_dim: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Resize grayscale image to 64x64, extract 1764D HOG vector, and render HOG gradient visualization."""
    resized = cv2.resize(img_gray, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)
    win_size = (resize_dim, resize_dim)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_vec = hog.compute(resized).flatten().astype(np.float64)

    # Render orientation histogram visualization per 8x8 cell
    gx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = np.mod(ang, 180.0)

    n_cells = resize_dim // 8
    vis = np.zeros((resize_dim, resize_dim), dtype=np.float64)
    bin_width = 180.0 / nbins

    cell_hists = np.zeros((n_cells, n_cells, nbins), dtype=np.float64)
    for cy in range(n_cells):
        for cx in range(n_cells):
            m_patch = mag[cy * 8 : (cy + 1) * 8, cx * 8 : (cx + 1) * 8]
            a_patch = ang[cy * 8 : (cy + 1) * 8, cx * 8 : (cx + 1) * 8]
            for b in range(nbins):
                mask = (a_patch >= b * bin_width) & (a_patch < (b + 1) * bin_width)
                cell_hists[cy, cx, b] = np.sum(m_patch[mask])

    max_val = max(np.max(cell_hists), 1e-6)
    cell_hists /= max_val

    for cy in range(n_cells):
        for cx in range(n_cells):
            center_x = cx * 8 + 4
            center_y = cy * 8 + 4
            for b in range(nbins):
                strength = cell_hists[cy, cx, b]
                if strength < 0.08:
                    continue
                # Draw tangent edge orientation (perpendicular to gradient direction)
                theta = math.radians(b * bin_width + bin_width / 2.0 + 90.0)
                dx = int(round(3.5 * math.cos(theta)))
                dy = int(round(3.5 * math.sin(theta)))
                cv2.line(
                    vis,
                    (center_x - dx, center_y - dy),
                    (center_x + dx, center_y + dy),
                    float(strength * 255.0),
                    1,
                )

    hog_vis = np.clip(vis, 0, 255).astype(np.uint8)
    return hog_vec, hog_vis


def prepare_bai2_arrays(
    samples: list[dict],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract Shape (7D) and HOG (1764D) arrays and shape DataFrame for all samples."""
    shape_list = []
    hog_list = []
    labels = []
    paths = []
    df_rows = []

    for item in samples:
        img_bgr = cv2.imread(item["path"])
        if img_bgr is None:
            continue
        seg = segment_and_find_contour(img_bgr)
        s_vec = extract_shape_features(seg)
        h_vec, _ = extract_hog_features_and_vis(seg["gray"], resize_dim=64)

        shape_list.append(s_vec)
        hog_list.append(h_vec)
        labels.append(item["label"])
        paths.append(item["path"])

        row = {"image_path": item["path"], "label": item["label"]}
        for col, val in zip(SHAPE_FEATURE_COLS, s_vec):
            row[col] = float(val)
        df_rows.append(row)

    X_shape = np.vstack(shape_list)
    X_hog = np.vstack(hog_list)
    y = np.array(labels)
    return pd.DataFrame(df_rows), X_shape, X_hog, y, paths


def run_bai2_experiments(
    X_shape: np.ndarray,
    X_hog: np.ndarray,
    y: np.ndarray,
    paths: list[str],
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Train and evaluate KNN, SVM, and Random Forest across 3 feature sets using the same 80/20 split."""
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=seed
    )

    X_combined = np.hstack([X_shape, X_hog])
    feature_sets = [
        ("Shape Only (7D)", X_shape),
        ("HOG Only (1764D)", X_hog),
        ("Shape + HOG (1771D)", X_combined),
    ]

    metrics_rows = []
    details = {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "y_test": y[test_idx],
        "test_paths": [paths[i] for i in test_idx],
        "runs": {},
        "misclassified": [],
    }

    for feat_name, X_mat in feature_sets:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_mat[train_idx])
        X_test = scaler.transform(X_mat[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]

        models = [
            ("KNN", KNeighborsClassifier(n_neighbors=5)),
            ("SVM", SVC(kernel="rbf", C=1.5, random_state=seed)),
            ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=seed)),
        ]

        for model_name, clf in models:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=SHAPE_CLASSES)

            metrics_rows.append(
                {
                    "Feature_Set": feat_name,
                    "Num_Dims": X_mat.shape[1],
                    "Model": model_name,
                    "Accuracy": round(acc, 4),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1-score": round(f1, 4),
                }
            )

            run_key = f"{feat_name} | {model_name}"
            details["runs"][run_key] = {
                "scaler": scaler,
                "model": clf,
                "feat_name": feat_name,
                "model_name": model_name,
                "confusion_matrix": cm,
                "y_pred": y_pred,
            }

            for idx_local, (true_lbl, pred_lbl) in enumerate(zip(y_test, y_pred)):
                if true_lbl != pred_lbl:
                    global_idx = test_idx[idx_local]
                    details["misclassified"].append(
                        {
                            "Feature_Set": feat_name,
                            "Model": model_name,
                            "image_path": paths[global_idx],
                            "true_label": true_lbl,
                            "pred_label": pred_lbl,
                            "Area_A": float(X_shape[global_idx, 0]),
                            "AspectRatio_AR": float(X_shape[global_idx, 4]),
                            "Circularity_C": float(X_shape[global_idx, 5]),
                        }
                    )

    return pd.DataFrame(metrics_rows), details


def _plot_bai2_figures(
    df_shape: pd.DataFrame,
    metrics_df: pd.DataFrame,
    details: dict,
    unseen_samples: list[dict],
    output_dir: str,
) -> pd.DataFrame:
    """Generate all 6 figures for Bài 2 and return unseen prediction DataFrame."""
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Segmentation Steps (Original -> Grayscale -> Blur -> Otsu -> Canny -> Contour)
    fig, axes = plt.subplots(3, 6, figsize=(16, 8.5))
    fig.suptitle(
        "Bài 2 - Yêu cầu 1: Chuỗi Bước Tiền xử lý và Phân đoạn Ảnh (Grayscale → Blur → Otsu → Canny → Contour)",
        fontsize=13.5,
        fontweight="bold",
    )
    for r_idx, cls_name in enumerate(SHAPE_CLASSES):
        sample_path = df_shape[df_shape["label"] == cls_name].iloc[1]["image_path"]
        img_bgr = cv2.imread(sample_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        seg = segment_and_find_contour(img_bgr)

        steps = [
            (img_rgb, f"{cls_name.upper()} - Ảnh gốc", None),
            (seg["gray"], "Grayscale", "gray"),
            (seg["blur"], "Làm mịn (Gaussian)", "gray"),
            (seg["thresh"], "Threshold / Otsu", "gray"),
            (seg["edges"], "Phát hiện biên Canny", "gray"),
            (seg["contour_vis"], "Contour & Khung bao", None),
        ]
        for c_idx, (im_arr, title, cmap) in enumerate(steps):
            axes[r_idx, c_idx].imshow(im_arr, cmap=cmap)
            axes[r_idx, c_idx].set_title(title, fontsize=9.5)
            axes[r_idx, c_idx].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_segmentation_steps.png"), dpi=150)
    plt.close(fig)

    # Figure 2: HOG Visualization & Dimensionality Comparison (7D Shape vs 1764D HOG)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.5))
    fig.suptitle(
        "Bài 2 - Yêu cầu 3: Trực quan hóa Đặc trưng HOG (1764 chiều) vs Đặc trưng Hình dạng (7 chiều)",
        fontsize=13,
        fontweight="bold",
    )
    for r_idx, cls_name in enumerate(SHAPE_CLASSES):
        row = df_shape[df_shape["label"] == cls_name].iloc[1]
        img_bgr = cv2.imread(row["image_path"])
        seg = segment_and_find_contour(img_bgr)
        hog_vec, hog_vis = extract_hog_features_and_vis(seg["gray"], resize_dim=64)

        axes[r_idx, 0].imshow(seg["contour_vis"])
        axes[r_idx, 0].set_title(
            f"{cls_name.upper()} (AR={row['AspectRatio_AR']:.2f}, C={row['Circularity_C']:.2f})"
        )
        axes[r_idx, 0].axis("off")

        axes[r_idx, 1].imshow(hog_vis, cmap="inferno")
        axes[r_idx, 1].set_title("Trực quan hóa Gradient HOG (64×64)")
        axes[r_idx, 1].axis("off")

        axes[r_idx, 2].plot(hog_vec[:250], color="#2980b9", linewidth=0.9)
        axes[r_idx, 2].set_title(f"Vector HOG (250/{len(hog_vec)} chiều đầu)")
        axes[r_idx, 2].set_xlabel("Chỉ số chiều (Dimension index)")
        axes[r_idx, 2].set_ylabel("Biên độ chuẩn hóa")
        axes[r_idx, 2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_hog_visualization.png"), dpi=150)
    plt.close(fig)

    # Figure 3: Model & Feature Comparison Bar Chart (9 combinations)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Bài 2 - Yêu cầu 4 & 5: So sánh KNN, SVM và Random Forest trên 3 Tập Đặc trưng",
        fontsize=13.5,
        fontweight="bold",
    )
    feat_groups = ["Shape Only (7D)", "HOG Only (1764D)", "Shape + HOG (1771D)"]
    models_order = ["KNN", "SVM", "Random Forest"]
    model_colors = {"KNN": "#3498db", "SVM": "#e74c3c", "Random Forest": "#2ecc71"}

    for plot_idx, metric_col in enumerate(["Accuracy", "F1-score"]):
        ax = axes[plot_idx]
        x = np.arange(len(feat_groups))
        width = 0.25
        for m_idx, m_name in enumerate(models_order):
            vals = [
                metrics_df[
                    (metrics_df["Feature_Set"] == fg) & (metrics_df["Model"] == m_name)
                ][metric_col].values[0]
                for fg in feat_groups
            ]
            rects = ax.bar(
                x + (m_idx - 1) * width, vals, width, label=m_name, color=model_colors[m_name]
            )
            for rect in rects:
                h = rect.get_height()
                ax.annotate(
                    f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(feat_groups)
        ax.set_ylim(0.65, 1.05)
        ax.set_ylabel(metric_col)
        ax.set_title(f"So sánh {metric_col} theo Tập đặc trưng & Mô hình")
        ax.legend(loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_model_comparison.png"), dpi=150)
    plt.close(fig)

    # Figure 4: 3x3 Confusion Matrices across all 9 combinations
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    fig.suptitle(
        "Bài 2 - Yêu cầu 5: Ma trận Nhầm lẫn (Confusion Matrix) của 9 Tổ hợp (Đặc trưng × Mô hình)",
        fontsize=14,
        fontweight="bold",
    )
    for r_idx, fg in enumerate(feat_groups):
        for c_idx, m_name in enumerate(models_order):
            ax = axes[r_idx, c_idx]
            run_key = f"{fg} | {m_name}"
            info = details["runs"][run_key]
            cm = info["confusion_matrix"]
            row_m = metrics_df[
                (metrics_df["Feature_Set"] == fg) & (metrics_df["Model"] == m_name)
            ].iloc[0]
            ax.imshow(cm, cmap="Greens")
            ax.set_title(
                f"{fg}\n{m_name} (Acc={row_m['Accuracy']:.3f}, F1={row_m['F1-score']:.3f})",
                fontsize=9.5,
                fontweight="bold",
            )
            ax.set_xticks(range(len(SHAPE_CLASSES)))
            ax.set_yticks(range(len(SHAPE_CLASSES)))
            ax.set_xticklabels(SHAPE_CLASSES)
            ax.set_yticklabels(SHAPE_CLASSES)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    val = cm[i, j]
                    color = "white" if val > cm.max() / 2 else "black"
                    ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_confusion_matrices.png"), dpi=150)
    plt.close(fig)

    # Figure 5: Predictions on Unseen Images
    _, X_sh_u, X_hg_u, y_u, paths_u = prepare_bai2_arrays(unseen_samples)
    X_cb_u = np.hstack([X_sh_u, X_hg_u])
    unseen_rows = []
    n_show = min(len(paths_u), 12)
    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    fig.suptitle(
        "Bài 2 - Yêu cầu 5: Dự đoán trên Ảnh Mới Chưa có trong Tập Huấn luyện (Unseen Images)",
        fontsize=13.5,
        fontweight="bold",
    )

    for idx in range(n_show):
        r_sh = details["runs"]["Shape Only (7D) | Random Forest"]
        r_hg = details["runs"]["HOG Only (1764D) | SVM"]
        r_cb = details["runs"]["Shape + HOG (1771D) | SVM"]

        p_sh = r_sh["model"].predict(r_sh["scaler"].transform(X_sh_u[idx : idx + 1]))[0]
        p_hg = r_hg["model"].predict(r_hg["scaler"].transform(X_hg_u[idx : idx + 1]))[0]
        p_cb = r_cb["model"].predict(r_cb["scaler"].transform(X_cb_u[idx : idx + 1]))[0]

        unseen_rows.append(
            {
                "image_path": paths_u[idx],
                "true_label": y_u[idx],
                "Pred_Shape_RF": p_sh,
                "Pred_HOG_SVM": p_hg,
                "Pred_Combined_SVM": p_cb,
            }
        )

        ax = axes[idx // 4, idx % 4]
        img_bgr = cv2.imread(paths_u[idx])
        seg = segment_and_find_contour(img_bgr)
        ax.imshow(seg["contour_vis"])
        ok = p_cb == y_u[idx]
        ax.set_title(
            f"True: {y_u[idx]}\nShape(RF): {p_sh}\nHOG(SVM): {p_hg} | Comb: {p_cb}",
            fontsize=8.5,
            color="green" if ok else "red",
        )
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_unseen_predictions.png"), dpi=150)
    plt.close(fig)

    # Figure 6: Misclassified Examples Analysis
    mis_list = details["misclassified"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle(
        "Bài 2 - Yêu cầu 5: Phân tích Ví dụ Phân loại Sai (Misclassified Examples)",
        fontsize=13.5,
        fontweight="bold",
    )
    axes_flat = axes.flatten()
    seen_paths = []
    unique_mis = []
    for item in mis_list:
        if item["image_path"] not in seen_paths:
            seen_paths.append(item["image_path"])
            unique_mis.append(item)

    for idx in range(8):
        ax = axes_flat[idx]
        if idx < len(unique_mis):
            m_item = unique_mis[idx]
            img_bgr = cv2.imread(m_item["image_path"])
            seg = segment_and_find_contour(img_bgr)
            ax.imshow(seg["contour_vis"])
            ax.set_title(
                f"[{m_item['Feature_Set']} - {m_item['Model']}]\nTrue: {m_item['true_label']} → Pred: {m_item['pred_label']}\nC={m_item['Circularity_C']:.2f}, AR={m_item['AspectRatio_AR']:.2f}",
                fontsize=8.5,
                color="#c0392b",
            )
        else:
            ax.text(0.5, 0.5, "Không còn mẫu sai khác", ha="center", va="center")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig6_misclassified_analysis.png"), dpi=150)
    plt.close(fig)

    return pd.DataFrame(unseen_rows)


def run_bai2_pipeline(
    output_dir: str = r"e:\Hoc tap\Hoc_tap\Hoc_ki_7\Xu_ly_anh\BTH06\outputs\bai2",
    samples_per_class: int = 150,
    seed: int = 42,
) -> dict:
    """Execute full Bài 2 pipeline, exporting synthetic dataset, CSVs, and 6 PNG figures."""
    os.makedirs(output_dir, exist_ok=True)
    synth_dir = os.path.join(output_dir, "synthetic_shapes")
    main_samples, unseen_samples = generate_shape_dataset(
        synth_dir, samples_per_class=samples_per_class, unseen_per_class=4, img_size=128, seed=seed
    )

    df_shape, X_shape, X_hog, y, paths = prepare_bai2_arrays(main_samples)
    df_shape.to_csv(os.path.join(output_dir, "bai2_shape_features.csv"), index=False, encoding="utf-8-sig")

    metrics_df, details = run_bai2_experiments(X_shape, X_hog, y, paths, seed=seed)
    metrics_df.to_csv(os.path.join(output_dir, "bai2_metrics.csv"), index=False, encoding="utf-8-sig")

    mis_df = pd.DataFrame(details["misclassified"])
    mis_df.to_csv(os.path.join(output_dir, "bai2_misclassified.csv"), index=False, encoding="utf-8-sig")

    unseen_preds_df = _plot_bai2_figures(df_shape, metrics_df, details, unseen_samples, output_dir)
    unseen_preds_df.to_csv(
        os.path.join(output_dir, "bai2_unseen_predictions.csv"), index=False, encoding="utf-8-sig"
    )

    print("=== BÀI 2: KẾT QUẢ 9 TỔ HỢP (ĐẶC TRƯNG × MÔ HÌNH) ===")
    print(metrics_df.to_string(index=False))
    print(f"Tổng số lượt mẫu phân loại sai ghi nhận: {len(mis_df)}")
    return {
        "df_shape": df_shape,
        "metrics_df": metrics_df,
        "misclassified_df": mis_df,
        "unseen_preds_df": unseen_preds_df,
        "details": details,
    }


if __name__ == "__main__":
    run_bai2_pipeline()

