import os
import sys
import cv2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
#khai bao dac trung 
RGB_COLS = ["R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"]
HSV_COLS = ["H_mean", "S_mean", "V_mean", "H_std", "S_std", "V_std"]
HIST_COLS = (
    [f"H_hist_{i}" for i in range(16)]
    + [f"S_hist_{i}" for i in range(8)]
    + [f"V_hist_{i}" for i in range(8)]
)
ENTROPY_COL = ["Gray_Entropy"]
ALL_FEATURE_COLS = RGB_COLS + HSV_COLS + HIST_COLS + ENTROPY_COL
CLASS_NAMES = ["apple", "banana", "orange"]


def discover_fruit_folders(dataset_root: str) -> dict[str, list[str]]:
    """Discover Apple, Banana, and Orange folders in Training directory without false positives."""
    train_dir = os.path.join(dataset_root, "Training")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    all_dirs = sorted(
        d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    )
    folders = {"apple": [], "banana": [], "orange": []}
    for name in all_dirs:
        low = name.lower()
        if low.startswith("apple "):
            folders["apple"].append(name)
        elif low.startswith("banana"):
            folders["banana"].append(name)
        elif low.startswith("orange"):
            folders["orange"].append(name)
    return folders


def sample_fruit_dataset(
    dataset_root: str,
    samples_per_class: int = 600,
    unseen_per_class: int = 4,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Sample balanced images across all subfolders of Apple, Banana, and Orange."""
    rng = np.random.RandomState(seed)
    folders_map = discover_fruit_folders(dataset_root)
    train_dir = os.path.join(dataset_root, "Training")
    test_dir = os.path.join(dataset_root, "Test")

    main_samples: list[dict] = []
    unseen_samples: list[dict] = []

    for label, subfolders in folders_map.items():
        n_sub = len(subfolders)
        base_quota = samples_per_class // n_sub
        remainder = samples_per_class % n_sub

        class_main: list[dict] = []
        class_pool_unseen: list[dict] = []

        for idx, sub in enumerate(subfolders):
            quota = base_quota + (1 if idx < remainder else 0)
            sub_train_path = os.path.join(train_dir, sub)
            files = sorted(
                f
                for f in os.listdir(sub_train_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            perm = rng.permutation(len(files))
            chosen_idx = perm[:quota]
            rest_idx = perm[quota:]

            for i in chosen_idx:
                class_main.append(
                    {
                        "path": os.path.join(sub_train_path, files[i]),
                        "sub_folder": sub,
                        "label": label,
                    }
                )

            sub_test_path = os.path.join(test_dir, sub)
            if os.path.isdir(sub_test_path):
                t_files = sorted(
                    f
                    for f in os.listdir(sub_test_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                )
                if t_files:
                    pick_t = t_files[rng.randint(0, len(t_files))]
                    class_pool_unseen.append(
                        {
                            "path": os.path.join(sub_test_path, pick_t),
                            "sub_folder": sub,
                            "label": label,
                        }
                    )
            elif len(rest_idx) > 0:
                class_pool_unseen.append(
                    {
                        "path": os.path.join(sub_train_path, files[rest_idx[0]]),
                        "sub_folder": sub,
                        "label": label,
                    }
                )

        main_samples.extend(class_main)
        unseen_perm = rng.permutation(len(class_pool_unseen))[:unseen_per_class]
        for i in unseen_perm:
            unseen_samples.append(class_pool_unseen[i])

    return main_samples, unseen_samples


def preprocess_image(
    img_bgr: np.ndarray, size: tuple[int, int] = (128, 128)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize BGR image to 128x128 and convert to RGB, HSV, and Grayscale."""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Input image is None or empty.")
    resized_bgr = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    return img_rgb, img_hsv, img_gray


def extract_color_stats_features(
    img_rgb: np.ndarray, img_hsv: np.ndarray, img_gray: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Extract Mean+Std (RGB & HSV), normalized HSV Histogram (32D), and Grayscale Entropy (1D)."""
    rgb_f = img_rgb.astype(np.float64)
    hsv_f = img_hsv.astype(np.float64)

    mean_rgb = rgb_f.mean(axis=(0, 1))
    std_rgb = rgb_f.std(axis=(0, 1))
    mean_std_rgb = np.concatenate([mean_rgb, std_rgb])

    mean_hsv = hsv_f.mean(axis=(0, 1))
    std_hsv = hsv_f.std(axis=(0, 1))
    mean_std_hsv = np.concatenate([mean_hsv, std_hsv])

    h_hist = cv2.calcHist([img_hsv], [0], None, [16], [0, 180]).flatten().astype(np.float64)
    s_hist = cv2.calcHist([img_hsv], [1], None, [8], [0, 256]).flatten().astype(np.float64)
    v_hist = cv2.calcHist([img_hsv], [2], None, [8], [0, 256]).flatten().astype(np.float64)

    h_hist /= max(h_hist.sum(), 1e-12)
    s_hist /= max(s_hist.sum(), 1e-12)
    v_hist /= max(v_hist.sum(), 1e-12)
    hist_hsv = np.concatenate([h_hist, s_hist, v_hist])

    gray_hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten().astype(np.float64)
    probs = gray_hist / max(gray_hist.sum(), 1e-12)
    nonzero_probs = probs[probs > 0]
    entropy = float(-np.sum(nonzero_probs * np.log2(nonzero_probs)))
    if abs(entropy) < 1e-12:
        entropy = 0.0

    all_features = np.concatenate([mean_std_rgb, mean_std_hsv, hist_hsv, [entropy]])

    return {
        "mean_std_rgb": mean_std_rgb,
        "mean_std_hsv": mean_std_hsv,
        "hist_hsv": hist_hsv,
        "entropy": entropy,
        "all_features": all_features,
    }


def build_bai1_dataframe(samples: list[dict]) -> pd.DataFrame:
    """Extract features for all sampled images and return a structured DataFrame."""
    rows = []
    for item in samples:
        img_bgr = cv2.imread(item["path"])
        if img_bgr is None:
            continue
        rgb, hsv, gray = preprocess_image(img_bgr, size=(128, 128))
        feats = extract_color_stats_features(rgb, hsv, gray)
        row = {
            "image_path": item["path"],
            "sub_folder": item["sub_folder"],
            "label": item["label"],
        }
        for col, val in zip(ALL_FEATURE_COLS, feats["all_features"]):
            row[col] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


def run_bai1_experiments(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Run TN1-TN4 with 80/20 stratified split and Train-only StandardScaler."""
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=seed
    )
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    experiments_spec = [
        ("TN1", "Mean + Std (RGB)", RGB_COLS, "KNN", KNeighborsClassifier(n_neighbors=5)),
        ("TN2", "Mean + Std (HSV)", HSV_COLS, "KNN", KNeighborsClassifier(n_neighbors=5)),
        (
            "TN3",
            "Histogram màu HSV",
            HIST_COLS,
            "SVM",
            SVC(kernel="rbf", C=1.0, random_state=seed),
        ),
        (
            "TN4",
            "Kết hợp toàn bộ đặc trưng",
            ALL_FEATURE_COLS,
            "SVM",
            SVC(kernel="rbf", C=1.0, random_state=seed),
        ),
    ]

    metrics_rows = []
    artifacts = {"train_df": train_df, "test_df": test_df}

    for exp_id, feat_desc, cols, model_name, clf in experiments_spec:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[cols].values)
        X_test = scaler.transform(test_df[cols].values)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)

        metrics_rows.append(
            {
                "Experiment": exp_id,
                "Features": feat_desc,
                "Num_Dims": len(cols),
                "Model": model_name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-score": round(f1, 4),
            }
        )
        artifacts[exp_id] = {
            "scaler": scaler,
            "model": clf,
            "cols": cols,
            "feat_desc": feat_desc,
            "model_name": model_name,
            "confusion_matrix": cm,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    return pd.DataFrame(metrics_rows), artifacts


def _plot_bai1_figures(
    df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    artifacts: dict,
    unseen_samples: list[dict],
    output_dir: str,
) -> pd.DataFrame:
    """Generate all 5 figures for Bài 1 and return unseen prediction DataFrame."""
    os.makedirs(output_dir, exist_ok=True)

    # Figure 1: Preprocessing RGB -> HSV -> Grayscale (128x128)
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle("Bài 1 - Yêu cầu 1: Tiền xử lý ảnh (Resize 128×128 & Chuyển đổi RGB sang HSV)", fontsize=14, fontweight="bold")
    for r_idx, cls_name in enumerate(CLASS_NAMES):
        sample_row = df[df["label"] == cls_name].iloc[0]
        img_bgr = cv2.imread(sample_row["image_path"])
        rgb, hsv, gray = preprocess_image(img_bgr, size=(128, 128))

        axes[r_idx, 0].imshow(rgb)
        axes[r_idx, 0].set_title(f"{cls_name.upper()} - RGB (128×128)")
        axes[r_idx, 1].imshow(hsv[:, :, 0], cmap="hsv")
        axes[r_idx, 1].set_title("Kênh H (Hue)")
        axes[r_idx, 2].imshow(hsv[:, :, 1], cmap="viridis")
        axes[r_idx, 2].set_title("Kênh S (Saturation)")
        axes[r_idx, 3].imshow(hsv[:, :, 2], cmap="gray")
        axes[r_idx, 3].set_title("Kênh V (Value)")
        axes[r_idx, 4].imshow(gray, cmap="gray")
        axes[r_idx, 4].set_title(f"Ảnh xám (Entropy={sample_row['Gray_Entropy']:.2f})")
        for c_idx in range(5):
            axes[r_idx, c_idx].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_preprocessing_rgb_hsv.png"), dpi=150)
    plt.close(fig)

    # Figure 2: Feature Distributions (RGB Mean, HSV Mean, Grayscale Entropy)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("Bài 1 - Yêu cầu 2: Phân bố Đặc trưng Thống kê & Entropy theo Nhãn", fontsize=13, fontweight="bold")
    colors = {"apple": "#e74c3c", "banana": "#f1c40f", "orange": "#e67e22"}
    for cls_name in CLASS_NAMES:
        sub = df[df["label"] == cls_name]
        axes[0].scatter(sub["R_mean"], sub["G_mean"], label=cls_name, alpha=0.6, s=18, c=colors[cls_name])
        axes[1].scatter(sub["H_mean"], sub["S_mean"], label=cls_name, alpha=0.6, s=18, c=colors[cls_name])
    axes[0].set_xlabel("R_mean")
    axes[0].set_ylabel("G_mean")
    axes[0].set_title("Không gian Mean RGB (R vs G)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("H_mean (Hue)")
    axes[1].set_ylabel("S_mean (Saturation)")
    axes[1].set_title("Không gian Mean HSV (H vs S)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    entropy_data = [df[df["label"] == c]["Gray_Entropy"].values for c in CLASS_NAMES]
    bp = axes[2].boxplot(entropy_data, tick_labels=CLASS_NAMES, patch_artist=True)
    for patch, c in zip(bp["boxes"], CLASS_NAMES):
        patch.set_facecolor(colors[c])
        patch.set_alpha(0.65)
    axes[2].set_title("Phân bố Shannon Entropy Ảnh xám")
    axes[2].set_ylabel("Entropy (bits)")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_feature_distributions.png"), dpi=150)
    plt.close(fig)

    # Figure 3: Confusion Matrices for TN1 - TN4
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    fig.suptitle("Bài 1 - Yêu cầu 4: Ma trận Nhầm lẫn (Confusion Matrix) của 4 Thí nghiệm", fontsize=14, fontweight="bold")
    for idx, exp_id in enumerate(["TN1", "TN2", "TN3", "TN4"]):
        ax = axes[idx // 2, idx % 2]
        info = artifacts[exp_id]
        cm = info["confusion_matrix"]
        im = ax.imshow(cm, cmap="Blues")
        row_m = metrics_df[metrics_df["Experiment"] == exp_id].iloc[0]
        ax.set_title(
            f"{exp_id}: {info['feat_desc']} ({info['model_name']})\nAcc={row_m['Accuracy']:.4f} | F1={row_m['F1-score']:.4f}",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Nhãn dự đoán (Predicted)")
        ax.set_ylabel("Nhãn thực tế (True)")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_confusion_matrices.png"), dpi=150)
    plt.close(fig)

    # Figure 4: Bar chart comparing Accuracy, Precision, Recall, F1-score
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(metrics_df))
    width = 0.2
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score"]
    bar_colors = ["#2980b9", "#27ae60", "#f39c12", "#8e44ad"]
    for m_idx, m_name in enumerate(metric_names):
        vals = metrics_df[m_name].values
        rects = ax.bar(x + (m_idx - 1.5) * width, vals, width, label=m_name, color=bar_colors[m_idx])
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{r['Experiment']}\n({r['Features']} - {r['Model']})" for _, r in metrics_df.iterrows()]
    )
    ax.set_ylim(0.7, 1.05)
    ax.set_ylabel("Điểm số (Score)")
    ax.set_title("Bài 1 - Yêu cầu 4: So sánh Kết quả Đánh giá giữa 4 Thí nghiệm (TN1 - TN4)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_metrics_comparison.png"), dpi=150)
    plt.close(fig)

    # Figure 5: Predictions on unseen images
    unseen_df = build_bai1_dataframe(unseen_samples)
    pred_rows = []
    n_show = min(len(unseen_df), 12)
    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    fig.suptitle("Bài 1 - Yêu cầu 3: Dự đoán Nhãn cho Ảnh mới (Unseen Test Images)", fontsize=14, fontweight="bold")

    for idx in range(n_show):
        row = unseen_df.iloc[idx]
        preds = {}
        for exp_id in ["TN1", "TN2", "TN3", "TN4"]:
            info = artifacts[exp_id]
            x_vec = row[info["cols"]].values.reshape(1, -1).astype(np.float64)
            x_scaled = info["scaler"].transform(x_vec)
            preds[exp_id] = info["model"].predict(x_scaled)[0]

        pred_rows.append(
            {
                "image_path": row["image_path"],
                "sub_folder": row["sub_folder"],
                "true_label": row["label"],
                "TN1_KNN_RGB": preds["TN1"],
                "TN2_KNN_HSV": preds["TN2"],
                "TN3_SVM_HistHSV": preds["TN3"],
                "TN4_SVM_All": preds["TN4"],
            }
        )

        ax = axes[idx // 4, idx % 4]
        img_bgr = cv2.imread(row["image_path"])
        rgb, _, _ = preprocess_image(img_bgr, size=(128, 128))
        ax.imshow(rgb)
        ok = preds["TN4"] == row["label"]
        title_color = "green" if ok else "red"
        ax.set_title(
            f"True: {row['label']} ({row['sub_folder']})\nTN1:{preds['TN1']} | TN2:{preds['TN2']}\nTN3:{preds['TN3']} | TN4:{preds['TN4']}",
            fontsize=8.5,
            color=title_color,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_unseen_predictions.png"), dpi=150)
    plt.close(fig)

    return pd.DataFrame(pred_rows)


def run_bai1_pipeline(
    dataset_root: str = r"e:\Hoc tap\Hoc_tap\Hoc_ki_7\Xu_ly_anh\BTH06\fruits-360-100x100",
    output_dir: str = r"e:\Hoc tap\Hoc_tap\Hoc_ki_7\Xu_ly_anh\BTH06\outputs\bai1",
    samples_per_class: int = 600,
    seed: int = 42,
) -> dict:
    """Execute full Bài 1 pipeline, exporting CSVs and PNG figures."""
    os.makedirs(output_dir, exist_ok=True)
    main_samples, unseen_samples = sample_fruit_dataset(
        dataset_root, samples_per_class=samples_per_class, unseen_per_class=4, seed=seed
    )
    df = build_bai1_dataframe(main_samples)
    df.to_csv(os.path.join(output_dir, "bai1_features.csv"), index=False, encoding="utf-8-sig")

    metrics_df, artifacts = run_bai1_experiments(df, seed=seed)
    metrics_df.to_csv(os.path.join(output_dir, "bai1_metrics.csv"), index=False, encoding="utf-8-sig")

    unseen_preds_df = _plot_bai1_figures(df, metrics_df, artifacts, unseen_samples, output_dir)
    unseen_preds_df.to_csv(
        os.path.join(output_dir, "bai1_unseen_predictions.csv"), index=False, encoding="utf-8-sig"
    )

    print("=== BÀI 1: KẾT QUẢ 4 THÍ NGHIỆM ===")
    print(metrics_df.to_string(index=False))
    return {
        "features_df": df,
        "metrics_df": metrics_df,
        "unseen_preds_df": unseen_preds_df,
        "artifacts": artifacts,
    }


if __name__ == "__main__":
    run_bai1_pipeline()
