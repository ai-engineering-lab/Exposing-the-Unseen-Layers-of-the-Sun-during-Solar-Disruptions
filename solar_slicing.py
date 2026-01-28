"""
Solar image slicing via unsupervised decomposition (NMF, PCA, ICA, Robust PCA) and clustering.
Produces separate "layer" images from GOES-16 SUVI-style solar imagery.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# -----------------------------------------------------------------------------
# Loading and feature building
# -----------------------------------------------------------------------------


def load_solar_image(path: str | Path) -> np.ndarray:
    """Load a single image as grayscale float in [0, 1]. Shape (H, W)."""
    path = Path(path)
    im = Image.open(path)
    if im.mode != "L":
        im = im.convert("L")
    arr = np.asarray(im, dtype=np.float64)
    if arr.max() > 1:
        arr = arr / max(arr.max(), 1.0)
    arr = np.nan_to_num(np.clip(arr, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
    return arr


def load_image_stack(paths: list[str | Path], resize_to: tuple[int, int] | None = None) -> np.ndarray:
    """
    Load multiple images, align to same size, return (H, W, C).
    If resize_to is given, all are resized to (H, W).
    """
    grids = [load_solar_image(p) for p in paths]
    if resize_to:
        grids = [
            np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((resize_to[1], resize_to[0])), dtype=np.float64) / 255.0
            for g in grids
        ]
    # Align to smallest H, W
    min_h = min(g.shape[0] for g in grids)
    min_w = min(g.shape[1] for g in grids)
    grids = [g[:min_h, :min_w] for g in grids]
    return np.stack(grids, axis=-1)


def pixel_features_single(im: np.ndarray) -> np.ndarray:
    """
    Build (n_pixels, n_features) for a single image.
    Features: intensity, x_norm, y_norm (all in [0,1] for NMF).
    """
    h, w = im.shape
    xs = np.linspace(0, 1, w)
    ys = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(xs, ys)
    i_flat = im.ravel()
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    return np.column_stack([i_flat, x_flat, y_flat])


# -----------------------------------------------------------------------------
# Decomposition
# -----------------------------------------------------------------------------


def run_pca(X: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """X: (n_samples, n_features). Returns (components, transformed)."""
    pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0]))
    T = pca.fit_transform(X)
    return pca.components_, T


def run_nmf(X: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """X: (n_samples, n_features), non-negative. Returns (W, H) s.t. X ~ W @ H."""
    n_components = min(n_components, X.shape[1], X.shape[0])
    nmf = NMF(n_components=n_components, init="nndsvda", max_iter=500)
    W = nmf.fit_transform(X)
    H = nmf.components_
    return W, H


def run_ica(X: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """X: (n_samples, n_features). Returns (mixing, sources)."""
    n_components = min(n_components, X.shape[1], X.shape[0])
    ica = FastICA(n_components=n_components, max_iter=500)
    S = ica.fit_transform(X)
    return ica.mixing_, S


def run_rpca(X: np.ndarray, lam: float | None = None, max_iter: int = 100, tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust PCA: X = L + S, L low-rank (background), S sparse (anomalies).
    Uses IALM (Inexact Augmented Lagrange Multiplier). Pure numpy.
    X: (m, n). Returns (L, S).
    """
    m, n = X.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
    mu = 1.25 / (np.linalg.norm(X, ord=2) + 1e-8)
    rho = 1.5
    Y = np.zeros_like(X)
    S = np.zeros_like(X)
    for _ in range(max_iter):
        # L = singular_value_threshold(X - S + Y/mu, 1/mu)
        Z = X - S + Y / mu
        try:
            U, sig, Vt = np.linalg.svd(Z, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        tau = 1.0 / mu
        sig_thresh = np.maximum(sig - tau, 0.0)
        L = (U * sig_thresh) @ Vt
        # S = soft_threshold(X - L + Y/mu, lam/mu)
        T = X - L + Y / mu
        S = np.sign(T) * np.maximum(np.abs(T) - lam / mu, 0.0)
        # Y = Y + mu * (X - L - S); mu = min(mu * rho, 1e10)
        Y = Y + mu * (X - L - S)
        rel_err = np.linalg.norm(X - L - S, ord="fro") / (np.linalg.norm(X, ord="fro") + 1e-8)
        if rel_err < tol:
            break
        mu = min(mu * rho, 1e10)
    return L, S


# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------


def run_kmeans(X: np.ndarray, n_clusters: int = 4, **kwargs) -> np.ndarray:
    """X: (n_samples, n_features). Returns labels (n_samples,)."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, **kwargs)
    return km.fit_predict(X)


# -----------------------------------------------------------------------------
# Slicing pipeline: from paths to saved layers
# -----------------------------------------------------------------------------


def layers_from_stack(
    stack: np.ndarray,
    n_components: int = 3,
    out_dir: Path | None = None,
    prefix: str = "",
) -> dict[str, np.ndarray]:
    """
    stack: (H, W, C). Treat each pixel as (C,) and run PCA / NMF / ICA.
    Returns dict of name -> (H, W) layer image.
    """
    h, w, c = stack.shape
    X = stack.reshape(-1, c)
    layers = {}

    # PCA
    _, T_pca = run_pca(X, n_components=n_components)
    for i in range(T_pca.shape[1]):
        layer = MinMaxScaler(feature_range=(0, 1)).fit_transform(T_pca[:, i : i + 1]).ravel()
        layers[f"{prefix}pca_{i+1}"] = layer.reshape(h, w)

    # NMF (non-negative inputs)
    X_nmf = np.clip(X, 1e-6, None)
    W_nmf, _ = run_nmf(X_nmf, n_components=n_components)
    for i in range(W_nmf.shape[1]):
        layer = MinMaxScaler(feature_range=(0, 1)).fit_transform(W_nmf[:, i : i + 1]).ravel()
        layers[f"{prefix}nmf_{i+1}"] = layer.reshape(h, w)

    # ICA
    T_ica = run_ica(X, n_components=n_components)[1]
    for i in range(T_ica.shape[1]):
        layer = MinMaxScaler(feature_range=(0, 1)).fit_transform(T_ica[:, i : i + 1]).ravel()
        layers[f"{prefix}ica_{i+1}"] = layer.reshape(h, w)

    # Robust PCA: background (L) + anomalies (S)
    L, S = run_rpca(X)
    bg = np.mean(L, axis=1).reshape(h, w)
    anom = np.abs(S).max(axis=1).reshape(h, w)
    layers[f"{prefix}rpca_background"] = MinMaxScaler(feature_range=(0, 1)).fit_transform(bg.reshape(-1, 1)).ravel().reshape(h, w)
    layers[f"{prefix}rpca_anomalies"] = MinMaxScaler(feature_range=(0, 1)).fit_transform(anom.reshape(-1, 1)).ravel().reshape(h, w)

    return layers


def layers_from_single_image(
    im: np.ndarray,
    n_components: int = 3,
    n_clusters: int = 4,
    out_dir: Path | None = None,
    prefix: str = "",
) -> dict[str, np.ndarray]:
    """
    Single image: use (I, x, y) for decomposition and (I, x, y) for clustering.
    Returns dict of name -> (H, W) layer image.
    """
    h, w = im.shape
    X = pixel_features_single(im)
    layers = {}

    # PCA
    _, T_pca = run_pca(X, n_components=n_components)
    for i in range(T_pca.shape[1]):
        layer = MinMaxScaler(feature_range=(0, 1)).fit_transform(T_pca[:, i : i + 1]).ravel()
        layers[f"{prefix}pca_{i+1}"] = layer.reshape(h, w)

    # NMF
    X_nmf = np.clip(X, 1e-6, None)
    W_nmf, _ = run_nmf(X_nmf, n_components=n_components)
    for i in range(W_nmf.shape[1]):
        layer = MinMaxScaler(feature_range=(0, 1)).fit_transform(W_nmf[:, i : i + 1]).ravel()
        layers[f"{prefix}nmf_{i+1}"] = layer.reshape(h, w)

    # ICA
    T_ica = run_ica(X, n_components=n_components)[1]
    for i in range(T_ica.shape[1]):
        layer = MinMaxScaler(feature_range=(0, 1)).fit_transform(T_ica[:, i : i + 1]).ravel()
        layers[f"{prefix}ica_{i+1}"] = layer.reshape(h, w)

    # K-means on (I, x, y): each cluster -> binary or mean intensity
    labels = run_kmeans(X, n_clusters=n_clusters)
    for k in range(n_clusters):
        mask = (labels == k).reshape(h, w)
        # Layer = mean intensity where cluster k, 0 else; then normalize for visibility
        layer = np.where(mask, im, 0.0)
        layer = MinMaxScaler(feature_range=(0, 1)).fit_transform(layer.reshape(-1, 1)).ravel().reshape(h, w)
        layers[f"{prefix}kmeans_{k+1}"] = layer

    return layers


def save_layers(layers: dict[str, np.ndarray], out_dir: Path) -> None:
    """Save each layer as PNG in out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in layers.items():
        u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(u8).save(out_dir / f"{name}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Solar slicing: NMF/PCA/ICA + clustering")
    ap.add_argument("--images", nargs="+", default=None, help="Paths to solar images (stack for decomposition)")
    ap.add_argument("--single", type=str, default=None, help="Single image path (decomposition + clustering on I,x,y)")
    ap.add_argument("--out", type=str, default="output_slices", help="Output directory for layer PNGs")
    ap.add_argument("--components", type=int, default=3, help="Number of components for PCA/NMF/ICA")
    ap.add_argument("--clusters", type=int, default=4, help="Number of clusters for K-means (single-image only)")
    args = ap.parse_args()

    out_dir = Path(args.out)

    if args.images and len(args.images) > 0:
        paths = [Path(p) for p in args.images]
        stack = load_image_stack(paths)
        layers = layers_from_stack(stack, n_components=args.components, prefix="stack_")
        save_layers(layers, out_dir)
        print(f"Stack mode: {len(paths)} images -> {len(layers)} layers in {out_dir}")
        return

    if args.single:
        im = load_solar_image(args.single)
        layers = layers_from_single_image(
            im,
            n_components=args.components,
            n_clusters=args.clusters,
            prefix="",
        )
        save_layers(layers, out_dir)
        print(f"Single-image mode: {args.single} -> {len(layers)} layers in {out_dir}")
        return

    # Default: use project source images if present
    base = Path(__file__).resolve().parent
    candidates = [base / "image.png", base / "image-3.png", base / "image-5.png"]
    existing = [p for p in candidates if p.exists()]
    if len(existing) >= 2:
        stack = load_image_stack(existing)
        layers = layers_from_stack(stack, n_components=args.components, prefix="stack_")
        save_layers(layers, out_dir)
        print(f"Default stack: {[str(p) for p in existing]} -> {len(layers)} layers in {out_dir}")
    elif (base / "image.png").exists():
        im = load_solar_image(base / "image.png")
        layers = layers_from_single_image(im, n_components=args.components, n_clusters=args.clusters)
        save_layers(layers, out_dir)
        print(f"Default single: image.png -> {len(layers)} layers in {out_dir}")
    else:
        print("Usage: provide --images <path...> or --single <path>; or run from project root with image.png / image-3.png / image-5.png")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
