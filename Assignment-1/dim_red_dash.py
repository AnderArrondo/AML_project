"""
Stroke Dataset — Dimensionality Reduction Dashboard
Run: streamlit run stroke_dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import streamlit as st

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stroke · Dim Reduction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Mono', monospace; letter-spacing: -0.02em; }

    [data-testid="stSidebar"] { background: #0f0f14; border-right: 1px solid #2a2a3a; }
    [data-testid="stSidebar"] * { color: #e0e0f0 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label { color: #9090b0 !important; font-size: 0.75rem !important; }

    .metric-card {
        background: #13131e; border: 1px solid #2a2a3a;
        border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.5rem;
    }
    .metric-label { font-size: 0.7rem; color: #6060a0; text-transform: uppercase; letter-spacing: 0.1em; }
    .metric-value { font-family: 'DM Mono', monospace; font-size: 1.4rem; color: #e8e8ff; margin-top: 2px; }
    .metric-sub   { font-size: 0.68rem; color: #5050a0; margin-top: 2px; }

    .sample-banner {
        background: #0e1a2e; border: 1px solid #1e3a5a; border-radius: 8px;
        padding: 0.6rem 1rem; margin-bottom: 1rem; font-size: 0.82rem; color: #7090c0;
    }
    .sample-banner b { color: #90c0f0; }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: #0a0a12; padding: 4px; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Mono', monospace; font-size: 0.8rem;
        color: #5050a0; background: transparent; border-radius: 6px; padding: 6px 14px;
    }
    .stTabs [aria-selected="true"] { background: #1e1e35 !important; color: #c0c0ff !important; }

    .stApp { background: #0a0a12; }
    .block-container { padding-top: 1.5rem; }
    h1 { color: #d0d0ff; font-size: 1.6rem !important; }
    h3 { color: #9090c0; font-size: 0.95rem !important; font-weight: 400; margin-bottom: 0.5rem; }
    hr { border-color: #1e1e35; margin: 0.5rem 0 1rem; }
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data(csv_path: str):
    df = pd.read_csv(csv_path).dropna()
    df.drop(columns=["id"], inplace=True)
    categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    X = df.drop(columns=["stroke"]).values.astype(float)
    y = df["stroke"].values
    feature_names = df.drop(columns=["stroke"]).columns.tolist()
    return X, y, feature_names


# ─── STRATIFIED SAMPLING ──────────────────────────────────────────────────────

def stratified_sample(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str,
    n_total: int,
    stroke_pct: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (X_samp, y_samp, idx) preserving class representation.

    Strategies
    ----------
    Proportional  – preserve original class ratio, subsample n_total rows
    Balanced      – equal samples per class (boosts minority visibility)
    Custom        – user-defined stroke % via stroke_pct slider
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    idx_per_cls = {int(c): np.where(y == c)[0] for c in classes}

    if strategy == "Proportional":
        ratios  = counts / counts.sum()
        per_cls = {int(c): max(1, int(round(ratios[i] * n_total)))
                   for i, c in enumerate(classes)}

    elif strategy == "Balanced":
        n_each  = n_total // len(classes)
        per_cls = {int(c): n_each for c in classes}

    else:  # Custom
        stroke_n    = max(1, int(round(stroke_pct / 100 * n_total)))
        no_stroke_n = max(1, n_total - stroke_n)
        per_cls     = {0: no_stroke_n, 1: stroke_n}

    chosen = []
    for c in classes:
        pool = idx_per_cls[int(c)]
        k    = min(per_cls[int(c)], len(pool))
        chosen.append(rng.choice(pool, size=k, replace=False))

    idx = np.concatenate(chosen)
    idx = rng.permutation(idx)   # shuffle → no ordering bias in scatter
    return X[idx], y[idx], idx


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Stroke Dashboard")
    st.markdown("---")

    csv_path    = st.text_input("CSV path", value="./data/stroke.csv")
    random_seed = st.number_input("Random seed", value=42, min_value=0, max_value=9999, step=1)

    # ── Sampling ──────────────────────────────────────────────────────────────
    st.markdown("### Sampling")
    use_sampling = st.toggle("Enable stratified sampling", value=True)

    if use_sampling:
        samp_strategy = st.selectbox(
            "Strategy",
            ["Proportional", "Balanced", "Custom"],
            help=(
                "**Proportional** — preserves original class ratio.\n\n"
                "**Balanced** — equal samples per class (minority gets boosted).\n\n"
                "**Custom** — you set the stroke % explicitly."
            ),
        )
        samp_n = st.slider("Sample size (n)", min_value=100, max_value=5000,
                           value=1000, step=100)
        stroke_pct = st.slider("Stroke % in sample", 1, 99, 20) \
                     if samp_strategy == "Custom" else 5.0
    else:
        samp_strategy, samp_n, stroke_pct = "Proportional", 0, 5.0

    # ── Visuals ───────────────────────────────────────────────────────────────
    st.markdown("### Global")
    palette     = st.selectbox("Color palette", ["Default", "Viridis", "Plasma", "Cool-Warm"])
    show_hull   = st.toggle("Show convex hulls", value=True)
    show_legend = st.toggle("Show legends", value=True)
    point_size  = st.slider("Point size",    10, 80,  30)
    point_alpha = st.slider("Point opacity", 0.2, 1.0, 0.65, 0.05)

    # ── PCA ───────────────────────────────────────────────────────────────────
    st.markdown("### PCA")
    pca_n_show = st.slider("Components in scree", 5, 30, 15)
    pca_pc_x   = st.selectbox("X-axis component", [f"PC{i}" for i in range(1, 11)], index=0)
    pca_pc_y   = st.selectbox("Y-axis component", [f"PC{i}" for i in range(1, 11)], index=1)
    pca_top_k  = st.slider("Top loadings shown", 3, 15, 8)

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    st.markdown("### t-SNE")
    tsne_perplexity = st.slider("Perplexity",      5,    100,  30)
    tsne_lr         = st.selectbox("Learning rate", ["auto", 50, 100, 200, 500], index=0)
    tsne_iter       = st.slider("Max iterations",  250, 2000, 1000, step=250)

    # ── UMAP ──────────────────────────────────────────────────────────────────
    st.markdown("### UMAP")
    umap_neighbors = st.slider("n_neighbors", 2,   100, 15)
    umap_min_dist  = st.slider("min_dist",    0.0, 1.0, 0.1, 0.05)
    umap_metric    = st.selectbox("Metric", ["euclidean", "manhattan", "cosine", "chebyshev"])


# ─── LOAD ─────────────────────────────────────────────────────────────────────

try:
    X_full, y_full, feature_names = load_data(csv_path)
except FileNotFoundError:
    st.error(f"File not found: `{csv_path}`. Update the path in the sidebar.")
    st.stop()

n_full, n_features = X_full.shape

# ─── APPLY STRATIFIED SAMPLING ────────────────────────────────────────────────
#
#  PCA  → ALWAYS fit on full data (cheap linear op; keeps loadings meaningful)
#          then project only the sample for scatter plots
#  t-SNE / UMAP → fit+transform on sample only (O(n²) and O(n log n) → must be small)
#

if use_sampling and samp_n < n_full:
    X_samp, y_samp, samp_idx = stratified_sample(
        X_full, y_full, samp_strategy, samp_n, stroke_pct, int(random_seed)
    )
    is_sampled = True
else:
    X_samp, y_samp, samp_idx = X_full, y_full, np.arange(n_full)
    is_sampled = False

n_samp = len(y_samp)

# Scaler fitted on full data → no data leakage from sample choice
scaler    = StandardScaler().fit(X_full)
X_sc_full = scaler.transform(X_full)
X_sc_samp = scaler.transform(X_samp)

# ─── PALETTE + HELPERS ────────────────────────────────────────────────────────

_palettes = {
    "Default":   ("#4c8ef7", "#f7714c"),
    "Viridis":   ("#3b528b", "#5ec962"),
    "Plasma":    ("#0d0887", "#f0f921"),
    "Cool-Warm": ("#3b4cc0", "#b40426"),
}
c0, c1       = _palettes[palette]
colors       = [c0, c1]
point_colors = [colors[int(lbl)] for lbl in y_samp]
SCATTER      = dict(alpha=point_alpha, s=point_size, edgecolors="none", linewidths=0)


def make_legend(ax):
    if not show_legend:
        return
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c0,
                   markersize=8, label="No Stroke"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c1,
                   markersize=8, label="Stroke"),
    ]
    ax.legend(handles=handles, fontsize=8, framealpha=0.15, labelcolor="white",
              facecolor="#1a1a2e", edgecolor="#3a3a5a")


def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#0d0d1a")
    ax.set_title(title, color="#c0c0ff", fontsize=10, fontweight="600", pad=6)
    ax.set_xlabel(xlabel, color="#6060a0", fontsize=8)
    ax.set_ylabel(ylabel, color="#6060a0", fontsize=8)
    ax.tick_params(colors="#3a3a6a", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a4a")


def draw_hulls(ax, X2d):
    if not show_hull:
        return
    for cls, color in zip([0, 1], colors):
        pts = X2d[y_samp == cls]
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                idx  = np.append(hull.vertices, hull.vertices[0])
                ax.plot(pts[idx, 0], pts[idx, 1], color=color,
                        lw=1.5, alpha=0.5, linestyle="--")
            except Exception:
                pass


def dark_fig(w=7, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0a0a12")
    ax.set_facecolor("#0d0d1a")
    return fig, ax


# ─── COMPUTE ──────────────────────────────────────────────────────────────────

@st.cache_data
def run_pca(X_full_sc, X_samp_sc, feat_names, seed):
    """Fit on full data; project sample. Loadings stay globally meaningful."""
    pca_full = PCA(random_state=seed).fit(X_full_sc)
    pca10    = PCA(n_components=min(10, X_full_sc.shape[1]), random_state=seed).fit(X_full_sc)
    X_proj   = pca10.transform(X_samp_sc)
    loadings = pd.DataFrame(
        pca10.components_.T,
        index=feat_names,
        columns=[f"PC{i+1}" for i in range(pca10.n_components_)],
    )
    return pca_full.explained_variance_ratio_, X_proj, loadings


@st.cache_data
def run_tsne(X_sc, perplexity, lr, n_iter, seed):
    lr_val = "auto" if lr == "auto" else int(lr)
    tsne   = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(X_sc) - 1),
        learning_rate=lr_val,
        max_iter=n_iter,
        random_state=seed,
    )
    return tsne.fit_transform(X_sc), tsne.kl_divergence_


@st.cache_data
def run_umap(X_sc, n_neighbors, min_dist, metric, seed):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, len(X_sc) - 1),
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    return reducer.fit_transform(X_sc)


with st.spinner("Computing projections…"):
    ev_ratio, X_pca_all, loadings_df = run_pca(
        X_sc_full, X_sc_samp, feature_names, int(random_seed)
    )
    X_tsne, kl_div = run_tsne(
        X_sc_samp, tsne_perplexity, tsne_lr, tsne_iter, int(random_seed)
    )
    X_umap = run_umap(
        X_sc_samp, umap_neighbors, umap_min_dist, umap_metric, int(random_seed)
    )

pc_x_idx = int(pca_pc_x[2:]) - 1
pc_y_idx = int(pca_pc_y[2:]) - 1
X_pca_2d = X_pca_all[:, [pc_x_idx, pc_y_idx]]
cum_var  = np.cumsum(ev_ratio)
n95      = int(np.argmax(cum_var >= 0.95) + 1)


# ─── HEADER ───────────────────────────────────────────────────────────────────

st.markdown("# Dimensionality Reduction · Stroke Dataset")
st.markdown("---")

# Sampling banner
if is_sampled:
    samp_ratio = (y_samp == 1).mean() * 100
    orig_ratio = (y_full  == 1).mean() * 100
    st.markdown(
        f'<div class="sample-banner">'
        f'⚡ <b>Stratified sampling active</b> — '
        f'showing <b>{n_samp:,}</b> of <b>{n_full:,}</b> rows '
        f'({n_samp / n_full * 100:.1f}%) · '
        f'strategy: <b>{samp_strategy}</b> · '
        f'no-stroke: <b>{(y_samp==0).sum():,}</b> · '
        f'stroke: <b>{(y_samp==1).sum():,}</b> · '
        f'stroke rate: <b>{samp_ratio:.1f}%</b> '
        f'(full dataset: {orig_ratio:.1f}%)'
        f'</div>',
        unsafe_allow_html=True,
    )

# Metric cards
c1m, c2m, c3m, c4m, c5m = st.columns(5)
for col, label, val, sub in [
    (c1m, "Plotted / Total", f"{n_samp:,}",           f"of {n_full:,} rows"),
    (c2m, "Features",        f"{n_features}",          "after encoding"),
    (c3m, "Stroke cases",    f"{(y_samp==1).sum()}",   f"{(y_samp==1).mean()*100:.1f}% of sample"),
    (c4m, "PCs for 95% var", f"{n95}",                 "fit on full data"),
    (c5m, "t-SNE KL div",    f"{kl_div:.3f}",          "lower = better"),
]:
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{val}</div>'
        f'<div class="metric-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("")


# ─── TABS ─────────────────────────────────────────────────────────────────────

tab_pca, tab_tsne, tab_umap, tab_compare, tab_sampling = st.tabs(
    ["◈  PCA", "◈  t-SNE", "◈  UMAP", "◈  Compare", "◈  Sampling"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · PCA
# ══════════════════════════════════════════════════════════════════════════════
with tab_pca:
    col_l, col_r = st.columns(2)

    with col_l:
        fig, ax = dark_fig()
        n_show  = min(pca_n_show, len(ev_ratio))
        xs      = np.arange(1, n_show + 1)
        ax.bar(xs, ev_ratio[:n_show] * 100, color="#4c8ef7", alpha=0.8, label="Individual")
        ax.step(xs, cum_var[:n_show] * 100, where="mid", color="#f7714c",
                lw=2, label="Cumulative")
        ax.axhline(95, color="#aaaacc", lw=1, linestyle="--", alpha=0.5)
        ax.text(n_show * 0.55, 96.5, f"95% → {n95} PCs", color="#aaaacc", fontsize=8)
        style(ax, "Scree Plot  (fit on full data)", "Principal Component", "Explained Variance (%)")
        ax.legend(fontsize=8, framealpha=0.15, labelcolor="white",
                  facecolor="#1a1a2e", edgecolor="#3a3a5a")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_r:
        fig, ax = dark_fig()
        ax.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=point_colors, **SCATTER)
        draw_hulls(ax, X_pca_2d)
        make_legend(ax)
        style(ax,
              f"{pca_pc_x} vs {pca_pc_y}  "
              f"({ev_ratio[pc_x_idx]*100:.1f}% + {ev_ratio[pc_y_idx]*100:.1f}%)",
              pca_pc_x, pca_pc_y)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("### Feature loadings")
    pc_sel   = st.selectbox("Component", [f"PC{i+1}" for i in range(min(10, n_features))], key="load_pc")
    sign_col = loadings_df[pc_sel].loc[loadings_df[pc_sel].abs().nlargest(pca_top_k).index]

    fig, ax  = dark_fig(9, 3.5)
    bar_cols = [c1 if v > 0 else c0 for v in sign_col]
    ax.barh(sign_col.index[::-1], sign_col.values[::-1], color=bar_cols[::-1], alpha=0.85)
    ax.axvline(0, color="#4a4a6a", lw=1)
    style(ax, f"Top {pca_top_k} loadings — {pc_sel}  (signed)", "Loading", "")
    ax.tick_params(axis="y", labelsize=8, colors="#9090c0")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Full loadings table"):
        st.dataframe(loadings_df.style.background_gradient(cmap="RdBu", axis=None),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · t-SNE
# ══════════════════════════════════════════════════════════════════════════════
with tab_tsne:
    col_l, col_r = st.columns(2)

    with col_l:
        fig, ax = dark_fig()
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=point_colors, **SCATTER)
        draw_hulls(ax, X_tsne)
        make_legend(ax)
        style(ax, f"t-SNE  (perplexity={tsne_perplexity}, iter={tsne_iter})", "Dim 1", "Dim 2")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_r:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
        fig.patch.set_facecolor("#0a0a12")
        for ax, cls, cmap_name, lbl in zip(axes, [0, 1], ["Blues", "Oranges"], ["No Stroke", "Stroke"]):
            mask = y_samp == cls
            ax.hexbin(X_tsne[mask, 0], X_tsne[mask, 1],
                      gridsize=25, cmap=cmap_name, linewidths=0.3, mincnt=1)
            style(ax, f"Density — {lbl}", "Dim 1", "Dim 2")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.info(
        f"**KL divergence:** `{kl_div:.4f}` — lower = better local preservation. "
        f"Perplexity `{tsne_perplexity}` should be ≪ n_samples ({n_samp:,})."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · UMAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_umap:
    col_l, col_r = st.columns(2)

    with col_l:
        fig, ax = dark_fig()
        ax.scatter(X_umap[:, 0], X_umap[:, 1], c=point_colors, **SCATTER)
        draw_hulls(ax, X_umap)
        make_legend(ax)
        style(ax,
              f"UMAP  (neighbors={umap_neighbors}, min_dist={umap_min_dist}, "
              f"metric={umap_metric})",
              "Dim 1", "Dim 2")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_r:
        feat_sel  = st.selectbox("Colour by feature", feature_names, key="umap_feat")
        feat_vals = X_samp[:, feature_names.index(feat_sel)]
        fig, ax   = dark_fig()
        sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=feat_vals,
                        cmap="plasma", **{**SCATTER, "alpha": min(point_alpha * 1.1, 1.0)})
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02).ax.tick_params(
            colors="#9090c0", labelsize=7)
        style(ax, f"UMAP — coloured by {feat_sel}", "Dim 1", "Dim 2")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("### Parameter sensitivity")
    s1, s2 = st.columns(2)
    for col, (title, kw) in zip([s1, s2], [
        ("Low neighbors  (n=5, d=0.0)",  dict(n_neighbors=5,  min_dist=0.0)),
        ("High neighbors (n=50, d=0.5)", dict(n_neighbors=50, min_dist=0.5)),
    ]):
        X_s = run_umap(X_sc_samp, kw["n_neighbors"], kw["min_dist"],
                       umap_metric, int(random_seed))
        fig, ax = dark_fig(5, 4)
        ax.scatter(X_s[:, 0], X_s[:, 1], c=point_colors,
                   alpha=point_alpha * 0.9, s=point_size * 0.8, edgecolors="none")
        style(ax, title, "Dim 1", "Dim 2")
        col.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0a0a12")
    for ax, (X2d, title) in zip(axes, [
        (X_pca_2d, f"PCA  ({ev_ratio[pc_x_idx]*100:.1f}% + {ev_ratio[pc_y_idx]*100:.1f}%)"),
        (X_tsne,   f"t-SNE  (perp={tsne_perplexity})"),
        (X_umap,   f"UMAP  (k={umap_neighbors})"),
    ]):
        ax.scatter(X2d[:, 0], X2d[:, 1], c=point_colors, **SCATTER)
        draw_hulls(ax, X2d)
        style(ax, title, "Dim 1", "Dim 2")
        if show_legend:
            make_legend(ax)
    fig.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Embedding correlation")
    emb_df = pd.DataFrame({
        "PCA-D1":  X_pca_2d[:, 0], "PCA-D2":  X_pca_2d[:, 1],
        "tSNE-D1": X_tsne[:, 0],   "tSNE-D2": X_tsne[:, 1],
        "UMAP-D1": X_umap[:, 0],   "UMAP-D2": X_umap[:, 1],
    })
    corr = emb_df.corr().abs()
    fig, ax = dark_fig(7, 4.5)
    im = ax.imshow(corr, cmap="plasma", vmin=0, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=40, ha="right", fontsize=8, color="#9090c0")
    ax.set_yticklabels(corr.columns, fontsize=8, color="#9090c0")
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if corr.iloc[i, j] < 0.7 else "#111")
    plt.colorbar(im, ax=ax, fraction=0.035).ax.tick_params(colors="#9090c0", labelsize=7)
    style(ax, "Absolute correlation between embedding dimensions", "", "")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Method cheat-sheet")
    st.dataframe(pd.DataFrame({
        "Method":        ["PCA",               "t-SNE",             "UMAP"],
        "Type":          ["Linear",            "Non-linear",        "Non-linear"],
        "Preserves":     ["Global variance",   "Local neighbours",  "Local + global"],
        "Deterministic": ["Yes",               "Seed-dependent",    "Seed-dependent"],
        "Key param":     ["n_components",      "perplexity",        "n_neighbors / min_dist"],
        "Speed":         ["Fast (full data)",  "Slow → use sample", "Medium → use sample"],
        "Fit data":      ["Full dataset",      "Sample only",       "Sample only"],
    }).set_index("Method"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 · SAMPLING DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_sampling:
    st.markdown("### Class distribution — full vs sample")

    full_counts = [(y_full == c).sum() for c in [0, 1]]
    samp_counts = [(y_samp == c).sum() for c in [0, 1]]

    col_l, col_r = st.columns(2)

    with col_l:
        fig, ax = dark_fig(6, 4)
        x, w    = np.arange(2), 0.35
        ax.bar(x - w/2, full_counts, w, label="Full dataset", color=c0, alpha=0.8)
        ax.bar(x + w/2, samp_counts, w, label="Sample",       color=c1, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["No Stroke", "Stroke"], color="#9090c0")
        style(ax, "Class counts", "", "Count")
        ax.legend(fontsize=8, framealpha=0.15, labelcolor="white",
                  facecolor="#1a1a2e", edgecolor="#3a3a5a")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_r:
        fig, axes = plt.subplots(1, 2, figsize=(6, 4))
        fig.patch.set_facecolor("#0a0a12")
        for ax, counts, title in zip(
            axes,
            [full_counts, samp_counts],
            ["Full dataset", f"Sample (n={n_samp:,})"],
        ):
            wedges, texts, autotexts = ax.pie(
                counts, labels=["No Stroke", "Stroke"], colors=colors,
                autopct="%1.1f%%", startangle=90,
                textprops={"color": "#9090c0", "fontsize": 8},
            )
            for at in autotexts:
                at.set_color("white"); at.set_fontsize(8)
            ax.set_title(title, color="#c0c0ff", fontsize=9, pad=4)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Feature distribution: full vs sample
    st.markdown("### Feature distributions — full vs sample")
    num_feats = [f for f in ["age", "avg_glucose_level", "bmi"] if f in feature_names]
    if num_feats:
        fig, axes = plt.subplots(1, len(num_feats), figsize=(5 * len(num_feats), 4))
        fig.patch.set_facecolor("#0a0a12")
        if len(num_feats) == 1:
            axes = [axes]
        for ax, feat in zip(axes, num_feats):
            fi = feature_names.index(feat)
            ax.hist(X_full[:, fi], bins=30, density=True, alpha=0.55,
                    color=c0, label="Full", edgecolor="none")
            ax.hist(X_samp[:, fi], bins=30, density=True, alpha=0.65,
                    color=c1, label="Sample", edgecolor="none")
            style(ax, feat, feat, "Density")
            ax.legend(fontsize=8, framealpha=0.15, labelcolor="white",
                      facecolor="#1a1a2e", edgecolor="#3a3a5a")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Summary table
    st.markdown("### Sampling summary")
    st.dataframe(pd.DataFrame({
        "Metric": [
            "Total rows", "Sampled rows", "Sampling ratio",
            "No-stroke (full)", "Stroke (full)",
            "No-stroke (sample)", "Stroke (sample)",
            "Strategy", "Seed",
        ],
        "Value": [
            f"{n_full:,}",
            f"{n_samp:,}",
            f"{n_samp / n_full * 100:.1f}%",
            f"{full_counts[0]:,} ({full_counts[0] / n_full * 100:.1f}%)",
            f"{full_counts[1]:,} ({full_counts[1] / n_full * 100:.1f}%)",
            f"{samp_counts[0]:,} ({samp_counts[0] / n_samp * 100:.1f}%)",
            f"{samp_counts[1]:,} ({samp_counts[1] / n_samp * 100:.1f}%)",
            samp_strategy if is_sampled else "None (full data)",
            str(int(random_seed)),
        ],
    }), use_container_width=True, hide_index=True)