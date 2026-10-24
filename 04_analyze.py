#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

sns.set_theme(style="ticks", context="talk")


def corr_with_n(x, y) -> dict:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return {"rho": np.nan, "p": np.nan, "n": int(m.sum())}
    rho, p = spearmanr(x[m], y[m])
    return {"rho": float(rho), "p": float(p), "n": int(m.sum())}


def partial_spearman(x, y, *z) -> dict:
    cov = [np.asarray(c, dtype=float) for c in z]
    m = np.isfinite(x) & np.isfinite(y)
    for c in cov:
        m &= np.isfinite(c)
    x, y = x[m], y[m]
    cov = [c[m] for c in cov]
    if len(x) < 20:
        return {"rho": np.nan, "p": np.nan, "n": int(len(x))}
    rx = pd.Series(x).rank().to_numpy().reshape(-1, 1)
    ry = pd.Series(y).rank().to_numpy().reshape(-1, 1)
    rz = np.column_stack([pd.Series(c).rank().to_numpy() for c in cov])
    ex = rx - LinearRegression().fit(rz, rx).predict(rz)
    ey = ry - LinearRegression().fit(rz, ry).predict(rz)
    rho, p = spearmanr(ex.ravel(), ey.ravel())
    return {"rho": float(rho), "p": float(p), "n": int(len(x))}


def rank01(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return pd.Series(a).rank().to_numpy() / len(a)


def r2_from(y: np.ndarray, cols: list[np.ndarray]) -> float:
    Y = rank01(y)
    X = np.column_stack([np.ones(len(Y))] + [rank01(c) for c in cols])
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ beta
    return float(1.0 - resid.var() / Y.var())


def bin_identity(pid: pd.Series) -> pd.Series:
    bins = [0, 0.25, 0.35, 0.45, 0.60, 0.80, 1.01]
    labels = ["<25%", "25-35%", "35-45%", "45-60%", "60-80%", "80-100%"]
    return pd.cut(pid, bins=bins, labels=labels, right=False)


def main() -> None:
    meta = pd.read_csv(ROOT / "data" / "proteins.csv")
    pairs = pd.read_csv(RESULTS / "pairwise.csv")
    emb = np.load(RESULTS / "esmc300m_meanpool.npy")
    trypsin = pairs[pairs["same_fold"] == 1].copy()
    trypsin = trypsin[(trypsin["fold_a"] == "trypsin") & (trypsin["fold_b"] == "trypsin")]

    stats = []

    def add(name, scope, d):
        stats.append({"test": name, "scope": scope, **d})

    add("emb~seq", "all_pairs", corr_with_n(pairs["emb_cos"], pairs["seq_dist"]))
    add("emb~str", "all_pairs", corr_with_n(pairs["emb_cos"], pairs["str_dist"]))
    add("str~seq", "all_pairs", corr_with_n(pairs["str_dist"], pairs["seq_dist"]))
    add("emb~seq", "trypsin_pairs", corr_with_n(trypsin["emb_cos"], trypsin["seq_dist"]))
    add("emb~str", "trypsin_pairs", corr_with_n(trypsin["emb_cos"], trypsin["str_dist"]))
    add("str~seq", "trypsin_pairs", corr_with_n(trypsin["str_dist"], trypsin["seq_dist"]))
    add(
        "emb~str|seq",
        "trypsin_pairs",
        partial_spearman(trypsin["emb_cos"].to_numpy(), trypsin["str_dist"].to_numpy(), trypsin["seq_dist"].to_numpy()),
    )
    add(
        "emb~seq|str",
        "trypsin_pairs",
        partial_spearman(trypsin["emb_cos"].to_numpy(), trypsin["seq_dist"].to_numpy(), trypsin["str_dist"].to_numpy()),
    )
    add(
        "emb~str|len_diff",
        "trypsin_pairs",
        partial_spearman(trypsin["emb_cos"].to_numpy(), trypsin["str_dist"].to_numpy(), trypsin["len_diff"].to_numpy()),
    )

    if "comp_dist" in trypsin.columns:
        E = trypsin["emb_cos"].to_numpy()
        S = trypsin["str_dist"].to_numpy()
        Q = trypsin["seq_dist"].to_numpy()
        C = trypsin["comp_dist"].to_numpy()
        L = trypsin["len_diff"].to_numpy()

        add("emb~comp", "trypsin_pairs", corr_with_n(trypsin["emb_cos"], trypsin["comp_dist"]))
        add("str~comp", "trypsin_pairs", corr_with_n(trypsin["str_dist"], trypsin["comp_dist"]))
        add("emb~comp|seq", "trypsin_pairs", partial_spearman(E, C, Q))
        add("emb~comp|seq+str", "trypsin_pairs", partial_spearman(E, C, Q, S))
        add("emb~str|comp", "trypsin_pairs", partial_spearman(E, S, C))
        add("emb~str|seq+comp", "trypsin_pairs", partial_spearman(E, S, Q, C))
        add("emb~str|seq+comp+len", "trypsin_pairs", partial_spearman(E, S, Q, C, L))

        for nm, cols in (("seq", [Q]), ("struct", [S]), ("comp", [C]),
                         ("len", [L]), ("seq+struct", [Q, S]),
                         ("all", [Q, S, C, L])):
            add(f"R2_{nm}", "trypsin_pairs",
                {"rho": r2_from(E, cols), "p": np.nan, "n": len(trypsin)})

    trypsin["pid_bin"] = bin_identity(trypsin["pid_short"])
    bin_rows = []
    for label, g in trypsin.groupby("pid_bin", observed=True):
        d = corr_with_n(g["emb_cos"], g["str_dist"])
        d_seq = corr_with_n(g["emb_cos"], g["seq_dist"])
        bin_rows.append(
            {
                "pid_bin": str(label),
                "n_pairs": len(g),
                "median_pid": float(g["pid_short"].median()),
                "median_tm": float(g["tm_score"].median()),
                "median_emb": float(g["emb_cos"].median()),
                "emb~str_rho": d["rho"],
                "emb~str_p": d["p"],
                "emb~seq_rho": d_seq["rho"],
                "mean_tm": float(g["tm_score"].mean()),
                "std_tm": float(g["tm_score"].std()),
            }
        )
        add("emb~str", f"trypsin_pid_{label}", d)

    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(RESULTS / "identity_bins.csv", index=False)

    m = trypsin[["emb_cos", "seq_dist", "str_dist"]].dropna()
    lr = LinearRegression().fit(m[["seq_dist"]], m["emb_cos"])
    resid = m["emb_cos"] - lr.predict(m[["seq_dist"]])
    add("residual_emb~str", "trypsin_pairs", corr_with_n(resid.to_numpy(), m["str_dist"].to_numpy()))
    r2 = lr.score(m[["seq_dist"]], m["emb_cos"])
    add("emb_r2_from_seq", "trypsin_pairs", {"rho": float(r2), "p": np.nan, "n": int(len(m))})

    cross = pairs[pairs["same_fold"] == 0].copy()
    add("emb~seq", "cross_fold", corr_with_n(cross["emb_cos"], cross["seq_dist"]))
    add("median_emb", "trypsin_pairs", {"rho": float(trypsin["emb_cos"].median()), "p": np.nan, "n": len(trypsin)})
    add("median_emb", "cross_fold", {"rho": float(cross["emb_cos"].median()), "p": np.nan, "n": len(cross)})
    add("median_pid", "trypsin_pairs", {"rho": float(trypsin["pid_short"].median()), "p": np.nan, "n": len(trypsin)})
    add("median_pid", "cross_fold", {"rho": float(cross["pid_short"].median()), "p": np.nan, "n": len(cross)})
    add("median_tm", "trypsin_pairs", {"rho": float(trypsin["tm_score"].median()), "p": np.nan, "n": len(trypsin)})
    add("median_tm", "cross_fold", {"rho": float(cross["tm_score"].median()), "p": np.nan, "n": len(cross)})

    matched = []
    rng = np.random.default_rng(0)
    for _, row in cross.iterrows():
        pid = row["pid_short"]
        window = trypsin[(trypsin["pid_short"] - pid).abs() <= 0.03]
        if window.empty:
            continue
        matched.append(
            {
                "pid": pid,
                "emb_cross": row["emb_cos"],
                "emb_same": float(window["emb_cos"].median()),
                "tm_cross": row["tm_score"],
                "tm_same": float(window["tm_score"].median()),
            }
        )
    matched_df = pd.DataFrame(matched)
    if not matched_df.empty:
        delta = matched_df["emb_cross"] - matched_df["emb_same"]
        add(
            "emb_cross_minus_same_at_matched_pid",
            "fold_control",
            {
                "rho": float(delta.median()),
                "p": float((delta > 0).mean()),
                "n": int(len(delta)),
            },
        )
        matched_df.to_csv(RESULTS / "fold_matched.csv", index=False)

    same_sub = trypsin[trypsin["same_subclass"] == 1]
    diff_sub = trypsin[trypsin["same_subclass"] == 0]
    add("median_emb", "trypsin_same_subclass", {"rho": float(same_sub["emb_cos"].median()), "p": np.nan, "n": len(same_sub)})
    add("median_emb", "trypsin_diff_subclass", {"rho": float(diff_sub["emb_cos"].median()), "p": np.nan, "n": len(diff_sub)})
    add("median_pid", "trypsin_same_subclass", {"rho": float(same_sub["pid_short"].median()), "p": np.nan, "n": len(same_sub)})
    add("median_pid", "trypsin_diff_subclass", {"rho": float(diff_sub["pid_short"].median()), "p": np.nan, "n": len(diff_sub)})

    X = StandardScaler().fit_transform(emb)
    pca = PCA(n_components=2, random_state=0).fit_transform(X)
    meta = meta.copy()
    meta["pc1"] = pca[:, 0]
    meta["pc2"] = pca[:, 1]
    labels_fold = meta["fold"].map({"trypsin": 0, "subtilisin": 1}).to_numpy()
    try:
        sil_fold = float(silhouette_score(emb, labels_fold, metric="cosine"))
    except Exception:
        sil_fold = np.nan
    add("silhouette_fold", "proteins", {"rho": sil_fold, "p": np.nan, "n": len(meta)})
    tryp_mask = meta["fold"] == "trypsin"
    tryp = meta[tryp_mask]
    if tryp["subclass"].nunique() > 1 and (tryp["subclass"].value_counts() >= 2).sum() >= 2:
        keep = tryp["subclass"].map(tryp["subclass"].value_counts() >= 2)
        tryp2 = tryp[keep]
        sil_sub = float(
            silhouette_score(emb[tryp2.index.to_numpy()], tryp2["subclass"].to_numpy(), metric="cosine")
        )
        add("silhouette_subclass", "trypsin_proteins", {"rho": sil_sub, "p": np.nan, "n": len(tryp2)})

    layer_path = RESULTS / "esmc300m_layers.npy"
    if layer_path.exists() and "comp_dist" in trypsin.columns:
        layers = np.load(layer_path)
        ids = meta["id"].tolist()
        pos = {pid: i for i, pid in enumerate(ids)}
        ia = trypsin["id_a"].map(pos).to_numpy()
        ib = trypsin["id_b"].map(pos).to_numpy()
        S = trypsin["str_dist"].to_numpy()
        Q = trypsin["seq_dist"].to_numpy()
        C = trypsin["comp_dist"].to_numpy()
        L = trypsin["len_diff"].to_numpy()

        rows = []
        for li in range(layers.shape[1]):
            V = layers[:, li, :]
            V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
            e = 1.0 - np.einsum("ij,ij->i", V[ia], V[ib])
            rows.append({
                "layer": li + 1,
                "emb~str": corr_with_n(e, S)["rho"],
                "emb~seq": corr_with_n(e, Q)["rho"],
                "emb~comp": corr_with_n(e, C)["rho"],
                "emb~str|seq": partial_spearman(e, S, Q)["rho"],
                "emb~str|seq+comp": partial_spearman(e, S, Q, C)["rho"],
                "emb~str|seq+comp+len": partial_spearman(e, S, Q, C, L)["rho"],
                "R2_comp": r2_from(e, [C]),
                "R2_struct": r2_from(e, [S]),
            })
        lay_df = pd.DataFrame(rows)
        lay_df.to_csv(RESULTS / "layer_sweep.csv", index=False)

        best = lay_df.loc[lay_df["emb~str|seq+comp"].idxmax()]
        add("best_layer", "layer_sweep",
            {"rho": float(best["layer"]), "p": np.nan, "n": len(trypsin)})
        add("emb~str|seq+comp@best", "layer_sweep",
            {"rho": float(best["emb~str|seq+comp"]), "p": np.nan, "n": len(trypsin)})
        add("emb~str|seq+comp@final", "layer_sweep",
            {"rho": float(lay_df.iloc[-1]["emb~str|seq+comp"]), "p": np.nan,
             "n": len(trypsin)})

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
        for col, lab in (("emb~str", "marginal"),
                         ("emb~str|seq", "| seq"),
                         ("emb~str|seq+comp", "| seq + composition"),
                         ("emb~str|seq+comp+len", "| seq + comp + length")):
            axes[0].plot(lay_df["layer"], lay_df[col], marker="o", ms=3, label=lab)
        axes[0].axhline(0, color="k", lw=0.8)
        axes[0].set_xlabel("ESMC-300M hidden layer")
        axes[0].set_ylabel("Spearman ρ (embedding vs 1−TM)")
        axes[0].set_title("Structural signal by layer, trypsin pairs")
        axes[0].legend(fontsize=9)
        axes[1].plot(lay_df["layer"], lay_df["R2_comp"], marker="o", ms=3,
                     label="composition", color="#d62728")
        axes[1].plot(lay_df["layer"], lay_df["R2_struct"], marker="o", ms=3,
                     label="structure", color="#1f77b4")
        axes[1].set_xlabel("ESMC-300M hidden layer")
        axes[1].set_ylabel("rank-space R²  of embedding distance")
        axes[1].set_title("What each layer encodes")
        axes[1].legend(fontsize=9)
        fig.savefig(FIGS / "layer_sweep.png", dpi=160)
        plt.close(fig)

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(RESULTS / "stats.csv", index=False)
    meta.to_csv(RESULTS / "proteins_with_pca.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, x, xlabel in [
        (axes[0], "pid_short", "sequence identity"),
        (axes[1], "tm_score", "TM-score (TM-align)"),
        (axes[2], "seq_dist", "1 − sequence identity"),
    ]:
        if x == "seq_dist":
            sns.scatterplot(
                data=trypsin.sample(min(len(trypsin), 4000), random_state=0),
                x="str_dist",
                y="emb_cos",
                hue="same_subclass",
                palette={1: "#1f77b4", 0: "#ff7f0e"},
                s=12,
                alpha=0.4,
                ax=ax,
                legend=False,
            )
            ax.set_xlabel("1 − TM-score")
            ax.set_ylabel("ESMC cosine distance")
        else:
            sns.scatterplot(
                data=trypsin.sample(min(len(trypsin), 4000), random_state=0),
                x=x,
                y="emb_cos",
                s=12,
                alpha=0.35,
                color="#1f77b4",
                ax=ax,
                edgecolor="none",
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel("ESMC cosine distance")
    axes[0].set_title("Sequence vs embedding")
    axes[1].set_title("Structure vs embedding")
    axes[2].set_title("Structure vs embedding")
    fig.savefig(FIGS / "trypsin_pairwise_scatter.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    plot_df = bin_df.dropna(subset=["emb~str_rho"])
    ax.bar(plot_df["pid_bin"].astype(str), plot_df["emb~str_rho"], color="#4c78a8")
    for i, r in plot_df.reset_index(drop=True).iterrows():
        ax.text(i, r["emb~str_rho"] + 0.02, f"n={int(r['n_pairs'])}", ha="center", fontsize=10)
    ax.set_ylabel("Spearman ρ (embedding vs 1−TM)")
    ax.set_xlabel("Sequence-identity bin (trypsin–trypsin pairs)")
    ax.set_title("Does structure still track embeddings after matching identity?")
    ax.axhline(0, color="k", lw=0.8)
    fig.savefig(FIGS / "identity_binned_structure.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    palette = {"trypsin": "#1f77b4", "subtilisin": "#d62728"}
    sns.scatterplot(data=meta, x="pc1", y="pc2", hue="fold", style="kingdom", s=70, palette=palette, ax=ax)
    ax.set_title("PCA of ESMC-300M mean-pooled embeddings")
    fig.savefig(FIGS / "pca_fold.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    tryp_plot = meta[meta["fold"] == "trypsin"]
    sns.scatterplot(data=tryp_plot, x="pc1", y="pc2", hue="subclass", s=70, ax=ax)
    ax.set_title("Trypsin-only PCA, colored by functional subclass")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, title="subclass")
    fig.savefig(FIGS / "pca_subclass.png", dpi=160)
    plt.close(fig)

    if not matched_df.empty:
        fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
        ax.scatter(matched_df["emb_same"], matched_df["emb_cross"], s=14, alpha=0.35, c="#d62728")
        lim = [
            min(matched_df["emb_same"].min(), matched_df["emb_cross"].min()),
            max(matched_df["emb_same"].max(), matched_df["emb_cross"].max()),
        ]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("ESMC distance, trypsin–trypsin (identity-matched)")
        ax.set_ylabel("ESMC distance, trypsin–subtilisin")
        ax.set_title("Fold control at matched sequence identity")
        fig.savefig(FIGS / "fold_matched_control.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    sample = pairs.sample(min(len(pairs), 6000), random_state=0)
    sns.scatterplot(
        data=sample,
        x="pid_short",
        y="tm_score",
        hue="same_fold",
        palette={1: "#1f77b4", 0: "#d62728"},
        s=12,
        alpha=0.4,
        ax=ax,
        edgecolor="none",
    )
    ax.set_xlabel("sequence identity")
    ax.set_ylabel("TM-score")
    ax.set_title("Sequence and structure are themselves confounded")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["different fold", "same fold"], title="")
    fig.savefig(FIGS / "seq_vs_structure.png", dpi=160)
    plt.close(fig)

    print(stats_df.to_string(index=False))
    print("\nIdentity bins:")
    print(bin_df.to_string(index=False))


if __name__ == "__main__":
    main()
