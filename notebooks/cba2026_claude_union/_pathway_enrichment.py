"""Pathway enrichment of the MOSA + TabPFN improvement.

Reads ``prediction_per_target_decomposed.csv`` from the union-variant
prediction analysis run, ranks the 500 CRISPR genes and 500 drugs by
``delta_headline`` (TabPFN+MOSA vs RF baseline), and runs MSigDB Hallmark
pre-ranked GSEA on both families.

Outputs land in the same ``reports/cba2026_claude_union/prediction_analysis/
20260511_174623/`` directory as figP1 to figP12:

- ``figP13_gene_pathway_gsea`` — gene-side Hallmark NES bar plot
- ``figP14_drug_pathway_gsea`` — drug-side Hallmark NES bar plot
- ``figP15_shared_pathways`` — gene/drug two-column NES dot plot
- ``prediction_pathway_gsea_genes.csv``
- ``prediction_pathway_gsea_drugs.csv``
- ``prediction_drug_to_pathway_map.csv`` (audit trail for the drug→gene→pathway map)

Cached data live under ``data/pathways/`` so subsequent runs are offline:

- ``MSigDB_Hallmark_2020.gmt`` (50 Hallmark terms from Enrichr)
- ``repurposing_drugs_20200324.txt`` (Broad Drug Repurposing Hub)

The GDSC drug-target table (``data/clines/drugresponse_drug_targets.csv``) is
combined with the Broad table to cover both screen formats — match rate on the
500 drugs is ~74% with this union.
"""

from __future__ import annotations

import re
import sys
import urllib.request
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gseapy as gp

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from _prediction_analysis import (
    FAMILY_COLORS,
    FAMILY_DISPLAY,
    FAMILY_ORDER,
    FIG_DIR,
    ROOT,
    SINGLE_FIG_DIR,
    load_decomposed,
)

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from _plot_style import (
    PALETTE,
    configure_nature_style,
    panel_label,
    save_figure,
)


PATHWAY_DIR = ROOT / "data" / "pathways"
HALLMARK_GMT = PATHWAY_DIR / "MSigDB_Hallmark_2020.gmt"
REPURPOSING_TXT = PATHWAY_DIR / "repurposing_drugs_20200324.txt"
GDSC_TARGETS_CSV = ROOT / "data" / "clines" / "drugresponse_drug_targets.csv"
DECOMPOSED_CSV = FIG_DIR / "prediction_per_target_decomposed.csv"

ENRICHR_HALLMARK_URL = (
    "https://maayanlab.cloud/Enrichr/geneSetLibrary"
    "?mode=text&libraryName=MSigDB_Hallmark_2020"
)
REPURPOSING_URL = (
    "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/"
    "repurposing_drugs_20200324.txt"
)

SALT_SUFFIXES = (
    "hcl", "hydrochloride", "phosphate", "sulfate", "sulphate", "monohydrate",
    "dihydrochloride", "fumarate", "mesylate", "acetate", "citrate", "tartrate",
    "sodium", "potassium", "besylate", "tosylate", "maleate", "lactate",
    "oxalate", "succinate", "bromide", "chloride", "iodide", "trifluoroacetate",
    "2hcl", "dimesylate", "dihydrate", "calcium", "disodium", "isethionate",
    "dipivoxil", "etabonate", "trihydrate", "heptahydrate", "ethanolamine",
)

SIG_TIERS = [(0.05, "***"), (0.10, "**"), (0.25, "*")]


def _fetch_to_cache(url: str, dest: Path) -> None:
    PATHWAY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  caching {url} -> {dest.relative_to(ROOT)}")
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        f.write(r.read())


def _norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _candidates(name: str) -> set[str]:
    """Aliases of a drug name to try when looking up targets.

    Splits on the synonym separator ``__``, then strips common pharmaceutical
    salt suffixes so that ``Terazosin_HCl`` and ``Rucaparib_Phosphate__AG_014699``
    both land on their parent compound name. As a final fallback, every
    underscore-separated token of length >= 4 is added — rescues names like
    ``Palbociclib_Isethionate__PD0332991`` where the salt isn't in the strip
    list. Short tokens are excluded to avoid false hits like ``s`` or ``ld``.
    """
    out = {_norm(name)}
    for chunk in re.split(r"__+", name):
        c = _norm(chunk)
        if not c:
            continue
        out.add(c)
        stripped = c
        changed = True
        while changed:
            changed = False
            for s in SALT_SUFFIXES:
                if stripped.endswith(s) and len(stripped) > len(s):
                    stripped = stripped[:-len(s)]
                    changed = True
                    break
        out.add(stripped)
        for tok in chunk.split("_"):
            tk = _norm(tok)
            if len(tk) >= 4:
                out.add(tk)
    out.discard("")
    return out


def _load_hallmark() -> dict[str, list[str]]:
    if not HALLMARK_GMT.exists():
        _fetch_to_cache(ENRICHR_HALLMARK_URL, HALLMARK_GMT)
    pathways: dict[str, list[str]] = {}
    for line in HALLMARK_GMT.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        name, *rest = parts
        genes = [g.strip() for g in rest if g.strip()]
        if genes:
            pathways[name] = genes
    return pathways


def _load_drug_to_genes() -> dict[str, set[str]]:
    """Build {normalised drug name → set of HGNC symbols} from GDSC + Broad."""
    lookup: dict[str, set[str]] = {}

    gdsc = pd.read_csv(GDSC_TARGETS_CSV, index_col=0)
    gdsc = gdsc[gdsc["putative_gene_target"].notna()]
    for idx, row in gdsc.iterrows():
        parts = str(idx).split(";")
        if len(parts) < 2:
            continue
        name = parts[1]
        genes = [g.strip() for g in str(row["putative_gene_target"]).split(";") if g.strip()]
        if not genes:
            continue
        for c in _candidates(name):
            lookup.setdefault(c, set()).update(genes)

    if not REPURPOSING_TXT.exists():
        _fetch_to_cache(REPURPOSING_URL, REPURPOSING_TXT)
    rep = pd.read_csv(REPURPOSING_TXT, sep="\t", skiprows=9)
    for _, row in rep.iterrows():
        name = row.get("pert_iname")
        tgts = row.get("target")
        if not isinstance(name, str) or not isinstance(tgts, str):
            continue
        genes = [g.strip() for g in tgts.split("|") if g.strip()]
        if not genes:
            continue
        for c in _candidates(name):
            lookup.setdefault(c, set()).update(genes)

    return lookup


def _map_drugs(drugs: list[str], lookup: dict[str, set[str]]
               ) -> tuple[dict[str, list[str]], pd.DataFrame]:
    drug_genes: dict[str, list[str]] = {}
    audit_rows = []
    for d in drugs:
        matched_key = None
        genes: set[str] = set()
        for c in _candidates(d):
            if c in lookup:
                matched_key = c
                genes = lookup[c]
                break
        audit_rows.append({
            "drug": d,
            "matched_normalised_name": matched_key or "",
            "target_genes": ";".join(sorted(genes)),
            "n_target_genes": len(genes),
            "matched": matched_key is not None,
        })
        if matched_key and genes:
            drug_genes[d] = sorted(genes)
    return drug_genes, pd.DataFrame(audit_rows)


def _build_drug_sets(pathways: dict[str, list[str]],
                     drug_genes: dict[str, list[str]]) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    for path_name, genes in pathways.items():
        gset = set(genes)
        members = [d for d, gs in drug_genes.items() if any(g in gset for g in gs)]
        if members:
            sets[path_name] = members
    return sets


def _prerank(rnk_series: pd.Series, gene_sets) -> pd.DataFrame:
    rnk = (
        rnk_series.dropna()
        .reset_index()
    )
    rnk.columns = ["name", "score"]
    rnk = rnk.drop_duplicates(subset=["name"]).sort_values("score", ascending=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gp.prerank(
            rnk=rnk,
            gene_sets=gene_sets,
            outdir=None,
            min_size=5,
            max_size=2000,
            permutation_num=1000,
            seed=0,
            no_plot=True,
            verbose=False,
        )
    out = res.res2d.copy()
    out = out.rename(columns={
        "Term": "term",
        "ES": "es",
        "NES": "nes",
        "NOM p-val": "p_nominal",
        "FDR q-val": "fdr",
        "FWER p-val": "fwer",
        "Lead_genes": "leading_edge",
        "Tag %": "tag_pct",
        "Gene %": "gene_pct",
    })
    keep = ["term", "es", "nes", "p_nominal", "fdr", "fwer",
            "tag_pct", "gene_pct", "leading_edge"]
    out = out[[c for c in keep if c in out.columns]]
    out["nes"] = pd.to_numeric(out["nes"], errors="coerce")
    out["fdr"] = pd.to_numeric(out["fdr"], errors="coerce")
    out["p_nominal"] = pd.to_numeric(out["p_nominal"], errors="coerce")
    return out.sort_values("nes", ascending=False).reset_index(drop=True)


def _sig_marker(fdr: float) -> str:
    if pd.isna(fdr):
        return ""
    for thr, mark in SIG_TIERS:
        if fdr < thr:
            return mark
    return ""


def _format_term(name: str) -> str:
    """Pretty-print an Enrichr Hallmark term name for plot labels."""
    return name.replace("HALLMARK_", "").strip()


def _bar_plot(df: pd.DataFrame, *, title: str, color_pos: str,
              color_neg: str = PALETTE["common"], top_n: int = 10) -> plt.Figure:
    sig = df.dropna(subset=["nes"]).copy()
    pos = sig[sig["nes"] > 0].sort_values("nes", ascending=False).head(top_n)
    neg = sig[sig["nes"] < 0].sort_values("nes", ascending=True).head(top_n)
    keep = pd.concat([pos, neg], ignore_index=True)
    keep = keep.sort_values("nes").reset_index(drop=True)

    n_rows = max(len(keep), 1)
    fig, ax = plt.subplots(figsize=(3.6, max(2.4, 0.24 * n_rows + 0.9)))
    fig.subplots_adjust(left=0.46, right=0.92, top=0.90, bottom=0.18)
    if keep.empty:
        ax.text(0.5, 0.5, "No enrichment results",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    y = np.arange(len(keep))
    colors = [color_pos if v > 0 else color_neg for v in keep["nes"]]
    ax.barh(y, keep["nes"], color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([_format_term(t) for t in keep["term"]])
    ax.set_xlabel("Normalised enrichment score (NES)")
    ax.set_title(title)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)

    xspan = keep["nes"].abs().max() if keep["nes"].abs().max() > 0 else 1.0
    pad = xspan * 0.04
    for yi, (nes, fdr) in enumerate(zip(keep["nes"], keep["fdr"])):
        mark = _sig_marker(fdr)
        if not mark:
            continue
        sign = 1 if nes >= 0 else -1
        ax.text(nes + sign * pad, yi, mark, va="center",
                ha="left" if sign > 0 else "right",
                fontsize=plt.rcParams["font.size"] - 0.5,
                color="#222222")

    lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
    ax.set_xlim(-lim * 1.15, lim * 1.15)
    return fig


def _shared_dotplot(gene_df: pd.DataFrame, drug_df: pd.DataFrame,
                    *, top_n: int = 15) -> plt.Figure:
    """Two-column dot plot ranked by combined |NES| of gene + drug enrichments."""
    g = gene_df.set_index("term")[["nes", "fdr"]].add_prefix("gene_")
    d = drug_df.set_index("term")[["nes", "fdr"]].add_prefix("drug_")
    joined = g.join(d, how="outer")
    joined["combined"] = (joined["gene_nes"].fillna(0).abs()
                          + joined["drug_nes"].fillna(0).abs())
    joined = joined.sort_values("combined", ascending=False).head(top_n)
    joined = joined.iloc[::-1]

    n_rows = max(len(joined), 1)
    fig_h = max(3.2, 0.26 * n_rows + 1.6)
    fig, ax = plt.subplots(figsize=(4.6, fig_h))
    fig.subplots_adjust(left=0.40, right=0.80, top=0.92, bottom=0.20)
    if joined.empty:
        ax.text(0.5, 0.5, "No enrichment results",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    y = np.arange(len(joined))
    nes_all = pd.concat([joined["gene_nes"], joined["drug_nes"]]).dropna()
    vmax = max(nes_all.abs().max() if not nes_all.empty else 1.0, 1.0)
    cmap = plt.cm.RdBu_r
    norm_n = plt.Normalize(vmin=-vmax, vmax=vmax)

    def _size(fdr):
        if pd.isna(fdr):
            return 6.0
        v = max(-np.log10(max(fdr, 1e-6)), 0.0)
        return 6.0 + 80.0 * min(v / 3.0, 1.0)

    for col_x, prefix in [(0.0, "gene"), (1.0, "drug")]:
        nes = joined[f"{prefix}_nes"].values
        fdr = joined[f"{prefix}_fdr"].values
        for i in range(len(joined)):
            if pd.isna(nes[i]):
                ax.scatter(col_x, y[i], s=8, c="white", edgecolors="#cccccc",
                           linewidth=0.4)
            else:
                ax.scatter(col_x, y[i], s=_size(fdr[i]),
                           c=[cmap(norm_n(nes[i]))],
                           edgecolors="black", linewidth=0.4)

    ax.set_yticks(y)
    ax.set_yticklabels([_format_term(t) for t in joined.index])
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["CRISPR\ngenes", "Drugs"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.6, len(joined) - 0.4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_title("Shared Hallmark enrichment", loc="left")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_n)
    sm.set_array([])
    cax = fig.add_axes([0.84, 0.40, 0.024, 0.40])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("NES", fontsize=plt.rcParams["axes.labelsize"] - 1)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2.5)

    leg_ax = fig.add_axes([0.40, 0.04, 0.42, 0.08])
    leg_ax.set_axis_off()
    legend_x = 0.05
    for tier, label in [(0.05, "FDR<0.05"), (0.1, "FDR<0.1"), (0.25, "FDR<0.25")]:
        v = -np.log10(tier)
        leg_ax.scatter(legend_x, 0.5, s=6.0 + 80.0 * min(v / 3.0, 1.0),
                       c="white", edgecolors="black", linewidth=0.4,
                       transform=leg_ax.transAxes)
        leg_ax.text(legend_x + 0.04, 0.5, label, va="center",
                    fontsize=plt.rcParams["font.size"] - 1,
                    transform=leg_ax.transAxes)
        legend_x += 0.33
    return fig


def figP13_gene_pathway_gsea(decomposed: pd.DataFrame,
                              pathways: dict[str, list[str]]) -> tuple[Path, pd.DataFrame]:
    configure_nature_style("composite")
    sub = decomposed[decomposed["target_family"] == "crisprcas9"].copy()
    rnk = sub.set_index("target_name")["delta_headline"]
    print(f"  CRISPR prerank: {(~rnk.isna()).sum()} genes")
    res = _prerank(rnk, pathways)
    fig = _bar_plot(
        res, title="CRISPR — Hallmark enrichment of Δheadline",
        color_pos=FAMILY_COLORS["crisprcas9"], top_n=10,
    )
    panel_label(fig.axes[0], "a", offset=(-0.40, 1.04))
    out = FIG_DIR / "figP13_gene_pathway_gsea"
    save_figure(fig, out)
    # Single-panel companion (same plot under singles/).
    fig2 = _bar_plot(
        res, title="CRISPR — Hallmark enrichment of Δheadline",
        color_pos=FAMILY_COLORS["crisprcas9"], top_n=10,
    )
    save_figure(fig2, SINGLE_FIG_DIR / "single_figP13a_gene_gsea")
    return Path(str(out) + ".pdf"), res


def figP14_drug_pathway_gsea(decomposed: pd.DataFrame,
                              pathways: dict[str, list[str]],
                              drug_genes: dict[str, list[str]],
                              ) -> tuple[Path, pd.DataFrame]:
    configure_nature_style("composite")
    sub = decomposed[decomposed["target_family"] == "drugresponse"].copy()
    rnk = sub.set_index("target_name")["delta_headline"]
    mapped_drugs = set(drug_genes.keys())
    rnk = rnk[rnk.index.isin(mapped_drugs)]
    print(f"  Drug prerank: {(~rnk.isna()).sum()} drugs (mapped subset)")
    drug_sets = _build_drug_sets(pathways, drug_genes)
    drug_sets = {k: v for k, v in drug_sets.items() if len(v) >= 5}
    print(f"  Drug-pathway sets: {len(drug_sets)} (≥5 drugs each)")
    res = _prerank(rnk, drug_sets)
    fig = _bar_plot(
        res, title="Drugs — Hallmark enrichment of Δheadline",
        color_pos=FAMILY_COLORS["drugresponse"], top_n=10,
    )
    panel_label(fig.axes[0], "a", offset=(-0.40, 1.04))
    out = FIG_DIR / "figP14_drug_pathway_gsea"
    save_figure(fig, out)
    fig2 = _bar_plot(
        res, title="Drugs — Hallmark enrichment of Δheadline",
        color_pos=FAMILY_COLORS["drugresponse"], top_n=10,
    )
    save_figure(fig2, SINGLE_FIG_DIR / "single_figP14a_drug_gsea")
    return Path(str(out) + ".pdf"), res


def figP15_shared_pathways(gene_res: pd.DataFrame, drug_res: pd.DataFrame) -> Path:
    configure_nature_style("composite")
    fig = _shared_dotplot(gene_res, drug_res, top_n=15)
    panel_label(fig.axes[0], "a", offset=(-0.42, 1.04))
    out = FIG_DIR / "figP15_shared_pathways"
    save_figure(fig, out)
    fig2 = _shared_dotplot(gene_res, drug_res, top_n=15)
    save_figure(fig2, SINGLE_FIG_DIR / "single_figP15a_shared_pathways")
    return Path(str(out) + ".pdf")


def run_all() -> None:
    if not DECOMPOSED_CSV.exists():
        decomposed = load_decomposed()
    else:
        decomposed = pd.read_csv(DECOMPOSED_CSV)

    pathways = _load_hallmark()
    print(f"Loaded {len(pathways)} Hallmark pathways from {HALLMARK_GMT.relative_to(ROOT)}")

    drug_names = (
        decomposed[decomposed["target_family"] == "drugresponse"]["target_name"]
        .dropna().tolist()
    )
    lookup = _load_drug_to_genes()
    drug_genes, audit_df = _map_drugs(drug_names, lookup)
    match_rate = audit_df["matched"].mean()
    print(f"Drug→gene mapping: {audit_df['matched'].sum()}/{len(audit_df)} "
          f"= {match_rate:.1%} of decomposed drugs")
    audit_df["target_pathways"] = audit_df["target_genes"].apply(
        lambda gs: ";".join(sorted({
            p for p, genes in pathways.items()
            if any(g in set(genes) for g in (gs.split(";") if gs else []))
        }))
    )
    audit_path = FIG_DIR / "prediction_drug_to_pathway_map.csv"
    audit_df.to_csv(audit_path, index=False)
    print(f"  wrote {audit_path.relative_to(ROOT)}")

    p13, gene_res = figP13_gene_pathway_gsea(decomposed, pathways)
    gene_csv = FIG_DIR / "prediction_pathway_gsea_genes.csv"
    gene_res.to_csv(gene_csv, index=False)
    print(f"  wrote {gene_csv.relative_to(ROOT)}")

    p14, drug_res = figP14_drug_pathway_gsea(decomposed, pathways, drug_genes)
    drug_csv = FIG_DIR / "prediction_pathway_gsea_drugs.csv"
    drug_res.to_csv(drug_csv, index=False)
    print(f"  wrote {drug_csv.relative_to(ROOT)}")

    p15 = figP15_shared_pathways(gene_res, drug_res)

    for label, p in [("figP13", p13), ("figP14", p14), ("figP15", p15)]:
        print(f"{label}: {p.relative_to(ROOT)}")

    print("\nTop 5 positive-NES Hallmark terms (CRISPR genes):")
    print(gene_res.head(5)[["term", "nes", "fdr"]].to_string(index=False))
    print("\nTop 5 positive-NES Hallmark terms (drugs):")
    print(drug_res.head(5)[["term", "nes", "fdr"]].to_string(index=False))


if __name__ == "__main__":
    run_all()
