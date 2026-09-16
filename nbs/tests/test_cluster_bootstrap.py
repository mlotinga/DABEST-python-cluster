"""Tests for the cluster-aware bootstrap and permutation test (`cluster_col`)."""

import numpy as np
import pandas as pd
import pytest

from dabest._api import load
from dabest._effsize_objects import TwoGroupsEffectSize, PermutationTest
from dabest._stats_tools import confint_2group_diff as ci2g


def make_clustered_data(n_participants=12, n_sets=4, sd_participant=1.5, seed=3):
    """
    Within-subject design in long format: every participant contributes
    `n_sets` paired sets of observations across three levels, so that the
    pairs (identified by `pair`) are nested within participants (`ID`).
    """
    rng = np.random.default_rng(seed)
    levels = ["L1", "L2", "L3"]
    rows = []
    for p in range(n_participants):
        u_p = rng.normal(0, sd_participant)  # participant-specific sensitivity
        for s in range(n_sets):
            u_s = rng.normal(0, 0.5)
            for k, level in enumerate(levels):
                rows.append(
                    dict(
                        ID="P{:02d}".format(p),
                        pair=p * n_sets + s,
                        Level=level,
                        Y=3 + 0.4 * k * (1 + u_p) + u_s + rng.normal(0, 0.5),
                    )
                )
    return pd.DataFrame(rows).sort_values(["pair", "Level"]).reset_index(drop=True)


DF = make_clustered_data()
PAIRED_KWARGS = dict(
    idx=("L1", "L2", "L3"),
    x="Level",
    y="Y",
    paired="sequential",
    id_col="pair",
    resamples=500,
    random_seed=11,
)


@pytest.fixture(scope="module")
def naive():
    return load(DF, **PAIRED_KWARGS)


@pytest.fixture(scope="module")
def clustered():
    return load(DF, cluster_col="ID", **PAIRED_KWARGS)


# ---------------------------------------------------------------------------
# Low-level resampling machinery
# ---------------------------------------------------------------------------
def test_cluster_codes_pool_labels_across_groups():
    (c0, c1), n = ci2g.cluster_codes(["a", "b", "a"], ["b", "c"])
    assert n == 3
    assert c0.tolist() == [0, 1, 0]
    assert c1.tolist() == [1, 2]

    with pytest.raises(ValueError):
        ci2g.cluster_codes(["a", None, "b"])


def test_cluster_tables_and_expand_cluster_draw():
    codes = np.array([2, 0, 2, 1, 0])
    offsets, members = ci2g.cluster_tables(codes, 3)
    assert offsets.tolist() == [0, 2, 3, 5]
    assert members[offsets[0] : offsets[1]].tolist() == [1, 4]
    assert members[offsets[2] : offsets[3]].tolist() == [0, 2]

    draw = np.array([2, 2, 1], dtype=np.int64)
    idx = ci2g.expand_cluster_draw(draw, offsets, members)
    assert idx.tolist() == [0, 2, 0, 2, 3]


def test_cluster_strata_follow_the_design():
    # Fully within-cluster: every cluster in both groups -> one stratum.
    strata, offsets = ci2g.cluster_strata(([0, 1, 2], [0, 1, 2]), 3)
    assert offsets.tolist() == [0, 3]

    # Nested: clusters 0-1 only in control, 2-3 only in test -> one stratum per group.
    strata, offsets = ci2g.cluster_strata(([0, 1], [2, 3]), 4)
    assert offsets.tolist() == [0, 2, 4]
    assert sorted(strata[:2].tolist()) == [0, 1]
    assert sorted(strata[2:].tolist()) == [2, 3]

    # Mixed: one stratum per membership pattern.
    strata, offsets = ci2g.cluster_strata(([0, 1, 2], [1, 2, 3]), 4)
    assert offsets.tolist() == [0, 1, 2, 4]


def test_cluster_bootstrap_reduces_to_ordinary_bootstrap_for_singleton_clusters():
    rng = np.random.default_rng(1)
    x0 = rng.normal(0, 1, 15)
    x1 = rng.normal(0.5, 1, 15)
    ids = np.arange(15)

    # Paired: every pair is its own cluster.
    plain = ci2g.compute_bootstrapped_diff(x0, x1, "baseline", "mean_diff", 300, 7)
    clus = ci2g.compute_cluster_bootstrapped_diff(x0, x1, ids, ids, "baseline", "mean_diff", 300, 7)
    assert np.array_equal(plain, clus)

    # Unpaired: every observation is its own cluster (nested design).
    x1u = rng.normal(0.5, 1, 20)
    plain = ci2g.compute_bootstrapped_diff(x0, x1u, None, "mean_diff", 300, 7)
    clus = ci2g.compute_cluster_bootstrapped_diff(
        x0, x1u, np.arange(15), 100 + np.arange(20), None, "mean_diff", 300, 7
    )
    assert np.array_equal(plain, clus)

    # The delete-one jackknife also coincides.
    plain = ci2g.compute_meandiff_jackknife(x0, x1, "baseline", "mean_diff")
    clus = ci2g.compute_cluster_jackknife(x0, x1, ids, ids, "baseline", "mean_diff")
    assert plain == pytest.approx(clus)


def test_cluster_bootstrap_keeps_pairs_together():
    # With a constant within-pair difference in every cluster, every paired
    # cluster-bootstrap resample must reproduce that difference exactly.
    x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    x1 = x0 + np.array([0.5, 0.5, 1.0, 1.0, 2.0, 2.0])
    clusters = np.array([0, 0, 1, 1, 2, 2])
    boots = ci2g.compute_cluster_bootstrapped_diff(x0, x1, clusters, clusters, "baseline", "mean_diff", 200, 3)
    # Three equally sized clusters are drawn with replacement, so every
    # resample's mean difference is the mean of three cluster differences.
    from itertools import combinations_with_replacement

    allowed = {round(np.mean(c), 10) for c in combinations_with_replacement([0.5, 1.0, 2.0], 3)}
    assert set(np.round(boots, 10)).issubset(allowed)


def test_cluster_bootstrap_rejects_misaligned_paired_clusters():
    x = np.arange(6, dtype=float)
    with pytest.raises(ValueError, match="same cluster"):
        ci2g.compute_cluster_bootstrapped_diff(x, x, [0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 0], "baseline", "mean_diff", 10, 1)


# ---------------------------------------------------------------------------
# dabest.load with cluster_col
# ---------------------------------------------------------------------------
def test_point_estimates_are_unchanged_by_clustering(naive, clustered):
    for effect_size in ["mean_diff", "cohens_d", "hedges_g"]:
        n = getattr(naive, effect_size).results["difference"].to_numpy()
        c = getattr(clustered, effect_size).results["difference"].to_numpy()
        assert n == pytest.approx(c)


def test_cluster_intervals_are_wider_under_participant_effects(naive, clustered):
    rn = naive.mean_diff.results
    rc = clustered.mean_diff.results
    sd_ratio = [np.std(c) / np.std(n) for c, n in zip(rc["bootstraps"], rn["bootstraps"])]
    assert all(r > 1.2 for r in sd_ratio)
    assert ((rc["bca_high"] - rc["bca_low"]) > (rn["bca_high"] - rn["bca_low"])).all()


def test_results_report_clusters(naive, clustered):
    rc = clustered.mean_diff.results
    assert rc["n_clusters"].tolist() == [12, 12]
    assert "n_clusters" not in naive.mean_diff.results.columns

    assert clustered.cluster_col == "ID"
    assert naive.cluster_col is None
    assert "clusters" in repr(clustered)
    assert "cluster" in repr(clustered.mean_diff)
    assert "cluster" not in repr(naive.mean_diff)


def test_baseline_pairing_with_clusters():
    result = load(DF, cluster_col="ID", **dict(PAIRED_KWARGS, paired="baseline")).mean_diff.results
    assert result["n_clusters"].tolist() == [12, 12]
    assert result["control"].tolist() == ["L1", "L1"]


def test_unpaired_with_clusters_spanning_both_groups():
    kwargs = dict(idx=("L1", "L3"), x="Level", y="Y", resamples=300, random_seed=5)
    naive = load(DF, **kwargs).mean_diff.results
    clustered = load(DF, cluster_col="ID", **kwargs).mean_diff.results
    assert clustered["n_clusters"].iloc[0] == 12
    assert clustered["difference"].iloc[0] == pytest.approx(naive["difference"].iloc[0])
    assert 0 <= clustered["pvalue_permutation"].iloc[0] <= 1
    assert len(clustered["permutations"].iloc[0]) == 5000


def test_unpaired_with_clusters_nested_within_groups():
    # Participants P00-P05 measured at L1 only; P06-P11 at L3 only.
    df = DF[DF["Level"].isin(["L1", "L3"])]
    first_half = df["ID"] < "P06"
    df = df[(first_half & (df["Level"] == "L1")) | (~first_half & (df["Level"] == "L3"))]
    result = load(
        df, idx=("L1", "L3"), x="Level", y="Y", cluster_col="ID", resamples=300, ps_adjust=True
    ).mean_diff.results
    assert result["n_clusters"].iloc[0] == 12
    assert result["control_N"].iloc[0] == 24
    assert 0 <= result["pvalue_permutation"].iloc[0] <= 1


def test_wide_format_with_clusters():
    wide = DF.pivot(index="pair", columns="Level", values="Y").reset_index()
    wide["ID"] = wide["pair"] // 4
    result = load(
        wide, idx=("L1", "L2", "L3"), paired="sequential", id_col="pair", cluster_col="ID", resamples=300
    ).mean_diff.results
    long_result = load(DF, cluster_col="ID", **dict(PAIRED_KWARGS, resamples=300)).mean_diff.results
    assert result["n_clusters"].tolist() == [12, 12]
    assert result["difference"].to_numpy() == pytest.approx(long_result["difference"].to_numpy())


def test_delta2_and_mini_meta_with_clusters():
    df = DF[DF["Level"].isin(["L1", "L3"])].copy()
    df["Env"] = np.where(df["pair"] % 4 < 2, "A", "B")

    delta2 = load(
        df, x=["Level", "Env"], y="Y", delta2=True, experiment="Env",
        paired="sequential", id_col="pair", cluster_col="ID", resamples=300,
    )
    dd = delta2.mean_diff.delta_delta
    assert dd.bca_low < dd.difference < dd.bca_high
    assert len(dd.bootstraps_delta_delta) == 300
    dg = delta2.hedges_g.delta_delta
    assert dg.bca_low < dg.difference < dg.bca_high

    mini_meta = load(
        df, idx=(("L1", "L3"),), x="Level", y="Y", mini_meta=True,
        paired="sequential", id_col="pair", cluster_col="ID", resamples=300,
    )
    mm = mini_meta.mean_diff.mini_meta
    assert mm.bca_low < mm.difference < mm.bca_high


def test_other_effect_sizes_and_plot_with_clusters(clustered):
    import matplotlib

    matplotlib.use("Agg")
    assert len(clustered.median_diff.results) == 2
    fig = clustered.mean_diff.plot()
    assert fig is not None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_cluster_col_validation():
    with pytest.raises(IndexError, match="not a column"):
        load(DF, cluster_col="missing", **PAIRED_KWARGS)

    with pytest.raises(ValueError, match="same column as `y`"):
        load(DF, cluster_col="Y", **PAIRED_KWARGS)

    bad = DF.copy()
    bad.loc[0, "ID"] = "P99"  # one row of pair 0 now belongs to another participant
    with pytest.raises(ValueError, match="single cluster"):
        load(bad, cluster_col="ID", **PAIRED_KWARGS)

    bad = DF.copy()
    bad.loc[0, "ID"] = None
    with pytest.raises(ValueError, match="missing values"):
        load(bad, cluster_col="ID", **PAIRED_KWARGS)

    wide = DF.pivot(index="pair", columns="Level", values="Y").reset_index()
    with pytest.raises(ValueError, match="one of the groups"):
        load(wide, idx=("L1", "L2"), paired="sequential", id_col="pair", cluster_col="L1")


def test_two_groups_effect_size_direct_api():
    rng = np.random.default_rng(2)
    control = rng.normal(0, 1, 8)
    test = rng.normal(0.5, 1, 8)
    clusters = np.repeat([0, 1, 2, 3], 2)

    result = TwoGroupsEffectSize(
        control, test, "mean_diff", is_paired="baseline", resamples=200,
        control_clusters=clusters, test_clusters=clusters,
    )
    assert result.is_clustered
    assert result.n_clusters == 4
    assert result.to_dict()["n_clusters"] == 4

    plain = TwoGroupsEffectSize(control, test, "mean_diff", is_paired="baseline", resamples=200)
    assert not plain.is_clustered
    assert plain.n_clusters is None

    with pytest.raises(ValueError, match="same lengths"):
        TwoGroupsEffectSize(control, test, "mean_diff", resamples=200, control_clusters=clusters[:4], test_clusters=clusters)

    with pytest.raises(ValueError, match="Both"):
        TwoGroupsEffectSize(control, test, "mean_diff", resamples=200, control_clusters=clusters)

    with pytest.raises(ValueError, match="same cluster"):
        TwoGroupsEffectSize(
            control, test, "mean_diff", is_paired="baseline", resamples=200,
            control_clusters=clusters, test_clusters=clusters[::-1],
        )


def test_cluster_permutation_test():
    rng = np.random.default_rng(4)
    control = rng.normal(0, 1, 12)
    test = control + 1.0
    clusters = np.repeat([0, 1, 2, 3], 3)

    # Paired: with a constant difference of 1 in every pair, each cluster-level
    # sign flip changes the mean difference by a multiple of 2 * 3 / 12.
    paired = PermutationTest(
        control, test, "mean_diff", is_paired="baseline", permutation_count=200,
        control_clusters=clusters, test_clusters=clusters,
    )
    assert 0 <= paired.pvalue <= 1
    assert set(np.round(paired.permutations, 10)).issubset({-1.0, -0.5, 0.0, 0.5, 1.0})

    # Unpaired with shared clusters, adjusted p-value.
    shared = PermutationTest(
        control, test, "mean_diff", permutation_count=200, ps_adjust=True,
        control_clusters=clusters, test_clusters=clusters,
    )
    assert 0 <= shared.pvalue <= 1
    assert len(shared.permutations_var) == 200

    # Unpaired with nested clusters, adjusted p-value.
    nested = PermutationTest(
        control, test, "mean_diff", permutation_count=200, ps_adjust=True,
        control_clusters=clusters, test_clusters=clusters + 10,
    )
    assert 0 <= nested.pvalue <= 1
