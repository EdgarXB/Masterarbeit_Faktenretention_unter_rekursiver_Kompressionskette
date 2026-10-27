# Originale gepaarte Artikelstatistik. Seed, Nenner und Ziehungen bleiben unverändert.
import numpy as np
import pandas as pd
SEED,B=32,10000
from faktenretention.protocol import METHODS, STAGES, LABELS, CONTRASTS
index_cache={}

def estimate(values):
    values = values.sort_index().dropna()
    assert values.index.is_unique
    ids = tuple(values.index.tolist())
    if not len(values):
        return dict(n_articles=0, mean=np.nan, ci_low=np.nan, ci_high=np.nan)
    if ids not in index_cache:
        index_cache[ids] = np.random.default_rng(SEED).integers(0, len(values), size=(B, len(values)))
    draws = values.to_numpy(dtype=float)[index_cache[ids]].mean(axis=1)
    low, high = np.quantile(draws, [.025, .975])
    return dict(n_articles=len(values), mean=float(values.mean()), ci_low=float(low), ci_high=float(high))

def means_for(frame, metric="retention_observed", policy="observed"):
    records = []
    for method in METHODS:
        for stage in STAGES:
            sub = frame[(frame.method == method) & (frame.stage_target_words == stage)]
            records.append({"method": method, "stage_target_words": stage, "policy": policy,
                            **estimate(sub.set_index("article_id")[metric])})
    return pd.DataFrame(records)

def contrasts_for(frame, metric="retention_observed", policy="observed", pair_complete=False):
    records = []
    for first, second in CONTRASTS:
        for stage in STAGES:
            sub = frame[frame.stage_target_words.eq(stage)]
            wide = sub.pivot(index="article_id", columns="method", values=metric)
            if pair_complete:
                valid = sub.pivot(index="article_id", columns="method", values="fully_observed")
                wide = wide.loc[valid[first] & valid[second]]
            paired = wide[[first, second]].dropna()
            delta = 100 * (paired[first] - paired[second])
            records.append({"first": first, "second": second, "contrast": f"{LABELS[first]} − {LABELS[second]}",
                "stage_target_words": stage, "policy": policy, **estimate(delta),
                "n_first_higher": int(delta.gt(1e-12).sum()), "n_tied": int(delta.abs().le(1e-12).sum()),
                "n_first_lower": int(delta.lt(-1e-12).sum())})
    return pd.DataFrame(records)
