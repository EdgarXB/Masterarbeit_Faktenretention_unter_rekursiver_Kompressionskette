# Statische Originalauswertung; Eingaben/Ausgaben relativ zur übergebenen Arbeitskopie.
# Keine LLM-Aufrufe. Standard: vorhandene Regressionscaches verwenden.
from pathlib import Path
import os,sys

def main(work, recompute_bootstrap=False, workers=1):
    WORK=__import__('pathlib').Path(work).resolve()
    __import__('os').chdir(WORK/('supplement' if 'hauptexperiment_01_qa_faktenretention'[0].isdigit() else 'project/02_Auswertung'))
    from pathlib import Path
    from collections import Counter
    import gzip, hashlib, importlib.util, importlib.metadata, json, platform, sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from IPython.display import display, Markdown

    candidates = [Path.cwd(), Path.cwd() / "Notebooks und Grafiken"]
    BASE = next((p for p in candidates if (p / "hauptexperiment_qa_daten/source_manifest.json").is_file()), None)
    if BASE is None:
        raise FileNotFoundError("Notebook im Ordner mit hauptexperiment_qa_daten ausführen.")
    DATA = BASE / "hauptexperiment_qa_daten"
    VERSION = "aktuell_20260930"  # alternativ: "original_frozen"
    SEED, B = 32, 10_000
    METHODS = ["lead_k", "plain", "zero_shot_cot", "few_shot_cot", "auto_cot", "automate_cot"]
    STAGES = [500, 300, 150, 75]
    N_FACTS = 20
    LABELS = dict(zip(METHODS, ["Lead-k", "Plain", "Zero-Shot-CoT", "Few-Shot-CoT", "Auto-CoT", "Automate-CoT"]))
    CONTRASTS = [("zero_shot_cot", "plain"), ("few_shot_cot", "plain"),
                 ("auto_cot", "few_shot_cot"), ("automate_cot", "auto_cot")]
    OUT = BASE / "hauptexperiment_qa_ergebnisse" / VERSION
    FIG = BASE / "hauptexperiment_qa_grafiken" / VERSION
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.max_columns", 16)
    pd.set_option("display.max_rows", 30)
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 220, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    COLORS = dict(zip(METHODS, ["#777777", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]))
    MARKERS = dict(zip(METHODS, ["o", "s", "^", "D", "v", "P"]))
    LINESTYLES = dict(zip(METHODS, ["--", "-", "-.", "--", ":", "-"]))

    def read(name, version=VERSION):
        return pd.read_csv(DATA / version / name, keep_default_na=False, low_memory=False)

    def savefig(fig, name):
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def export(frame, name):
        frame.to_csv(OUT / f"{name}.csv", index=False)

    def fmt(x):
        return f"{x:.1f}".replace(".", ",")

    manifest = json.loads((DATA / "source_manifest.json").read_text())
    for entry in manifest["files"]:
        file = DATA / entry["file"]
        assert hashlib.sha256(file.read_bytes()).hexdigest() == entry["sha256"], file
        if file.suffix == ".gz":
            assert hashlib.sha256(gzip.decompress(file.read_bytes())).hexdigest() == entry["original_sha256"], file
    import qa_scorer_snapshot as scorer
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "matplotlib", "nbformat", "nbclient"]}
    display(Markdown(f"**Version:** `{VERSION}` · **Bootstrap:** {B:,} Artikelstichproben, Seed {SEED}."))
    display(pd.DataFrame([{"Komponente": "Python", "Version": platform.python_version()}] +
                         [{"Komponente": k, "Version": v} for k, v in versions.items()]))

    provenance = []
    for version in ["original_frozen", "aktuell_20260930"]:
        folder = DATA / version / "provenienz"
        for rel in ["frozen_protocol.json", "frozen_downstream_protocol.json", "protocol.json", "qa/manifest.json", "qa_new/manifest.json", "downstream_completion.json"]:
            path = folder / rel
            if path.exists():
                obj = json.loads(path.read_text())
                provenance.append({"version": version, "source": rel,
                    "status": obj.get("status", obj.get("missing_data_amendment_timing", "")),
                    "model": obj.get("model", obj.get("qa_answerer", "")),
                    "missing_policy_timing": obj.get("missing_data_amendment_timing", ""),
                    "retry_seeds": str(obj.get("retry_seeds", "")),
                    "complete": str(obj.get("complete", obj.get("all_expected_units_processed", "")))})
    provenance = pd.DataFrame(provenance)
    display(provenance); export(provenance, "provenienz")
    if VERSION == "aktuell_20260930":
        merged = json.loads((DATA / VERSION / "provenienz/result_merge_manifest.json").read_text())
        qa_merge = json.loads((DATA / VERSION / "provenienz/qa/merge_manifest.json").read_text())
        assert merged["retained_historical_rows"] == 400
        assert merged["replaced_plain_rows"] == merged["replaced_few_shot_rows"] == 100
        assert qa_merge["retained_historical_units"] == 32_000
        assert qa_merge["new_plain_units"] == qa_merge["new_few_shot_units"] == 8_000
        display(pd.Series(merged, name="Zusammenfassungs-Merge").to_frame())

    facts = pd.read_csv(DATA / "shared/canonical_facts.csv.gz", keep_default_na=False, low_memory=False)
    summaries = read("summaries.csv.gz")
    qa = read("answers.csv.gz").rename(columns={"summary_prompt_variant": "method"})
    archived = read("scored_archived.csv.gz").rename(columns={"summary_prompt_variant": "method"})
    article_cache = read("article_archived.csv.gz")
    KEY = ["article_id", "method", "stage_target_words", "canonical_fact_id"]
    CELL = ["article_id", "method", "stage_target_words"]
    assert len(facts) == 2_000 and facts.canonical_fact_id.is_unique
    assert facts.groupby("article_id").size().eq(N_FACTS).all()
    IDS = sorted(facts.article_id.unique().tolist())
    assert len(IDS) == 100
    assert len(summaries) == 600 and not summaries.duplicated(["article_id", "method"]).any()
    assert set(summaries.method) == set(METHODS)
    assert summaries.groupby("article_id").method.nunique().eq(6).all()
    assert len(qa) == 48_000 and not qa.duplicated(KEY).any()
    expected = {(a, m, t, f) for a, fs in facts.groupby("article_id") for f in fs.canonical_fact_id
                for m in METHODS for t in STAGES}
    assert set(qa[KEY].itertuples(index=False, name=None)) == expected
    assert set(archived[KEY].itertuples(index=False, name=None)) == expected
    fact_join = qa.merge(facts[["article_id", "canonical_fact_id", "dataset_index", "question", "reference_answer"]],
                        on=["article_id", "canonical_fact_id"], validate="many_to_one", suffixes=("", "_fact"))
    for field in ["dataset_index", "question", "reference_answer"]:
        assert fact_join[field].eq(fact_join[field + "_fact"]).all(), field
    assert summaries.groupby("article_id").article_sha256.nunique().eq(1).all()
    assert summaries.groupby("article_id").article.nunique().eq(1).all()
    test_articles = pd.read_csv(DATA / "shared/test_articles.csv.gz", keep_default_na=False)
    assert len(test_articles) == 100 and test_articles.article_id.is_unique
    input_check = summaries.merge(test_articles[["article_id","dataset_index","article_sha256","article"]], on="article_id", validate="many_to_one", suffixes=("", "_test"))
    for field in ["dataset_index","article_sha256","article"]:
        assert input_check[field].eq(input_check[field+"_test"]).all(), field
    for row in summaries.itertuples():
        assert hashlib.sha256(row.article.encode()).hexdigest() == row.article_sha256
    summary_long = pd.concat([summaries[["article_id", "method", "dataset_index", "article_sha256", "collection_phase", f"summary_{t}"]]
        .rename(columns={f"summary_{t}": "summary"}).assign(stage_target_words=t) for t in STAGES], ignore_index=True)
    assert len(summary_long) == 2_400 and not summary_long.duplicated(CELL).any()
    qa = qa.merge(summary_long[CELL + ["summary", "dataset_index"]], on=CELL,
                  validate="many_to_one", suffixes=("", "_summary"))
    assert qa.dataset_index.eq(qa.dataset_index_summary).all()
    with gzip.open(DATA / VERSION / "generation_states.jsonl.gz", "rt") as stream:
        states = [json.loads(line) for line in stream]
    state_audit = []
    lookup = summaries.set_index(["article_id", "method"])
    for rec in states:
        state = rec["state"]; row = lookup.loc[(rec["article_id"], rec["method"])]
        assert state["complete"] and state["method"] == rec["method"]
        assert state["source_sha256"] == row.article_sha256
        assert [s["target_words"] for s in state["stages"]] == STAGES
        previous = row.article
        chain_failure = False
        for stage in state["stages"]:
            target = stage["target_words"]; selected = stage["selected"]
            chain_failure = chain_failure or not selected["response"].strip()
            assert stage["input_text"] == previous, (rec["article_id"], rec["method"], target)
            assert selected["response"] == row[f"summary_{target}"], (rec["article_id"], rec["method"], target)
            state_audit.append({"article_id": rec["article_id"], "method": rec["method"],
                "stage_target_words": target, "complete": state["complete"],
                "run_signature": state["run_signature"], "source_state_file": rec["source_file"],
                "source_state_sha256": rec["source_sha256"],
                "input_sha256": hashlib.sha256(previous.encode()).hexdigest(),
                "output_sha256": hashlib.sha256(selected["response"].encode()).hexdigest(),
                "input_reference": "original" if target == 500 else f"summary_{STAGES[STAGES.index(target)-1]}",
                "empty_input": not previous.strip(), "chain_failure": chain_failure,
                "empty_output": selected.get("empty_output", not selected["response"].strip()),
                "done_reason": selected.get("done_reason", "deterministic"),
                "attempts": len(stage["attempts"]), "selected_attempt": stage.get("selected_attempt_index", selected.get("attempt_index", 1)),
                "sampling_seed": selected.get("sampling_seed", "deterministic"),
                "prompt_sha256": selected.get("prompt_sha256", "deterministic")})
            previous = selected["response"]
    state_audit = pd.DataFrame(state_audit)
    assert len(state_audit) == 2_400 and not state_audit.duplicated(CELL).any()
    assert state_audit.attempts.eq(1).all()
    assert state_audit.empty_output.eq(summary_long.set_index(CELL).loc[pd.MultiIndex.from_frame(state_audit[CELL]), "summary"].str.strip().eq("").to_numpy()).all()
    qa = qa.merge(state_audit[CELL + ["empty_output", "empty_input", "chain_failure"]], on=CELL, validate="many_to_one")
    failed_ids = sorted(state_audit.loc[state_audit.chain_failure, "article_id"].unique().tolist())
    qa["answer_origin"] = np.where(qa.empty_output, "empty_summary_rule", "qa_model")
    assert qa.loc[qa.empty_output, "answer"].eq("NOT_FOUND").all()
    display(Markdown(f"**Generierungsfehler:** {state_audit.empty_output.sum()} leere Ausgaben in {len(failed_ids)} Artikeln; "
        f"{state_audit.chain_failure.sum()} Stufenpositionen liegen in einer abgebrochenen Informationskette. "
        f"{qa.empty_output.sum()} exportierte NOT_FOUND-Positionen stammen aus der Regel für leere Summaries."))
    display(state_audit[state_audit.chain_failure][["article_id","method","stage_target_words","empty_output","empty_input","done_reason"]])
    export(state_audit, "generation_status_und_rekursive_eingaben")
    display(pd.DataFrame({"Einheit": ["Artikel", "Originalfakten", "Artikel × Methode", "Zusammenfassungspositionen", "QA-Positionen"],
                          "Soll": [100, 2_000, 600, 2_400, 48_000],
                          "Ist": [len(IDS), len(facts), len(summaries), len(summary_long), len(qa)]}))
    display(state_audit.groupby(["method", "done_reason"], dropna=False).size().rename("Stufen").reset_index())

    qa["technical_missing"] = qa.error.astype(str).ne("")
    qa["analysis_eligible"] = ~qa.technical_missing
    qa["grounded"] = [scorer.support_is_grounded(s, a, e) for s, a, e in zip(qa.summary, qa.answer, qa.support)]
    assert qa.loc[qa.analysis_eligible, "grounded"].all()
    assert qa.loc[qa.analysis_eligible, "answer"].str.strip().ne("").all()
    qa["retained"] = [scorer.retained(ref, ans) if ok else False for ref, ans, ok in
                      zip(qa.reference_answer, qa.answer, qa.analysis_eligible)]
    qa["lost"] = qa.analysis_eligible & qa.answer.str.upper().eq("NOT_FOUND")
    qa["reference_discordant"] = qa.analysis_eligible & ~qa.retained & ~qa.lost
    qa["label"] = np.select([qa.technical_missing, qa.retained, qa.lost, qa.reference_discordant],
                            ["technical_missing", "retained", "lost", "reference_discordant"], default="INVALID")
    assert not qa.label.eq("INVALID").any()
    qa["retention_value"] = np.where(qa.analysis_eligible, qa.retained.astype(float), np.nan)
    check = qa[KEY + ["analysis_eligible", "technical_missing", "retained", "lost", "reference_discordant"]].merge(
        archived[KEY + ["analysis_eligible", "technical_missing", "retention", "loss", "reference_discordant_answer"]],
        on=KEY, suffixes=("", "_archive"), validate="one_to_one")
    for local, stored in [("analysis_eligible", "analysis_eligible_archive"), ("technical_missing", "technical_missing_archive")]:
        assert check[local].eq(check[stored].astype(str).str.lower().eq("true")).all(), local
    for local, stored in [("retained", "retention"), ("lost", "loss"), ("reference_discordant", "reference_discordant_answer")]:
        valid = check.analysis_eligible
        assert check.loc[valid, local].eq(check.loc[valid, stored].astype(str).str.lower().eq("true")).all(), local
        assert check.loc[~valid, stored].astype(str).eq("").all(), stored
    status = qa.groupby(["method", "stage_target_words"]).agg(
        n_expected=("canonical_fact_id", "size"), n_valid=("analysis_eligible", "sum"),
        n_missing=("technical_missing", "sum"), n_synthetic_not_found=("empty_output", "sum"), n_retained=("retained", "sum"),
        n_lost=("lost", "sum"), n_reference_discordant=("reference_discordant", "sum")).reset_index()
    status["missing_percent"] = 100 * status.n_missing / status.n_expected
    assert (status.n_retained + status.n_lost + status.n_reference_discordant + status.n_missing).eq(status.n_expected).all()
    export(status, "qa_status_methoden_stufen")
    display(status)
    display(qa.groupby(["answer_status", "error", "label"], dropna=False).size().rename("QA-Positionen").reset_index())
    display(Markdown(f"**{qa.analysis_eligible.sum():,}/{len(qa):,} gültige Urteile; {qa.technical_missing.sum():,} technisch fehlend ({fmt(100*qa.technical_missing.mean())} %).**"))

    article = qa.groupby(CELL).agg(n_total=("canonical_fact_id", "size"),
        n_valid=("analysis_eligible", "sum"), n_missing=("technical_missing", "sum"),
        n_retained=("retained", "sum"), n_lost=("lost", "sum"),
        generation_chain_failure=("chain_failure", "any"),
        n_synthetic_not_found=("empty_output", "sum"),
        n_reference_discordant=("reference_discordant", "sum")).reset_index()
    assert article.n_total.eq(N_FACTS).all()
    assert (article.n_valid + article.n_missing).eq(N_FACTS).all()
    article["retention_observed"] = article.n_retained.div(article.n_valid.replace(0, np.nan))
    article["retention_lower"] = article.n_retained / N_FACTS
    article["retention_upper"] = (article.n_retained + article.n_missing) / N_FACTS
    article["fully_observed"] = article.n_missing.eq(0) & ~article.generation_chain_failure
    cache_check = article.merge(article_cache, on=CELL, validate="one_to_one")
    for new, old in [("retention_observed", "retention"), ("retention_lower", "retention_all_missing_zero"),
                     ("retention_upper", "retention_all_missing_one")]:
        assert np.allclose(cache_check[new], pd.to_numeric(cache_check[old]), atol=1e-12), new
    qa_complete_ids = sorted(article.groupby("article_id").n_missing.sum().loc[lambda s: s.eq(0)].index.tolist())
    complete_ids = sorted(set(qa_complete_ids) - set(failed_ids))
    export(article, "retention_artikelebene")
    display(article.groupby(["method", "stage_target_words"]).agg(
        n_articles=("article_id", "size"), n_complete_cells=("fully_observed", "sum"),
        min_valid_facts=("n_valid", "min"), max_valid_facts=("n_valid", "max"), n_missing=("n_missing", "sum")).reset_index())
    display(Markdown(f"**QA-vollständiger Panel:** {len(qa_complete_ids)}/100 Artikel; **ohne leere Generierungsketten:** {len(complete_ids)}/100."))

    index_cache = {}
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

    means = means_for(article)
    means_percent = means.assign(mean_percent=100*means["mean"], ci_low_percent=100*means.ci_low,
                                 ci_high_percent=100*means.ci_high)
    export(means_percent, "retention_mittelwerte_95ci")
    mean_table = means.assign(method=lambda d: d.method.map(LABELS),
        Ergebnis=lambda d: [f"{fmt(100*r.mean)} [{fmt(100*r.ci_low)}; {fmt(100*r.ci_high)}]; n={r.n_articles}" for r in d.itertuples()])
    display(mean_table.pivot(index="method", columns="stage_target_words", values="Ergebnis").reindex(index=list(LABELS.values()), columns=STAGES))

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    x = np.arange(4)
    for i, method in enumerate(METHODS):
        sub = means[means.method.eq(method)].set_index("stage_target_words").loc[STAGES]
        offset = (i - 2.5) * .025
        y = 100 * sub["mean"].to_numpy()
        err = np.vstack([y - 100*sub.ci_low.to_numpy(), 100*sub.ci_high.to_numpy() - y])
        ax.errorbar(x + offset, y, yerr=err, color=COLORS[method], marker=MARKERS[method],
                    linestyle=LINESTYLES[method], linewidth=1.7, capsize=2.5, label=LABELS[method])
    ax.set(xticks=x, xticklabels=STAGES, ylim=(0,100), yticks=np.arange(0,101,20),
           xlabel="Zielwortzahl entlang der rekursiven Kette", ylabel="QA-Retention unter gültigen Urteilen (%)",
           title="Faktenretention über die rekursive Kompression")
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=.24)
    fig.text(.12, .02, f"n = 100 Artikel je Methode/Stufe · 20 geplante Fakten je Artikel · {qa.technical_missing.sum()} QA-Fehlwerte separat\n"
             "Punktweise 95-%-Artikelbootstrap-Intervalle; 10.000 Ziehungen, Seed 32 · 2 leere Ausgaben eingeschlossen", fontsize=9)
    savefig(fig, "01_retentionsverlauf")

    TRANSITIONS = [(500,300), (300,150), (150,75), (500,75)]
    change_rows = []
    for method in METHODS:
        wide = article[article.method.eq(method)].pivot(index="article_id", columns="stage_target_words", values="retention_observed")
        for before, after in TRANSITIONS:
            diff = 100 * (wide[after] - wide[before])
            change_rows.append({"method": method, "transition": f"{before}→{after}", "from_stage": before,
                "to_stage": after, **estimate(diff), "n_increase": int(diff.gt(1e-12).sum()),
                "n_equal": int(diff.abs().le(1e-12).sum()), "n_decrease": int(diff.lt(-1e-12).sum())})
    changes = pd.DataFrame(change_rows)
    export(changes, "stufenaenderungen_pp_95ci")
    display(changes)
    adjacent = changes[changes.transition.ne("500→75")]
    steepest = adjacent.loc[adjacent.groupby("method")["mean"].idxmin(), ["method", "transition", "mean", "ci_low", "ci_high", "n_articles"]]
    display(Markdown("**Größter mittlerer Rückgang je Methode (deskriptive Auswahl):**")); display(steepest)

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
    contrasts = contrasts_for(article)
    assert len(contrasts) == 16 and contrasts.n_articles.eq(100).all()
    export(contrasts, "vier_gepaarte_kontraste_pp_95ci")
    contrast_table = contrasts.assign(Ergebnis=[f"{fmt(r.mean)} [{fmt(r.ci_low)}; {fmt(r.ci_high)}]; n={r.n_articles}" for r in contrasts.itertuples()])
    contrast_table = contrast_table.pivot(index="contrast", columns="stage_target_words", values="Ergebnis").reindex(
        index=[f"{LABELS[a]} − {LABELS[b]}" for a,b in CONTRASTS], columns=STAGES)
    display(contrast_table)
    contrast_table.to_csv(OUT / "tabelle_kontraste_formatiert.csv")

    fig, axes = plt.subplots(2,2,figsize=(10,8),sharey=True,sharex=True)
    for ax,(first,second) in zip(axes.flat,CONTRASTS):
        sub=contrasts[(contrasts["first"]==first)&(contrasts.second==second)].set_index("stage_target_words").loc[STAGES]
        y=np.arange(4)
        ax.errorbar(sub["mean"],y,xerr=np.vstack([sub["mean"]-sub.ci_low,sub.ci_high-sub["mean"]]),
                    fmt="o",color=COLORS[first],capsize=4,markersize=6)
        ax.axvline(0,color="#777777",linestyle="--",linewidth=1)
        ax.set_title(f"{LABELS[first]}\n− {LABELS[second]}",fontsize=10)
        ax.set_xlabel("Differenz (Prozentpunkte)");ax.grid(axis="x",alpha=.15)
    for ax in axes[:,0]:
        ax.set(yticks=np.arange(4),yticklabels=STAGES,ylabel="Zielwortzahl")
    axes[0,0].invert_yaxis()
    fig.suptitle("Gepaarte Methodenunterschiede",fontsize=13)
    fig.text(.5,.02,"Punktweise 95-%-Artikelbootstrap-Intervalle · Positive Werte begünstigen die erstgenannte Methode",fontsize=9,ha="center")
    fig.tight_layout(rect=(0,.06,1,.95));savefig(fig,"02_gepaarte_kontraste")

    sensitivity_means = pd.concat([means,
        means_for(article,"retention_lower","all_missing_zero"),
        means_for(article,"retention_upper","all_missing_one"),
        means_for(article[article.article_id.isin(complete_ids)],policy="complete_panel"),
        means_for(article[~article.article_id.isin(failed_ids)],policy="no_generation_failure")],ignore_index=True)
    sensitivity = pd.concat([contrasts,
        contrasts_for(article,"retention_lower","all_missing_zero"),
        contrasts_for(article,"retention_upper","all_missing_one"),
        contrasts_for(article,policy="complete_pairs",pair_complete=True),
        contrasts_for(article[article.article_id.isin(complete_ids)],policy="complete_panel"),
        contrasts_for(article[~article.article_id.isin(failed_ids)],policy="no_generation_failure")],ignore_index=True)
    missing_bounds=[]
    for first,second in CONTRASTS:
        for stage in STAGES:
            sub=article[article.stage_target_words.eq(stage)]
            lo=sub.pivot(index="article_id",columns="method",values="retention_lower")
            hi=sub.pivot(index="article_id",columns="method",values="retention_upper")
            missing_bounds.append({"contrast":f"{LABELS[first]} − {LABELS[second]}","stage_target_words":stage,
                "lower_missing_bound_pp":100*(lo[first]-hi[second]).mean(),
                "upper_missing_bound_pp":100*(hi[first]-lo[second]).mean(),"n_articles":len(lo)})
    missing_bounds=pd.DataFrame(missing_bounds)
    export(sensitivity_means,"sensitivitaet_mittelwerte")
    export(sensitivity,"sensitivitaet_kontraste")
    export(missing_bounds,"gegenlaeufige_missingness_grenzen")
    display(sensitivity.pivot(index=["contrast","stage_target_words"],columns="policy",values="mean"))
    display(missing_bounds)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    policies=[("observed","Hauptauswertung","o"),("all_missing_zero","Fehlend = 0","s"),
              ("all_missing_one","Fehlend = 1","^"),("complete_pairs","Vollständige Paare","D"),
              ("complete_panel",f"Vollständiger Panel (n={len(complete_ids)})","v"),
              ("no_generation_failure",f"Ohne leere Ketten (n={len(IDS)-len(failed_ids)})","P")]
    for ax,(first,second) in zip(axes.flat,CONTRASTS):
        for j,(policy,label,marker) in enumerate(policies):
            sub=sensitivity[(sensitivity["first"]==first)&(sensitivity.second==second)&(sensitivity.policy==policy)].set_index("stage_target_words").loc[STAGES]
            ax.plot(np.arange(4),sub["mean"],marker=marker,linewidth=1.1,label=label,alpha=.9)
        ax.axhline(0,color="#777777",linestyle="--",linewidth=1)
        ax.set_title(f"{LABELS[first]} − {LABELS[second]}")
        ax.set(xticks=np.arange(4),xticklabels=STAGES,xlabel="Zielwortzahl",ylabel="Mittlere Differenz (pp)")
        ax.grid(axis="y",alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,bbox_to_anchor=(.5,-.04),frameon=False,fontsize=9)
    fig.suptitle("Missingness-Sensitivität der gepaarten Punktschätzer")
    fig.tight_layout();savefig(fig,"03_missingness_sensitivitaet")

    fig,axes=plt.subplots(2,2,figsize=(11,9),sharey=True)
    jitter=np.random.default_rng(SEED).uniform(-.12,.12,size=len(IDS))
    for ax,stage in zip(axes.flat,STAGES):
        arrays=[100*article[(article.method==m)&(article.stage_target_words==stage)].sort_values("article_id").retention_observed.to_numpy() for m in METHODS]
        boxes=ax.boxplot(arrays,positions=np.arange(6),widths=.5,patch_artist=True,showfliers=False,
                         medianprops={"color":"black","linewidth":1.2})
        for i,(method,values) in enumerate(zip(METHODS,arrays)):
            boxes["boxes"][i].set_facecolor(COLORS[method]);boxes["boxes"][i].set_alpha(.25)
            ax.scatter(i+jitter,values,s=8,color=COLORS[method],alpha=.35,edgecolors="none")
        ax.set(title=f"{stage} Wörter",xticks=np.arange(6),xticklabels=[LABELS[m] for m in METHODS],ylim=(0,100))
        ax.tick_params(axis="x",labelrotation=75,labelsize=8);ax.grid(axis="y",alpha=.15)
    for ax in axes[:,0]:
        ax.set_ylabel("Artikelweise QA-Retention (%)")
    fig.suptitle("Verteilung der Retention auf den 100 Artikeln")
    fig.tight_layout();savefig(fig,"04_artikelverteilungen")

    transition_rows=[];examples=[]
    for before,after in TRANSITIONS[:3]:
        left=qa[qa.stage_target_words.eq(before)]
        right=qa[qa.stage_target_words.eq(after)]
        paired=left.merge(right,on=["article_id","method","canonical_fact_id"],suffixes=("_before","_after"),validate="one_to_one")
        paired["both_valid"]=paired.analysis_eligible_before & paired.analysis_eligible_after & ~paired.chain_failure_before & ~paired.chain_failure_after
        paired["upward"]=paired.both_valid & ~paired.retained_before & paired.retained_after
        paired["downward"]=paired.both_valid & paired.retained_before & ~paired.retained_after
        for method,sub in paired.groupby("method"):
            upward=sub[sub.upward]
            transition_rows.append({"method":method,"transition":f"{before}→{after}","n_fact_pairs":len(sub),
                "n_valid_fact_pairs":int(sub.both_valid.sum()),
                "n_generation_affected_pairs":int((sub.chain_failure_before | sub.chain_failure_after).sum()),"n_upward_label_changes":len(upward),
                "n_downward_label_changes":int(sub.downward.sum()),
                "n_unique_article_facts_upward":len(upward[["article_id","canonical_fact_id"]].drop_duplicates()),
                "n_articles_upward":upward.article_id.nunique()})
        upward=paired[paired.upward].copy();upward["transition"]=f"{before}→{after}"
        examples.append(upward[["article_id","method","canonical_fact_id","transition","question_before","reference_answer_before",
            "label_before","label_after","answer_before","answer_after","support_before","support_after","summary_before","summary_after"]])
    transition_audit=pd.DataFrame(transition_rows)
    upward_cases=pd.concat(examples,ignore_index=True)
    export(transition_audit,"qa_labelwechsel_ueber_stufen")
    export(upward_cases,"qa_labelwechsel_einzelfaelle")
    display(transition_audit)
    display(Markdown(f"**Anstiege auf Faktenebene:** {len(upward_cases)} Übergangspositionen betreffen "
        f"{len(upward_cases[['article_id','canonical_fact_id']].drop_duplicates())} unterschiedliche Originalfakten "
        f"in {upward_cases.article_id.nunique()} Artikeln. Mehrere Methoden/Stufen können dasselbe Faktum betreffen."))
    display(upward_cases.sort_values(["article_id","canonical_fact_id","method","transition"]).head(10).drop(columns=["summary_before","summary_after"]))

    version_means=[];version_contrasts=[]
    for version in ["original_frozen","aktuell_20260930"]:
        cache=read("article_archived.csv.gz",version).rename(columns={"retention":"retention_observed"})
        cache["retention_observed"]=pd.to_numeric(cache.retention_observed)
        version_means.append(means_for(cache).assign(version=version))
        version_contrasts.append(contrasts_for(cache).assign(version=version))
    version_means=pd.concat(version_means,ignore_index=True)
    version_contrasts=pd.concat(version_contrasts,ignore_index=True)
    export(version_means,"versionsvergleich_mittelwerte")
    export(version_contrasts,"versionsvergleich_kontraste")
    display(version_means.pivot(index=["method","stage_target_words"],columns="version",values="mean").mul(100))
    old_summary=read("summaries.csv.gz","original_frozen").set_index(["article_id","method"])
    new_summary=read("summaries.csv.gz","aktuell_20260930").set_index(["article_id","method"])
    old_qa=read("answers.csv.gz","original_frozen").rename(columns={"summary_prompt_variant":"method"}).set_index(KEY).sort_index()
    new_qa=read("answers.csv.gz","aktuell_20260930").rename(columns={"summary_prompt_variant":"method"}).set_index(KEY).sort_index()
    for method in ["lead_k","zero_shot_cot","auto_cot","automate_cot"]:
        for stage in STAGES:
            assert old_summary.xs(method,level="method")[f"summary_{stage}"].equals(new_summary.xs(method,level="method")[f"summary_{stage}"])
        assert old_qa.xs(method,level="method").equals(new_qa.xs(method,level="method"))
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharey=True)
    for ax,method in zip(axes,["plain","few_shot_cot"]):
        for version,label,style in [("original_frozen","Ursprünglich eingefroren","--"),("aktuell_20260930","Neuer Lauf vom 30. September","-")]:
            sub=version_means[(version_means.method==method)&(version_means.version==version)].set_index("stage_target_words").loc[STAGES]
            ax.plot(np.arange(4),100*sub["mean"],marker="o",linestyle=style,label=label)
        ax.set(title=LABELS[method],xticks=np.arange(4),xticklabels=STAGES,ylim=(0,100),xlabel="Zielwortzahl")
        ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("QA-Retention unter gültigen Urteilen (%)")
    axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle("Versionsvergleich der beiden ersetzten Methoden")
    fig.tight_layout();savefig(fig,"05_versionsvergleich")

    from itertools import combinations
    PAIRWISE_LABELS = ["retained", "lost", "reference_discordant"]
    AUDIT_LABELS = PAIRWISE_LABELS + ["technical_missing", "generation_failure"]
    METHOD_PAIRS = list(combinations(METHODS, 2))
    assert len(METHOD_PAIRS) == 15
    PAIR_KEY = ["article_id", "canonical_fact_id", "stage_target_words"]
    pairwise_matrices = {}
    pairwise_audit_matrices = {}
    pairwise_records = []
    count_records = []
    audit_records = []

    for first, second in METHOD_PAIRS:
        columns = PAIR_KEY + ["label", "technical_missing", "chain_failure"]
        left = qa.loc[qa.method.eq(first), columns]
        right = qa.loc[qa.method.eq(second), columns]
        matched = left.merge(right, on=PAIR_KEY, suffixes=("_first", "_second"),
                             validate="one_to_one", indicator=True)
        assert len(matched) == len(IDS) * N_FACTS * len(STAGES)
        assert matched["_merge"].eq("both").all()
        for stage in STAGES + ["all_stages"]:
            paired = matched if stage == "all_stages" else matched[matched.stage_target_words.eq(stage)]
            missing = paired.technical_missing_first | paired.technical_missing_second
            generation = paired.chain_failure_first | paired.chain_failure_second
            valid = ~(missing | generation)
            subset = paired[valid]
            matrix = pd.crosstab(subset.label_first, subset.label_second).reindex(
                index=PAIRWISE_LABELS, columns=PAIRWISE_LABELS, fill_value=0).astype(int)
            assert matrix.to_numpy().sum() == len(subset)
            assert np.array_equal(matrix.sum(axis=1).to_numpy(),
                subset.label_first.value_counts().reindex(PAIRWISE_LABELS, fill_value=0).to_numpy())
            assert np.array_equal(matrix.sum(axis=0).to_numpy(),
                subset.label_second.value_counts().reindex(PAIRWISE_LABELS, fill_value=0).to_numpy())
            reverse = pd.crosstab(subset.label_second, subset.label_first).reindex(
                index=PAIRWISE_LABELS, columns=PAIRWISE_LABELS, fill_value=0).astype(int)
            assert np.array_equal(matrix.to_numpy().T, reverse.to_numpy())
            pairwise_matrices[(first, second, stage)] = matrix
            same = int(np.trace(matrix.to_numpy()))
            record = {"first": first, "second": second, "stage": stage,
                "n_expected": len(paired), "n_valid_pairs": len(subset),
                "n_articles": subset.article_id.nunique(),
                "n_excluded_union": int((missing | generation).sum()),
                "n_technical_missing_either": int(missing.sum()),
                "n_generation_failure_either": int(generation.sum()),
                "n_both_failure_types": int((missing & generation).sum()),
                "n_same_label": same, "n_different_label": len(subset)-same,
                "agreement_percent": 100*same/len(subset) if len(subset) else np.nan}
            assert record["n_valid_pairs"] + record["n_excluded_union"] == record["n_expected"]
            pairwise_records.append(record)
            for row_label in PAIRWISE_LABELS:
                row_total = int(matrix.loc[row_label].sum())
                for col_label in PAIRWISE_LABELS:
                    n = int(matrix.loc[row_label, col_label])
                    count_records.append({"first": first, "second": second, "stage": stage,
                        "row_label": row_label, "column_label": col_label, "n": n,
                        "row_total": row_total, "row_percent": 100*n/row_total if row_total else np.nan,
                        "n_valid_pairs": len(subset)})
            audited_first = np.where(paired.chain_failure_first, "generation_failure", paired.label_first)
            audited_second = np.where(paired.chain_failure_second, "generation_failure", paired.label_second)
            full_matrix = pd.crosstab(pd.Series(audited_first), pd.Series(audited_second)).reindex(
                index=AUDIT_LABELS, columns=AUDIT_LABELS, fill_value=0).astype(int)
            assert full_matrix.to_numpy().sum() == len(paired)
            pairwise_audit_matrices[(first, second, stage)] = full_matrix
            for row_label in AUDIT_LABELS:
                for col_label in AUDIT_LABELS:
                    audit_records.append({"first": first, "second": second, "stage": stage,
                        "row_label": row_label, "column_label": col_label,
                        "n": int(full_matrix.loc[row_label, col_label]), "n_expected": len(paired)})

    pairwise_summary = pd.DataFrame(pairwise_records)
    pairwise_counts = pd.DataFrame(count_records)
    pairwise_audit_counts = pd.DataFrame(audit_records)
    assert len(pairwise_summary) == 15 * 5
    assert len(pairwise_counts) == 15 * 5 * 9
    assert len(pairwise_audit_counts) == 15 * 5 * 25
    for first, second in METHOD_PAIRS:
        assert np.array_equal(pairwise_matrices[(first, second, "all_stages")].to_numpy(),
            sum(pairwise_matrices[(first, second, stage)].to_numpy() for stage in STAGES))
    export(pairwise_summary, "paarweise_labeluebereinstimmung_und_nenner")
    export(pairwise_counts, "paarweise_konfusionsmatrizen_3x3")
    export(pairwise_audit_counts, "paarweise_auditmatrizen_5x5")
    summary_display = pairwise_summary[pairwise_summary.stage.eq("all_stages")].copy()
    summary_display["Vergleich"] = summary_display["first"].map(LABELS) + " × " + summary_display.second.map(LABELS)
    display(summary_display[["Vergleich", "n_valid_pairs", "n_excluded_union", "n_technical_missing_either",
                             "n_generation_failure_either", "agreement_percent"]].round(1))

    from matplotlib.colors import Normalize
    PAIRWISE_CMAP = plt.colormaps["Blues"].copy()
    PAIRWISE_CMAP.set_bad("#eeeeee")
    method_ticks = ["Lead-k", "Plain", "Zero-Shot\nCoT", "Few-Shot\nCoT", "Auto-CoT", "Automate-CoT"]
    agreement_matrices = {}
    for stage in STAGES + ["all_stages"]:
        values = pd.DataFrame(np.nan, index=METHODS, columns=METHODS)
        for record in pairwise_summary[pairwise_summary.stage.eq(stage)].itertuples(index=False):
            values.loc[record.first, record.second] = record.agreement_percent
            values.loc[record.second, record.first] = record.agreement_percent
        assert np.allclose(values.to_numpy(), values.to_numpy().T, equal_nan=True)
        agreement_matrices[stage] = values
        values.rename(index=LABELS, columns=LABELS).to_csv(OUT / f"labeluebereinstimmung_6x6_{stage}.csv")

    def plot_agreement(ax, stage):
        values = agreement_matrices[stage].to_numpy()
        img = ax.imshow(np.ma.masked_invalid(values), cmap=PAIRWISE_CMAP, vmin=0, vmax=100)
        for row in range(6):
            for col in range(6):
                v = values[row, col]
                ax.text(col, row, "—" if np.isnan(v) else fmt(v), ha="center", va="center",
                        fontsize=9, color="white" if np.isfinite(v) and v >= 60 else "#222222")
        ax.set(xticks=np.arange(6), yticks=np.arange(6), xticklabels=method_ticks, yticklabels=method_ticks)
        ax.tick_params(axis="both", labelsize=9, length=0)
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title("Alle vier Stufen (deskriptiv gepoolt)" if stage == "all_stages" else f"{stage} Zielwörter")
        return img

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.5), layout="constrained")
    for ax, stage in zip(axes.flat, STAGES):
        img = plot_agreement(ax, stage)
    fig.colorbar(img, ax=axes.ravel().tolist(), shrink=.65, label="Gleiche automatische Labels (%)")
    fig.suptitle("Paarweise Labelübereinstimmung der sechs Verfahren", fontsize=14)
    savefig(fig, "06_labeluebereinstimmung_6x6_stufen")

    fig, ax = plt.subplots(figsize=(8.5, 7), layout="constrained")
    img = plot_agreement(ax, "all_stages")
    fig.colorbar(img, ax=ax, shrink=.8, label="Gleiche automatische Labels (%)")
    fig.suptitle("Paarweise Labelübereinstimmung über alle Stufen", fontsize=13)
    savefig(fig, "07_labeluebereinstimmung_6x6_gepoolt")

    def plot_confusion_stage(stage):
        fig, axes = plt.subplots(3, 5, figsize=(17.5, 10.5), layout="constrained")
        for ax, (first, second) in zip(axes.flat, METHOD_PAIRS):
            matrix = pairwise_matrices[(first, second, stage)].to_numpy()
            totals = matrix.sum(axis=1, keepdims=True)
            percentages = np.divide(100*matrix, totals,
                out=np.full(matrix.shape, np.nan, dtype=float), where=totals > 0)
            img = ax.imshow(np.ma.masked_invalid(percentages), cmap=PAIRWISE_CMAP, vmin=0, vmax=100)
            for row in range(3):
                for col in range(3):
                    pct = percentages[row, col]
                    label = f"{matrix[row,col]}\n({fmt(pct)} %)" if np.isfinite(pct) else "–"
                    ax.text(col, row, label, ha="center", va="center", fontsize=9,
                            color="white" if np.isfinite(pct) and pct >= 60 else "#222222")
            rec = pairwise_summary[(pairwise_summary["first"] == first) &
                (pairwise_summary.second == second) & (pairwise_summary.stage == stage)].iloc[0]
            ax.set_title(f"Zeile: {LABELS[first]}\nSpalte: {LABELS[second]}\n"
                         f"n = {rec.n_valid_pairs}; ausgeschlossen = {rec.n_excluded_union}", fontsize=9)
            ax.set(xticks=np.arange(3), yticks=np.arange(3), xticklabels=["R","L","D"], yticklabels=["R","L","D"])
            ax.tick_params(length=0, labelsize=10)
            ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(img, ax=axes.ravel().tolist(), shrink=.6, label="Anteil innerhalb der Zeile (%)")
        fig.suptitle(f"15 paarweise Konfusionsmatrizen bei {stage} Zielwörtern\n"
                     "R = retained · L = lost · D = Referenzabweichung · Fallzahl und Zeilenprozent", fontsize=13)
        return fig

    savefig(plot_confusion_stage(500), "08_konfusionsmatrizen_500")

    savefig(plot_confusion_stage(300), "08_konfusionsmatrizen_300")

    savefig(plot_confusion_stage(150), "08_konfusionsmatrizen_150")

    savefig(plot_confusion_stage(75), "08_konfusionsmatrizen_75")

    row_lookup = {label: i for i, label in enumerate(PAIRWISE_LABELS)}
    composition = []
    for stage in STAGES:
        for first, second in METHOD_PAIRS:
            matrix = pairwise_matrices[(first, second, stage)].to_numpy()
            diagonal = np.diag(matrix)
            composition.append({"first": first, "second": second, "stage": stage,
                "n_both_retained": int(diagonal[row_lookup["retained"]]),
                "n_both_lost": int(diagonal[row_lookup["lost"]]),
                "n_both_reference_discordant": int(diagonal[row_lookup["reference_discordant"]]),
                "n_valid_pairs": int(matrix.sum())})
    export(pd.DataFrame(composition), "zusammensetzung_paarweiser_labeluebereinstimmung")
    for stage in STAGES:
        records = pairwise_summary[pairwise_summary.stage.eq(stage)]
        highest = records.loc[records.agreement_percent.idxmax()]
        lowest = records.loc[records.agreement_percent.idxmin()]
        display(Markdown(f"**{stage} Zielwörter:** Höchste rohe Labelübereinstimmung: "
            f"{LABELS[highest['first']]} × {LABELS[highest.second]} mit {fmt(highest.agreement_percent)} % "
            f"(n = {highest.n_valid_pairs}); niedrigste: {LABELS[lowest['first']]} × {LABELS[lowest.second]} "
            f"mit {fmt(lowest.agreement_percent)} % (n = {lowest.n_valid_pairs}). "
            "Die Auswahl der höchsten/niedrigsten Quote ist deskriptiv."))

    mlookup=means.set_index(["method","stage_target_words"])
    clookup=changes.set_index(["method","transition"])
    def mean_text(method,stage):
        r=mlookup.loc[(method,stage)]
        return f"{fmt(100*r['mean'])} % (95-%-KI [{fmt(100*r.ci_low)}; {fmt(100*r.ci_high)}])"

    version_text=("Verwendet wurde der am 1. Oktober 2026 abgeschlossene Ergebnisstand, der die ursprünglichen Ergebnisse von Lead-k, Zero-Shot-CoT, Auto-CoT und Automate-CoT mit neuen Plain- und Few-shot-CoT-Ketten vom 30. September verbindet. Die Ersetzung erfolgte nach dem ursprünglichen Protokoll-Freeze; der ursprüngliche Lauf wird daher als separate Versionssensitivität ausgewiesen."
        if VERSION=="aktuell_20260930" else "Verwendet wurde der Ergebnisstand des ursprünglich eingefrorenen Generierungslaufs. Der spätere Ergebnisstand mit ersetzten Plain- und Few-shot-Ketten wird separat als Versionssensitivität ausgewiesen.")
    text=["# 4.2 QA-basierte Faktenretention",version_text,
        f"Die Analyse umfasste 100 Artikel, sechs Verfahren und vier rekursive Zielstufen von 500, 300, 150 und 75 Wörtern. Je Artikel wurden dieselben 20 kanonischen Originalfakten verwendet. Alle 2.400 Zusammenfassungspositionen und 48.000 vorgesehenen QA-Positionen lagen vor. Davon waren {qa.analysis_eligible.sum():,} exportierte QA-Positionen auswertbar und {qa.technical_missing.sum()} aufgrund fehlender maschinell prüfbarer Textbelege technisch nicht auswertbar. Technisch fehlende QA-Urteile erhielten kein inhaltliches Verlustlabel. Zusätzlich gab es zwei leere Zero-shot-CoT-Ausgaben. Dafür wurden insgesamt 40 NOT_FOUND-Positionen regelbasiert ohne QA-Modellantwort eingetragen; sie sind in der reproduzierten End-to-end-Auswertung enthalten und werden als Generierungsfehler separat ausgewiesen. Eine der beiden Ketten wurde anschließend mit leerer Eingabe fortgesetzt. Die Sensitivität ohne beide betroffenen Artikel erhält 98 Artikel für alle Methoden. Die primäre Retention wurde entsprechend der implementierten Auswertung zunächst pro Artikel als Anteil retained unter den gültigen Urteilen und anschließend gleichgewichtet über die Artikel berechnet. Je Artikel-Methoden-Stufe standen {article.n_valid.min()} bis {article.n_valid.max()} gültige Urteile zur Verfügung. Ergänzend wurden Grenzen mit festem Nenner von 20 Fakten berechnet.",
        "Unsicherheitsintervalle beruhen auf 10.000 Bootstrap-Stichproben auf Artikelebene mit Seed 32. Bei Methoden- und Stufenvergleichen wurde die Paarung derselben Artikel erhalten. Die Intervalle sind punktweise 95-%-Perzentilintervalle; sie beschreiben Artikelheterogenität bei fest vorliegenden automatischen Labels.",
        "## 4.2.1 Retentionsverlauf über die Kompressionsstufen",
        "Die gemessene Retention nahm bei allen sechs Verfahren im Mittel von 500 auf 75 Zielwörter ab. Abbildung 4.X zeigt die Verläufe auf einer gemeinsamen Skala von 0 bis 100 %. Die Stufen beziehen sich auf Zielwortzahlen; ihre tatsächliche Einhaltung wird im folgenden Abschnitt getrennt untersucht."]

    if VERSION == "aktuell_20260930":
        text.append(f"Lead-k lag bei 500 Zielwörtern mit {mean_text('lead_k',500)} am höchsten, fiel jedoch bis 75 Wörter auf {mean_text('lead_k',75)}. Die gesamte gepaarte Änderung betrug {fmt(clookup.loc[('lead_k','500→75'),'mean'])} Prozentpunkte. Unter den generativen Verfahren lag Automate-CoT bei 500 Wörtern mit {mean_text('automate_cot',500)} am höchsten. Bei 75 Wörtern erreichte Auto-CoT den höchsten Mittelwert ({mean_text('auto_cot',75)}), gefolgt von Automate-CoT ({mean_text('automate_cot',75)}), Few-Shot-CoT ({mean_text('few_shot_cot',75)}), Plain ({mean_text('plain',75)}) und Zero-Shot-CoT ({mean_text('zero_shot_cot',75)}). Diese Rangfolge ist deskriptiv; ihre Unsicherheit wird durch gepaarte Kontraste untersucht.")
        text.append("Bei allen fünf generativen Verfahren trat der größte mittlere Rückgang im letzten Übergang von 150 auf 75 Zielwörter auf. Er reichte von "
            f"{fmt(-clookup.loc[('plain','150→75'),'mean'])} Prozentpunkten bei Plain bis {fmt(-clookup.loc[('zero_shot_cot','150→75'),'mean'])} Prozentpunkten bei Zero-Shot-CoT. Lead-k zeigte seinen stärksten Rückgang bereits bei 500→300 Wörtern "
            f"({fmt(-clookup.loc[('lead_k','500→300'),'mean'])} Prozentpunkte). Plain wies insgesamt den geringsten absoluten Rückgang auf "
            f"({fmt(-clookup.loc[('plain','500→75'),'mean'])} Prozentpunkte), begann aber auch mit dem niedrigsten Ausgangsniveau. Ein geringerer Rückgang allein bedeutet deshalb keine höhere Retention.")
    else:
        for method in METHODS:
            delta=clookup.loc[(method,"500→75")]
            largest=steepest[steepest.method.eq(method)].iloc[0]
            text.append(f"{LABELS[method]} erreichte bei 500 Wörtern {mean_text(method,500)} und bei 75 Wörtern {mean_text(method,75)}. Die gepaarte Änderung betrug {fmt(delta['mean'])} Prozentpunkte (95-%-KI [{fmt(delta.ci_low)}; {fmt(delta.ci_high)}]). Der größte mittlere Rückgang trat bei {largest.transition} Wörtern auf ({fmt(largest['mean'])} Prozentpunkte).")
    text += ["Die Position des stärksten Rückgangs wird deskriptiv aus drei Stufenübergängen gewählt und ist kein gesondert getesteter Unterschied zwischen Rückgängen. Spätere Anstiege einzelner Artikel oder Fakten wurden als QA-Labelwechsel erfasst; aus ihnen lässt sich ohne Textprüfung keine tatsächliche Wiedergewinnung von Information ableiten.",
        "## 4.2.2 Gepaarter Vergleich der Verfahren",
        "Tabelle 4.X berichtet die vier in den Auswertungsvorgaben genannten Kontraste für jede Zielstufe. Positive Werte entsprechen einer höheren Retention der erstgenannten Methode. Lead-k wird als zusätzliche deskriptive Baseline gezeigt und gehört nicht zu diesen vier Kontrasten."]
    for first,second in CONTRASTS:
        sub=contrasts[(contrasts['first']==first)&(contrasts.second==second)].set_index("stage_target_words").loc[STAGES]
        vals="; ".join(f"{stage} Wörter: {fmt(r['mean'])} pp [{fmt(r.ci_low)}; {fmt(r.ci_high)}]" for stage,r in sub.iterrows())
        text.append(f"Für {LABELS[first]} − {LABELS[second]} ergaben sich folgende gepaarte Differenzen mit 95-%-Intervallen: {vals}. Jeder Vergleich umfasste 100 Artikelpaare.")
    if VERSION == "aktuell_20260930":
        text.append("Damit wechselte Zero-Shot-CoT gegenüber Plain von einem positiven Unterschied bei den drei längeren Zielstufen zu einem negativen Unterschied bei 75 Wörtern. Few-Shot-CoT lag dort nur geringfügig über Plain, und sein Intervall umfasste 0. Auto-CoT lag in allen vier Zielstufen über Few-Shot-CoT. Automate-CoT zeigte gegenüber Auto-CoT keinen durchgehend positiven Unterschied; bei 150 Wörtern war der Mittelwert niedriger. Diese Befunde sprechen für stufenabhängige Unterschiede und nicht für einen einheitlichen Vorteil sämtlicher CoT-Verfahren.")
    max_mean_shift=100*(sensitivity_means[sensitivity_means.policy.eq("all_missing_zero")].set_index(["method","stage_target_words"])["mean"]-mlookup["mean"]).abs().max()
    sensitivity_comment = ("Im vollständigen Panel wechselte der kleine Few-Shot-CoT−Plain-Unterschied bei 75 Wörtern das Vorzeichen. Der Ausschluss der beiden Generierungsfehler änderte die Richtung der vier Kontraste nicht." if VERSION == "aktuell_20260930" else "Die Ausschlussanalysen werden mit ihren jeweiligen Artikelzahlen separat berichtet.")
    text += [f"Die konservative Setzung fehlender Urteile auf 0 veränderte die mittlere Retention gegenüber der Hauptauswertung um höchstens {fmt(max_mean_shift)} Prozentpunkte. Zusätzlich wurden gegenläufige Missingness-Grenzen, vollständig bewertete Paare sowie ein gemeinsamer vollständiger Panel von {len(complete_ids)} Artikeln ausgewertet. {sensitivity_comment} Der vollständige Panel ändert die Artikelbasis und kann daher selektiv sein; die entsprechenden Ergebnisse und Artikelzahlen stehen im Anhang.",
        "Die 16 dargestellten Intervalle sind nicht für Mehrfachvergleiche korrigiert. Da das eingefrorene Protokoll keine formale Entscheidungsregel für H1 enthält, werden die Methodenunterschiede als Effektgrößen mit Unsicherheit berichtet und nicht als nachträglich definierte Bestätigung der Hypothese. Höhere QA-Retention bei derselben Zielwortzahl ist außerdem ohne Berücksichtigung der tatsächlichen Textlänge keine vollständige Aussage über die Überlegenheit eines Verfahrens.",
        "Die Befunde beruhen auf der kalibrierten automatischen QA-Pipeline. Die historische Human-Validierung wurde zur Entwicklung dieser Bewertung verwendet und liefert keine Fehlerrate für den finalen Faktensatz. NOT_FOUND und Referenzabweichungen sind daher automatische Messurteile; insbesondere sind Referenzabweichungen kein direkter Nachweis von Halluzinationen der Zusammenfassung."]
    chapter="\n\n".join(text).replace("47,804", "47.804").replace("47,820", "47.820")
    table_lines=["| Vergleich | 500 Wörter | 300 Wörter | 150 Wörter | 75 Wörter |",
        "|---|---:|---:|---:|---:|"]
    for idx,row in contrast_table.iterrows():
        table_lines.append("| " + str(idx) + " | " + " | ".join(str(row[t]) for t in STAGES) + " |")
    chapter += "\n\n**Tabelle 4.X.** Gepaarte Differenzen in Prozentpunkten mit punktweisen 95-%-Artikelbootstrap-Intervallen und Artikelzahl.\n\n" + "\n".join(table_lines)
    chapter += "\n\n**Abbildung 4.X.** Retentionsverlauf der sechs Verfahren; n = 100 Artikel je Methode/Stufe. Technische QA-Fehlwerte werden aus dem primären Artikelnenner ausgeschlossen. Zwei leere Zero-shot-Ausgaben gehen gemäß implementierter End-to-end-Regel mit 0 ein; Sensitivitäten sind separat berichtet.\n"
    (OUT/"Kapitel_4_2_QA_Faktenretention.md").write_text(chapter,encoding="utf-8")
    display(Markdown(chapter))

    execution_manifest={"version":VERSION,"seed":SEED,"bootstrap_samples":B,"bootstrap_unit":"article",
        "interval":"pointwise percentile 95%","primary_retention":"retained / valid QA judgments per article cell",
        "n_articles":len(IDS),"n_facts":len(facts),"n_summary_positions":len(summary_long),
        "n_qa_positions":len(qa),"n_valid_qa":int(qa.analysis_eligible.sum()),
        "n_missing_qa":int(qa.technical_missing.sum()),"n_complete_panel":len(complete_ids),"n_qa_complete_panel":len(qa_complete_ids),
        "n_empty_outputs":int(state_audit.empty_output.sum()),"n_synthetic_not_found":int(qa.empty_output.sum()),
        "n_articles_without_generation_failure":len(IDS)-len(failed_ids),
        "planned_contrasts_source":"user-provided evaluation specification; temporal preregistration not verified",
        "source_manifest_sha256":hashlib.sha256((DATA/"source_manifest.json").read_bytes()).hexdigest(),
        "pairwise_confusion_matrices":{"n_method_pairs":len(METHOD_PAIRS),"n_stage_matrices":len(METHOD_PAIRS)*len(STAGES),
            "pair_key":PAIR_KEY,"labels":PAIRWISE_LABELS,"exclusion":"technical missing or generation chain failure on either side",
            "normalization":"row percentage; absolute counts also exported","human_reference":False},
        "software":{"python":platform.python_version(),**versions}}
    (OUT/"execution_manifest.json").write_text(json.dumps(execution_manifest,ensure_ascii=False,indent=2))
    qa[KEY+["dataset_index","question","reference_answer","answer","support","answer_status","error",
              "label","analysis_eligible","technical_missing","grounded","answer_origin","empty_output","chain_failure"]].to_csv(OUT/"qa_labels_reproduziert.csv.gz",index=False)
    files=pd.DataFrame([{"Datei":str(p.relative_to(BASE)),"Bytes":p.stat().st_size}
                        for folder in [OUT,FIG] for p in sorted(folder.iterdir()) if p.is_file()])
    display(files)
    display(Markdown("**Alle Schlüssel-, Lineage-, Beleg-, Scorer- und Aggregatprüfungen erfolgreich.**"))

if __name__ == '__main__':
    from offline_guard import install
    install(allow_statistics_workers='--recompute-regression-bootstrap' in sys.argv)
    main(sys.argv[1], '--recompute-regression-bootstrap' in sys.argv, int(sys.argv[2]))
