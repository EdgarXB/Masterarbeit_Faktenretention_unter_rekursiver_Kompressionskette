# Statische Originalauswertung; Eingaben/Ausgaben relativ zur übergebenen Arbeitskopie.
# Keine LLM-Aufrufe. Standard: vorhandene Regressionscaches verwenden.
from pathlib import Path
import os,sys

def main(work, recompute_bootstrap=False, workers=1):
    WORK=__import__('pathlib').Path(work).resolve()
    __import__('os').chdir(WORK/('supplement' if 'hauptexperiment_05_alle_texttabellen'[0].isdigit() else 'project/02_Auswertung'))
    from pathlib import Path
    from fractions import Fraction
    from collections import Counter
    import gzip, hashlib, importlib.util, importlib.metadata, json, platform, re
    from textwrap import fill
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr, rankdata

    BASE = next((p for p in [Path.cwd(), Path.cwd()/"Notebooks und Grafiken"]
                 if (p/"hauptexperiment_qa_daten/source_manifest.json").is_file()), None)
    if BASE is None:
        raise FileNotFoundError("Notebook im Ordner mit den mitgelieferten Daten ausführen.")
    VERSION, SEED, B = "aktuell_20260930", 32, 10_000
    METHODS = ["lead_k", "plain", "zero_shot_cot", "few_shot_cot", "auto_cot", "automate_cot"]
    NAMES = dict(zip(METHODS, ["Lead-k", "Plain", "Zero-Shot-CoT", "Few-Shot-CoT", "Auto-CoT", "Automate-CoT"]))
    STAGES = [500, 300, 150, 75]
    WINDOWS = {500:(475,525), 300:(285,315), 150:(142,158), 75:(71,79)}
    CELL = ["article_id", "method", "stage_target_words"]
    FACTKEY = ["article_id", "canonical_fact_id"]
    KEY = CELL + ["canonical_fact_id"]
    R, L, D_LABEL, M = "retained", "lost", "reference_discordant", "technical_missing"
    NON = {L, D_LABEL}
    QA_DATA = BASE/"hauptexperiment_qa_daten"
    MET_DATA = BASE/"hauptexperiment_metriken_daten"
    VAL_DATA = BASE/"human_validation_qa_daten"
    OUT = BASE/"hauptexperiment_texttabellen_ergebnisse"/VERSION
    OUT.mkdir(parents=True, exist_ok=True)
    TEXT_FILE = OUT/"alle_zehn_auswertungen.txt"
    TEXT_FILE.write_text("Finales Hauptexperiment – zehn Auswertungen\n"
                         "10.000 Artikelbootstrap-Ziehungen, Seed 32, 95-%-Perzentilintervalle.\n\n", encoding="utf-8")
    table_registry, sources, validation_checks = [], [], []

    def read(path):
        return pd.read_csv(path, keep_default_na=False, low_memory=False, float_precision="round_trip")

    def record_source(path, role, expected=None):
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None:
            assert digest == expected, f"Quellenhash verändert: {path}"
        rel = str(path.relative_to(BASE))
        if rel not in {x["Datei"] for x in sources}:
            sources.append({"Datei":rel, "SHA256":digest, "Rolle":role})

    def frozen_input(root, relative):
        manifest_path = root/"source_manifest.json"
        record_source(manifest_path, "Quellenmanifest")
        manifest = json.loads(manifest_path.read_text())
        entry = next(e for e in manifest["files"] if e["file"] == relative)
        path = root/relative
        record_source(path, "eingefrorene Eingabe", entry["sha256"])
        if path.suffix == ".gz" and "original_sha256" in entry:
            assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == entry["original_sha256"]
        return path

    def numeric(series):
        return pd.to_numeric(series.replace("", np.nan), errors="raise")

    def bool_values(series):
        assert series.astype(str).str.lower().isin(["true", "false"]).all()
        return series.astype(str).str.lower().eq("true")

    def table(frame, name, title, note, digits=4):
        # Print all rows as copyable text; retain full precision in the CSV export.
        assert name not in table_registry
        table_registry.append(name)
        frame.to_csv(OUT/f"{name}.csv", index=False, na_rep="fehlend")
        shown = frame.copy()
        for col in shown.select_dtypes(include="floating"):
            shown[col] = shown[col].map(lambda v: "fehlend" if pd.isna(v) else f"{v:.{digits}f}")
        shown = shown.fillna("fehlend")
        text = title + "\n" + shown.to_string(index=False, max_rows=None, max_cols=None) + "\n\n" + fill(note,width=105) + "\n"
        print(text)
        with TEXT_FILE.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")

    def percentile_interval(values):
        finite = np.asarray(values)[np.isfinite(values)]
        return np.quantile(finite, [.025,.975], method="linear") if len(finite) else np.array([np.nan,np.nan])

    facts = read(frozen_input(QA_DATA, "shared/canonical_facts.csv.gz"))
    summaries = read(frozen_input(QA_DATA, VERSION+"/summaries.csv.gz"))
    raw_qa = read(frozen_input(QA_DATA, VERSION+"/answers.csv.gz")).rename(columns={"summary_prompt_variant":"method"})
    states_path = frozen_input(QA_DATA, VERSION+"/generation_states.jsonl.gz")
    scorer_path = frozen_input(QA_DATA, "code/qa_scorer_snapshot.py")
    freeze_path = frozen_input(QA_DATA, "shared/FINAL_100_EXPERIMENT_FREEZE.md")
    merge_path = frozen_input(QA_DATA, VERSION+"/provenienz/result_merge_manifest.json")
    merge_info = json.loads(merge_path.read_text())
    assert merge_info["retained_historical_rows"] == 400
    assert merge_info["replaced_plain_rows"] == merge_info["replaced_few_shot_rows"] == 100
    assert merge_info["historical_plain_or_few_shot_rows_in_new_table"] == 0
    IDS = sorted(facts.article_id.unique())
    N = len(IDS)
    assert N == 100 and len(facts) == 2000 and facts.canonical_fact_id.is_unique
    assert facts.groupby("article_id").size().eq(20).all() and facts.entity_type.ne("").all()
    assert len(summaries) == 600 and not summaries.duplicated(["article_id","method"]).any()
    assert set(summaries[["article_id","method"]].itertuples(index=False,name=None)) == {(a,m) for a in IDS for m in METHODS}
    phase = {m: "plain_fewshot_rerun_20260930" if m in ["plain","few_shot_cot"]
             else "final_100_frozen_temperature_0" for m in METHODS}
    assert summaries.collection_phase.eq(summaries.method.map(phase)).all()
    expected_keys = {(a,m,t,f) for a,f in facts[FACTKEY].itertuples(index=False,name=None) for m in METHODS for t in STAGES}
    assert len(raw_qa) == 48000 and not raw_qa.duplicated(KEY).any()
    assert set(raw_qa[KEY].itertuples(index=False,name=None)) == expected_keys

    rows = []
    for t in STAGES:
        previous = "article" if t == 500 else f"summary_{STAGES[STAGES.index(t)-1]}"
        sub = summaries[["article_id","method",f"summary_{t}", previous,
                         f"summary_{t}_word_count",f"summary_{t}_complete_sentence_success",
                         f"summary_{t}_final_window_success"]].copy()
        sub.columns = ["article_id","method","summary","stage_input","W_stored","sentence_stored","window_stored"]
        sub["stage_target_words"] = t
        rows.append(sub)
    outputs = pd.concat(rows, ignore_index=True)
    outputs["W"] = outputs.summary.map(lambda s: len(re.findall(r"\b\w+\b", s)))
    outputs["U"] = outputs.stage_target_words.map(lambda t: WINDOWS[t][0])
    outputs["O"] = outputs.stage_target_words.map(lambda t: WINDOWS[t][1])
    outputs["violation_words"] = (outputs.U-outputs.W).clip(lower=0)+(outputs.W-outputs.O).clip(lower=0)
    outputs["D"] = outputs.violation_words/outputs.stage_target_words
    outputs["window_status"] = np.select([outputs.W.lt(outputs.U), outputs.W.gt(outputs.O)], ["unterhalb","oberhalb"], default="innerhalb")
    outputs["complete_sentence"] = bool_values(outputs.sentence_stored)
    outputs["empty_output"] = outputs.summary.str.strip().eq("")
    outputs["empty_input"] = outputs.stage_input.str.strip().eq("")
    assert outputs.W.eq(outputs.W_stored.astype(int)).all()
    assert bool_values(outputs.window_stored).eq(outputs.window_status.eq("innerhalb")).all()
    output_lookup = outputs.set_index(CELL)
    with gzip.open(states_path, "rt") as stream:
        states = [json.loads(line) for line in stream]
    assert len(states) == 600 and len({(s["article_id"],s["method"]) for s in states}) == 600
    for rec in states:
        for stage in rec["state"]["stages"]:
            row = output_lookup.loc[(rec["article_id"],rec["method"],stage["target_words"])]
            selected = stage["selected"]
            assert row.summary == selected["response"] and row.stage_input == stage["input_text"]
            assert row.W == selected["response_word_count"]
            assert row.complete_sentence == selected["ends_with_complete_sentence"]
            if "word_window_violation_rate" in selected:
                assert np.isclose(row.D, selected["word_window_violation_rate"], rtol=0, atol=1e-12)

    import qa_scorer_snapshot as scorer
    qa = raw_qa.merge(facts[FACTKEY+["entity_type","reference_answer"]], on=FACTKEY,
                      validate="many_to_one", suffixes=("","_canonical"))
    assert qa.entity_type.eq(qa.entity_type_canonical).all()
    assert qa.reference_answer.eq(qa.reference_answer_canonical).all()
    qa = qa.drop(columns=["entity_type_canonical","reference_answer_canonical"])
    qa = qa.merge(outputs[CELL+["summary"]], on=CELL, validate="many_to_one")
    qa["technical_missing"] = qa.error.ne("")
    qa["valid"] = ~qa.technical_missing
    assert qa.loc[qa.valid,"answer"].str.strip().ne("").all()
    assert all(scorer.support_is_grounded(s,a,e) for s,a,e in qa.loc[qa.valid,["summary","answer","support"]].itertuples(index=False,name=None))
    qa["is_retained"] = [scorer.retained(r,a) if valid else False
                         for r,a,valid in qa[["reference_answer","answer","valid"]].itertuples(index=False,name=None)]
    qa["label"] = np.select([qa.technical_missing, qa.is_retained, qa.answer.str.upper().eq("NOT_FOUND")], [M,R,L], default=D_LABEL)
    assert set(qa.label.unique()) == {R,L,D_LABEL,M}
    qa["is_lost"] = qa.label.eq(L)
    qa["is_discordant"] = qa.label.eq(D_LABEL)
    counts = qa.groupby(CELL).agg(K=("is_retained","sum"), V=("valid","sum"),
                                 n_missing=("technical_missing","sum"), n_lost=("is_lost","sum"),
                                 n_discordant=("is_discordant","sum"), n_expected=("label","size")).reset_index()
    assert counts.n_expected.eq(20).all()
    assert counts.K.add(counts.n_lost).add(counts.n_discordant).eq(counts.V).all()
    articles = outputs.merge(counts,on=CELL,validate="one_to_one")
    articles["R_obs"] = articles.K.div(articles.V.where(articles.V.gt(0)))
    articles["R_20"] = articles.K/20
    articles["R_adj"] = articles.R_20/(1+articles.D)
    assert (articles.R_adj <= articles.R_20).all()
    articles.drop(columns=["summary","stage_input"]).to_csv(OUT/"artikelwerte_laenge_qa.csv",index=False)
    qa[KEY+["entity_type","label"]].to_csv(OUT/"qa_labels.csv.gz",index=False)

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0,N,size=(B,N))
    weights = np.zeros((B,N),dtype=np.int16)
    np.add.at(weights,(np.repeat(np.arange(B),N),draws.ravel()),1)
    assert np.all(weights.sum(axis=1) == N)
    pass  # Große Bootstrap-Zwischendatei wird nicht benötigt.

    def vector(method, stage, metric):
        return articles.loc[articles.method.eq(method)&articles.stage_target_words.eq(stage)].set_index("article_id").reindex(IDS)[metric].to_numpy(dtype=float)

    def estimate_mean(values):
        values = np.asarray(values,dtype=float)
        valid = np.isfinite(values)
        numerator = weights @ np.where(valid,values,0.)
        denominator = weights @ valid.astype(float)
        boot = np.divide(numerator,denominator,out=np.full(B,np.nan),where=denominator>0)
        lo,hi = percentile_interval(boot)
        return {"mean":float(values[valid].mean()) if valid.any() else np.nan,
                "low":lo,"high":hi,"n":int(valid.sum()),"n_boot_valid":int(np.isfinite(boot).sum())},boot

    def estimate_ratio(frame, numerator, denominator):
        # Pooled count ratio; all facts/methods of each article resampled together.
        per_article = frame.groupby("article_id")[[numerator,denominator]].sum().reindex(IDS,fill_value=0)
        totals = per_article.sum()
        boot_counts = weights @ per_article.to_numpy(dtype=float)
        boot = np.divide(boot_counts[:,0],boot_counts[:,1],out=np.full(B,np.nan),where=boot_counts[:,1]>0)
        lo,hi = percentile_interval(boot)
        return {"count":int(totals.iloc[0]),"den":int(totals.iloc[1]),
                "mean":totals.iloc[0]/totals.iloc[1] if totals.iloc[1] else np.nan,
                "low":lo,"high":hi,"n_articles":int(per_article.iloc[:,1].gt(0).sum()),
                "n_boot_valid":int(np.isfinite(boot).sum())}

    retention_stats, exact_means, retention_boot = {}, {}, {}
    for m in METHODS:
        for t in STAGES:
            sub = articles.loc[articles.method.eq(m)&articles.stage_target_words.eq(t)]
            for metric in ["R_obs","R_20","R_adj"]:
                retention_stats[m,t,metric], retention_boot[m,t,metric] = estimate_mean(vector(m,t,metric))
                exact = []
                for row in sub.itertuples():
                    if metric == "R_obs" and row.V > 0: exact.append(Fraction(int(row.K),int(row.V)))
                    elif metric == "R_20": exact.append(Fraction(int(row.K),20))
                    elif metric == "R_adj": exact.append(Fraction(int(row.K)*int(t),20*(int(t)+int(row.violation_words))))
                exact_means[m,t,metric] = sum(exact,Fraction(0))/len(exact) if exact else None

    def rank(method, stage, metric):
        own = exact_means[method,stage,metric]
        return 1+sum(v is not None and v > own for (m,t,k),v in exact_means.items() if t==stage and k==metric) if own is not None else np.nan

    print(f"Geprüft: {N} Artikel, {len(articles)} Ausgaben, {len(qa)} QA-Positionen; "
          f"{int(qa.technical_missing.sum())} technische Fehlwerte.")

    rows = []
    for t in STAGES:
        for m in METHODS:
            sub = articles[articles.method.eq(m)&articles.stage_target_words.eq(t)]
            row = {"Verfahren":NAMES[m],"Wörter":t,"QA_gültig":int(sub.V.sum()),"QA_fehlend":int(sub.n_missing.sum())}
            for metric,title in [("R_obs","R_gültig"),("R_20","R_20")]:
                s = retention_stats[m,t,metric]
                row.update({f"{title}_%":100*s["mean"],f"{title}_KI_lo":100*s["low"],f"{title}_KI_hi":100*s["high"],f"Rang_{title}":rank(m,t,metric)})
            row["n_Artikel_R_gültig"] = retention_stats[m,t,"R_obs"]["n"]
            rows.append(row)
    t1 = pd.DataFrame(rows)
    table(t1,"01_qa_nenner_retention","1. QA-Missingness, Retention (%) und Ränge",
          "Nenner: je Artikel V gültige QA-Urteile bzw. fest 20; je Gruppe 2.000 erwartete QA-Positionen. "
          "Aggregation: ungewichtetes Mittel der Artikelquoten, Ränge je Stufe. "
          "Fehlwerte: technische QA-Fehler aus V ausgeschlossen; bei Nenner 20 liefern sie keinen Retained-Beitrag. "
          "V=0 ergibt fehlende R_gültig; gültige Artikelanzahl ist ausgewiesen. KI: gemeinsamer Artikelbootstrap.")

    length_rows, status_rows = [], []
    for t in STAGES:
        for m in METHODS:
            sub = articles[articles.method.eq(m)&articles.stage_target_words.eq(t)]
            q1,med,q3 = sub.W.quantile([.25,.5,.75],interpolation="linear")
            length_rows.append({"Verfahren":NAMES[m],"Wörter":t,"n":len(sub),"MW":sub.W.mean(),
                                "SD":sub.W.std(ddof=1),"Q1":q1,"Median":med,"Q3":q3})
            row = {"Verfahren":NAMES[m],"Wörter":t,"Fenster":f"{WINDOWS[t][0]}–{WINDOWS[t][1]}","n":len(sub)}
            for label in ["unterhalb","innerhalb","oberhalb"]:
                count = int(sub.window_status.eq(label).sum())
                row[f"n_{label}"] = count
                row[f"{label}_%"] = 100*count/len(sub)
            row.update({"D_MW":sub.D.mean(),"n_Satzende":int(sub.complete_sentence.sum()),
                        "Satzende_%":100*sub.complete_sentence.mean(),"n_Satzende_erfasst":int(sub.complete_sentence.notna().sum())})
            status_rows.append(row)
    table(pd.DataFrame(length_rows),"02a_wortzahlen","2a. Tatsächliche Wortzahl",
          "Nenner: 100 Ausgaben je Verfahren/Stufe. Aggregation: Artikelverteilung; SD mit ddof=1; "
          "Quartile mit linearer Interpolation. Fehlwerte: keine Wortzahl fehlt; Leertexte zählen als 0 Wörter.")
    table(pd.DataFrame(status_rows),"02b_wortfenster_satzende","2b. Wortfenster, normalisierte Abweichung und vollständiges Satzende",
          "Nenner: 100 Ausgaben je Gruppe, einschließlich Leertexte; Satzende-Anteil unter den erfassten "
          "Satzende-Flags (hier jeweils 100). Aggregation: Ausgabenzählung und Artikelmittel von D; "
          "Fenstergrenzen inklusive. Fehlwerte: keine Flags fehlen. Satzende aus gespeicherter Generierungsdiagnose, "
          "keine nachträgliche linguistische Neubewertung.")

    rows, changes = [], []
    for t in STAGES:
        for m in METHODS:
            s = retention_stats[m,t,"R_adj"]
            penalty = vector(m,t,"R_20")-vector(m,t,"R_adj")
            rows.append({"Verfahren":NAMES[m],"Wörter":t,"R_adj_%":100*s["mean"],"KI_lo_%":100*s["low"],"KI_hi_%":100*s["high"],
                         "Abschlag_ggü_R20_pp":100*np.mean(penalty),"Rang_gültig":rank(m,t,"R_obs"),
                         "Rang_20":rank(m,t,"R_20"),"Rang_adj":rank(m,t,"R_adj"),"n":s["n"]})
        changes.append({"Wörter":t,
                        "Positionen_Nennerwechsel":sum(rank(m,t,"R_obs")!=rank(m,t,"R_20") for m in METHODS),
                        "Positionen_Längenadjustierung":sum(rank(m,t,"R_20")!=rank(m,t,"R_adj") for m in METHODS)})
    t3 = pd.DataFrame(rows)
    changes = pd.DataFrame(changes)
    table(t3,"03a_adjustierung_rangfolgen","3a. Adjustierte Retention (%), Abschlag (pp) und drei Ränge",
          "Nenner: R_adj und Abschlag verwenden je Artikel 20 QA-Positionen; R_gültig verwendet V. "
          "Aggregation: Adjustierung pro Artikel, danach ungewichtetes Artikelmittel; "
          "Abschlag = 100 × mean(R_20 − R_adj), nicht R_gültig − R_adj. "
          "Fehlwerte: technische QA-Fehler geben keinen Retained-Beitrag im Nenner 20. Alle 100 Artikel bleiben enthalten.")
    table(changes,"03b_rangwechsel_stufen","3b. Anzahl veränderter Methodenrangpositionen je Stufe",
          "Nenner: sechs Methodenpositionen je Stufe. Aggregation: Anzahl Methoden mit geändertem Rang, "
          "nicht Summe der zurückgelegten Rangplätze. Nennerwechsel: R_gültig → R_20; "
          "Längenadjustierung: R_20 → R_adj. Fehlwerte: keine Gruppenränge fehlen.")
    change_total = pd.DataFrame([
        {"Änderung":"Nennerwechsel R_gültig→R_20", "Positionen":int(changes.Positionen_Nennerwechsel.sum()),
         "von_Positionen":24,"betroffene_Stufen":int(changes.Positionen_Nennerwechsel.gt(0).sum()),"von_Stufen":4},
        {"Änderung":"Längenadjustierung R_20→R_adj", "Positionen":int(changes.Positionen_Längenadjustierung.sum()),
         "von_Positionen":24,"betroffene_Stufen":int(changes.Positionen_Längenadjustierung.gt(0).sum()),"von_Stufen":4}])
    table(change_total,"03c_rangwechsel_gesamt","3c. Rangwechsel insgesamt",
          "Nenner: 24 Methoden-Stufen-Positionen bzw. vier Stufen. Aggregation: Summe geänderter "
          "Positionen; eine Stufe ist betroffen, sobald mindestens eine Position wechselt. Fehlwerte: keine.")

    CONTRASTS = [("zero_shot_cot","plain"),("few_shot_cot","plain"),
                 ("auto_cot","few_shot_cot"),("automate_cot","auto_cot")]
    contrast_records = []
    for metric,title in [("R_obs","gültiger Nenner"),("R_20","fester Nenner 20"),("R_adj","längenadjustiert")]:
        rows = []
        for t in STAGES:
            for a,b in CONTRASTS:
                x,y = vector(a,t,metric),vector(b,t,metric)
                diff = np.where(np.isfinite(x)&np.isfinite(y),x-y,np.nan)
                s,_ = estimate_mean(diff)
                rows.append({"Wörter":t,"Vergleich":f"{NAMES[a]} minus {NAMES[b]}","Δ_pp":100*s["mean"],
                             "KI_lo_pp":100*s["low"],"KI_hi_pp":100*s["high"],"n_Paare":s["n"]})
                contrast_records.append({"method_a":a,"method_b":b,"stage_target_words":t,"metric":metric,**s})
        table(pd.DataFrame(rows),f"04_kontraste_{metric}",f"4. Gepaarte Differenzen: {title}",
              "Nenner: vollständige gemeinsame Artikelpaare je Vergleich, n_Paare angegeben; QA-Nenner wie im Tabellentitel. "
              "Aggregation: Mittel artikelweiser Differenzen in Prozentpunkten. "
              "Fehlwerte: Paar nur bei zwei gültigen Artikelwerten; beide Verfahren mit derselben Artikelziehung resampelt. "
              "Technische QA-Fehler folgen der jeweiligen Retentionsdefinition.")
    top_rows = []
    for t in STAGES:
        order = sorted(METHODS,key=lambda m:exact_means[m,t,"R_adj"],reverse=True)
        a,b = order[:2]
        s,_ = estimate_mean(vector(a,t,"R_adj")-vector(b,t,"R_adj"))
        top_rows.append({"Wörter":t,"Erstplatziert":NAMES[a],"Zweitplatziert":NAMES[b],"Δ_pp":100*s["mean"],
                         "KI_lo_pp":100*s["low"],"KI_hi_pp":100*s["high"],"n_Paare":s["n"]})
    top_pairs = pd.DataFrame(top_rows)
    table(top_pairs,"04_spitzenvergleich_adjustiert","4. Erst- minus zweitplatziertes Verfahren nach R_adj",
          "Nenner: 100 gemeinsame Artikelpaare je Stufe; QA-Nenner 20. Aggregation: Mittel der gepaarten "
          "adjustierten Differenzen in pp. Fehlwerte: keine Artikelwerte fehlen. Methoden nach ungerundeten "
          "Gruppenpunktschätzern ausgewählt; Intervall punktweise, ohne Korrektur für diese datenabhängige Auswahl.")

    rows = []
    for t in STAGES:
        for m in METHODS:
            sub = qa[qa.method.eq(m)&qa.stage_target_words.eq(t)]
            row = {"Verfahren":NAMES[m],"Wörter":t,"Nenner":len(sub)}
            for label,title in [(R,"Retained"),(L,"Lost"),(D_LABEL,"U_ref"),(M,"Fehlend")]:
                n = int(sub.label.eq(label).sum())
                row[f"n_{title}"] = n
                row[f"{title}_%"] = 100*n/len(sub)
            rows.append(row)
    qa_labels_table = pd.DataFrame(rows)
    table(qa_labels_table,"05_qa_labels","5. QA-Labels: Anzahlen und Anteile (%)",
          "Nenner: alle 2.000 erwarteten QA-Positionen je Verfahren/Stufe, einschließlich technischer Fehlwerte. "
          "Aggregation: gepoolte Faktenzählung; diese Anteile sind keine Mittel artikelweiser Retention. "
          "U_ref = Unsupported/reference_discordant (tatsächliches Quelllabel reference_discordant). "
          "Fehlwerte: eigene Kategorie; die vier Kategorien partitionieren den Nenner vollständig.")

    PATHKEY = ["article_id","method","canonical_fact_id"]
    paths = qa.pivot(index=PATHKEY,columns="stage_target_words",values="label").reindex(columns=STAGES).reset_index()
    paths = paths.merge(facts[FACTKEY+["entity_type"]],on=FACTKEY,validate="many_to_one")
    assert len(paths)==12000 and paths[STAGES].notna().all().all()

    def classify_path(seq):
        first_index = next((i for i,label in enumerate(seq) if label in NON),None)
        if first_index is None:
            return ("offen" if M in seq else "durchgehend", "nicht_im_Risikoset")
        if M in seq[:first_index]:
            return "offen", "nicht_im_Risikoset"
        first = str(STAGES[first_index])
        if first_index == len(STAGES)-1:
            return first,"nicht_im_Risikoset"
        later = seq[first_index+1:]
        if R in later: return first,"später_Retained"
        if M in later: return first,"offen"
        return first,"durchgehend_nicht_Retained"

    assert classify_path([R,M,L,R])==("offen","nicht_im_Risikoset")
    assert classify_path([R,L,M,L])==("300","offen")
    assert classify_path([R,R,R,L])==("75","nicht_im_Risikoset")
    assert classify_path([L,M,R,L])==("500","später_Retained")
    classification = [classify_path(seq) for seq in paths[STAGES].itertuples(index=False,name=None)]
    paths[["first","later"]] = pd.DataFrame(classification,index=paths.index)
    paths.to_csv(OUT/"faktenlaufbahnen.csv",index=False)
    rows = []
    for m in METHODS:
        sub = paths[paths.method.eq(m)]
        for cat,title in [(str(t),f"erstmals {t}") for t in STAGES]+[("durchgehend","durchgehend Retained"),("offen","erstes Nicht-Retained unklar")]:
            n = int(sub.first.eq(cat).sum())
            rows.append({"Verfahren":NAMES[m],"Verlauf":title,"Anzahl":n,"Nenner":len(sub),"Anteil_%":100*n/len(sub)})
    first_table = pd.DataFrame(rows)
    assert first_table.groupby("Verfahren").Anzahl.sum().eq(2000).all()
    table(first_table,"06a_erste_nicht_retained_stufe","6a. Erste Nicht-Retained-Stufe und durchgehend Retained",
          "Nenner: alle 2.000 Ausgangsfakten/Laufbahnen pro Verfahren. Aggregation: Faktenzählung über "
          "500→300→150→75; Nicht-Retained = Lost oder reference_discordant. "
          "Fehlwerte: erste Stufe unklar, wenn ein Fehlwert vor dem ersten beobachteten Nicht-Retained liegt, "
          "oder bei nur Retained/Fehlwert ohne beobachtetes Nicht-Retained. Fehlwerte danach ändern die erste Zuordnung nicht.")
    rows = []
    for m in METHODS:
        sub = paths[paths.method.eq(m)&paths.first.isin(["500","300","150"])]
        for cat,title in [("durchgehend_nicht_Retained","danach durchgehend nicht Retained"),
                          ("später_Retained","später mindestens einmal Retained"),("offen","offen wegen späterem Fehlwert")]:
            n = int(sub.later.eq(cat).sum())
            rows.append({"Verfahren":NAMES[m],"Folgeverlauf":title,"Anzahl":n,"Nenner":len(sub),
                         "Anteil_%":100*n/len(sub) if len(sub) else np.nan})
    persistence_table = pd.DataFrame(rows)
    assert persistence_table.groupby("Verfahren").Anzahl.sum().equals(persistence_table.groupby("Verfahren").Nenner.first())
    table(persistence_table,"06b_persistenz_rueckkehr","6b. Verlauf nach erster Nicht-Retained-Bewertung vor der letzten Stufe",
          "Nenner: eindeutig erstmals bei 500, 300 oder 150 Nicht-Retained bewertete Laufbahnen je Verfahren. "
          "Aggregation: Faktenzählung; erste Bewertung bei 75 und unklare erste Stufe ausgeschlossen. "
          "Fehlwerte: offen nur, wenn kein späteres Retained beobachtet wird und mindestens ein späterer Fehlwert vorliegt. "
          "Mindestens ein beobachtetes späteres Retained genügt für die Rückkehrkategorie, auch mit weiteren Fehlwerten.")

    type_counts = facts.groupby("entity_type").agg(Fakten=("canonical_fact_id","nunique"),Artikel=("article_id","nunique")).reset_index().rename(columns={"entity_type":"Entitätstyp"})
    assert type_counts.Fakten.sum()==2000
    table(type_counts,"07a_entitaetstypen_umfang","7a. Ausgangsfakten und beteiligte Artikel je Entitätstyp",
          "Nenner: 2.000 einzigartige Ausgangsfakten aus 100 Artikeln. Aggregation: einzigartige Fakten und "
          "Artikel je Typ, ohne Vervielfachung durch Verfahren/Stufen; Artikel können mehrere Typen enthalten. "
          "Fehlwerte: kein Entitätstyp fehlt.")
    entity_rows = []
    for m in METHODS:
        rows = []
        for t in STAGES:
            for typ in sorted(facts.entity_type.unique()):
                sub = qa[qa.method.eq(m)&qa.stage_target_words.eq(t)&qa.entity_type.eq(typ)]
                s = estimate_ratio(sub,"is_retained","valid")
                row = {"Verfahren":NAMES[m],"Wörter":t,"Entitätstyp":typ,"Retained":s["count"],"QA_gültig":s["den"],
                       "QA_fehlend":int(sub.technical_missing.sum()),"R_%":100*s["mean"],"KI_lo_%":100*s["low"],
                       "KI_hi_%":100*s["high"],"Artikel_gültig":s["n_articles"],"Boot_gültig":s["n_boot_valid"]}
                rows.append(row)
                entity_rows.append({"method":m,"stage_target_words":t,"entity_type":typ,**s})
        table(pd.DataFrame(rows),f"07_retention_typ_{m}",f"7b. Retention je Entitätstyp: {NAMES[m]}",
              "Nenner: gültige QA-Positionen des Typs/Verfahrens/der Stufe (QA_gültig). "
              "Aggregation: Summe Retained / Summe gültig, faktengewichtet; kein Mittel der Artikelquoten. "
              "Fehlwerte: technische QA-Urteile ausgeschlossen; Artikel ohne gültige Typ-Fakten tragen 0/0 Zählbeiträge bei. "
              "Bootstrap: alle 100 Artikel als Cluster; Ziehungen ohne gültigen Nenner sind fehlend, Boot_gültig angegeben.")

    FIG14_METHODS = ["zero_shot_cot","auto_cot","automate_cot"]
    fig_rows = []
    selected = paths[paths.method.isin(FIG14_METHODS)]
    for before,after in zip(STAGES[:-1],STAGES[1:]):
        valid_pair = selected[before].ne(M)&selected[after].ne(M)
        for typ in sorted(facts.entity_type.unique()):
            sub = selected[valid_pair&selected.entity_type.eq(typ)].copy()
            sub["risk"] = sub[before].eq(R)
            sub["loss"] = sub.risk&sub[after].isin(NON)
            s = estimate_ratio(sub,"loss","risk")
            fig_rows.append({"Übergang":f"{before}→{after}","Entitätstyp":typ,"Nicht_Retained_danach":s["count"],
                             "Retained_vorher_gültig":s["den"],"Verlust_%":100*s["mean"],"KI_lo_%":100*s["low"],
                             "KI_hi_%":100*s["high"],"Boot_gültig":s["n_boot_valid"]})
    fig14 = pd.DataFrame(fig_rows)
    table(fig14,"07c_abbildung14_verlustanteile","7c. Stufenspezifische Verlustanteile hinter Abbildung 14",
          "Nenner: zuvor Retained bewertete Fakten-Verfahrens-Läufe mit gültiger QA an beiden Endpunkten. "
          "Aggregation: Zero-Shot-CoT, Auto-CoT und Automate-CoT gepoolt; Zähler und Nenner über diese drei "
          "Verfahren summiert, kein ungewichtetes Mittel ihrer Prozentwerte. Eine Linie je Entitätstyp über "
          "500→300, 300→150 und 150→75. Zähler: Wechsel zu Lost/reference_discordant. "
          "Fehlwerte: Paar mit fehlender QA an einem Endpunkt ausgeschlossen; 0-Risiko-Bootstrap-Ziehung fehlend. "
          "Alle Methoden/Fakten eines gezogenen Artikels bleiben gemeinsam im Cluster.")

    METRICS = {"highlight_rouge1_f1":"ROUGE-1 F1", "highlight_rouge2_f1":"ROUGE-2 F1",
               "highlight_rougeL_f1":"ROUGE-L F1", "highlight_bertscore_f1":"BERTScore F1",
               "source_summac":"SummaC Original", "stage_input_summac":"SummaC Stufeneingang"}
    metric_path = frozen_input(MET_DATA,VERSION+"/metrics_detailed.csv")
    metric_manifest = frozen_input(MET_DATA,VERSION+"/metrics_manifest.json")
    metric_source_manifest = json.loads((MET_DATA/"source_manifest.json").read_text())
    assert hashlib.sha256(gzip.decompress((QA_DATA/VERSION/"summaries.csv.gz").read_bytes())).hexdigest() == metric_source_manifest["summary_original_sha256"]
    metrics = read(metric_path)
    assert len(metrics)==2400 and not metrics.duplicated(CELL).any()
    assert set(metrics[CELL].itertuples(index=False,name=None)) == set(articles[CELL].itertuples(index=False,name=None))
    for col in METRICS:
        metrics[col] = numeric(metrics[col])
    metrics = metrics.merge(articles[CELL+["W","R_obs","empty_input","empty_output"]],on=CELL,
                            validate="one_to_one",suffixes=("","_qa"))
    assert metrics.stage_actual_words.eq(metrics.W).all()
    metrics["stage_input_summac_stored"] = metrics.stage_input_summac
    metrics.loc[metrics.empty_input,"stage_input_summac"] = np.nan
    assert metrics.empty_input.sum()==1
    assert metrics.loc[metrics.stage_target_words.eq(500),"source_summac"].eq(metrics.loc[metrics.stage_target_words.eq(500),"stage_input_summac"]).all()
    metrics.to_csv(OUT/"artikelwerte_sechs_metriken.csv",index=False,na_rep="fehlend")
    metric_rows, metric_boot = [], {}
    for metric,title in METRICS.items():
        rows = []
        for t in STAGES:
            for m in METHODS:
                sub = metrics[metrics.method.eq(m)&metrics.stage_target_words.eq(t)].set_index("article_id").reindex(IDS)
                s,metric_boot[m,t,metric] = estimate_mean(sub[metric].to_numpy(dtype=float))
                rows.append({"Verfahren":NAMES[m],"Wörter":t,"MW":s["mean"],"KI_lo":s["low"],"KI_hi":s["high"],
                             "n_gültig":s["n"],"n_fehlend":N-s["n"]})
                metric_rows.append({"method":m,"stage_target_words":t,"metric":metric,**s})
        table(pd.DataFrame(rows),f"08_{metric}",f"8. {title}: Mittelwert und 95-%-KI",
              "Nenner: gültige Artikelwerte je Verfahren/Stufe (n_gültig), höchstens 100. Aggregation: "
              "gleichgewichtetes Artikelmittel auf Score-Skala; ROUGE/BERTScore gegen CNN/DailyMail-Highlights; "
              "SummaC gegen Original bzw. tatsächlichen unmittelbaren Stufeneingang. "
              "Fehlwerte: fehlende Scores ausgeschlossen; lokaler SummaC bei leerem Stufeneingang fehlend. "
              "Zwei leere Ausgaben behalten die dokumentierten End-to-end-Nullwerte. "
              "BERTScore/SummaC sind eingefrorene Modellwerte, kein neuer Modelllauf.",digits=6)

    def spearman_bootstrap(x,y):
        x,y = np.asarray(x,dtype=float),np.asarray(y,dtype=float)
        valid = np.isfinite(x)&np.isfinite(y)
        x,y = np.where(valid,x,np.nan),np.where(valid,y,np.nan)
        if valid.sum()<2 or np.unique(x[valid]).size<2 or np.unique(y[valid]).size<2:
            return {"rho":np.nan,"low":np.nan,"high":np.nan,"n":int(valid.sum()),"n_boot_valid":0},np.full(B,np.nan)
        rho = float(spearmanr(x[valid],y[valid]).statistic)
        boot = np.full(B,np.nan)
        # Re-rank each resampled article pair, including repeated articles and ties.
        for start in range(0,B,500):
            sample = draws[start:start+500]
            xr,yr = rankdata(x[sample],axis=1,method="average",nan_policy="omit"),rankdata(y[sample],axis=1,method="average",nan_policy="omit")
            n_valid = np.isfinite(xr).sum(axis=1)
            center_x = np.divide(np.nansum(xr,axis=1),n_valid,out=np.full(len(sample),np.nan),where=n_valid>0)
            center_y = np.divide(np.nansum(yr,axis=1),n_valid,out=np.full(len(sample),np.nan),where=n_valid>0)
            xr,yr = xr-center_x[:,None],yr-center_y[:,None]
            cross = np.nansum(xr*yr,axis=1)
            denominator = np.sqrt(np.nansum(xr*xr,axis=1)*np.nansum(yr*yr,axis=1))
            boot[start:start+len(sample)] = np.divide(cross,denominator,out=np.full(len(sample),np.nan),where=(denominator>0)&(n_valid>=2))
        lo,hi = percentile_interval(boot)
        return {"rho":rho,"low":lo,"high":hi,"n":int(valid.sum()),"n_boot_valid":int(np.isfinite(boot).sum())},boot

    # Check resampled ranks against scipy on tied values and missing paired values.
    test_x = np.arange(N,dtype=float)%7
    test_y = np.arange(N,dtype=float)%11
    test_x[3]=np.nan
    _, test_boot = spearman_bootstrap(test_x,test_y)
    for i in [0,1,32,100,9999]:
        sx,sy = test_x[draws[i]],test_y[draws[i]]
        valid = np.isfinite(sx)&np.isfinite(sy)
        assert np.isclose(test_boot[i],spearmanr(sx[valid],sy[valid]).statistic,rtol=0,atol=1e-12)
    assert np.isnan(spearman_bootstrap(np.ones(N),np.arange(N))[0]["rho"])
    correlation_rows, correlation_boot = [], {}
    for metric,title in METRICS.items():
        rows = []
        for t in STAGES:
            for m in METHODS:
                sub = metrics[metrics.method.eq(m)&metrics.stage_target_words.eq(t)].set_index("article_id").reindex(IDS)
                s,correlation_boot[m,t,metric] = spearman_bootstrap(sub.R_obs,sub[metric])
                rows.append({"Verfahren":NAMES[m],"Wörter":t,"Spearman_ρ":s["rho"],"KI_lo":s["low"],"KI_hi":s["high"],
                             "n_Wertepaare":s["n"],"Boot_gültig":s["n_boot_valid"]})
                correlation_rows.append({"method":m,"stage_target_words":t,"metric":metric,**s})
        table(pd.DataFrame(rows),f"09_spearman_{metric}",f"9. QA-Retention (K/V) vs. {title}: Spearman",
              "Nenner: Artikel mit gültiger QA-Retention K/V und gültigem Metrikwert innerhalb der jeweiligen "
              "Verfahren-Stufe-Gruppe (n_Wertepaare). Aggregation: Spearman über Artikel, Mittelränge bei Ties. "
              "Fehlwerte: paarweise ausgeschlossen; ρ bei konstantem Rand oder weniger als zwei Paaren fehlend. "
              "Bootstrap: gemeinsame Artikelpaare, Ränge in jeder Ziehung neu; undefinierte Ziehungen ausgeschlossen, "
              "Boot_gültig von 10.000 angegeben. Keine Korrelation über Verfahren oder Stufen gepoolt.",digits=6)

    calibration_items = None
    if (VAL_DATA/"qa/internal_model_label_key.csv").is_file():
        human_key_path = frozen_input(VAL_DATA,"qa/internal_model_label_key.csv")
        human_key = read(human_key_path)
        assert len(human_key)==60 and human_key.review_id.is_unique
        human_returns = []
        for path in sorted((VAL_DATA/"qa/returns").glob("reviewer_*_answers.csv")):
            path = frozen_input(VAL_DATA,str(path.relative_to(VAL_DATA)))
            frame = read(path)
            assert len(frame)==60 and frame.review_id.is_unique and set(frame.review_id)==set(human_key.review_id)
            frame["human_answer_label"] = frame.human_answer_label.str.strip().str.lower().str.replace(" ","_",regex=False).str.replace("-","_",regex=False)
            assert frame.human_answer_label.isin(["retained","lost","unsupported","unclear",""]).all()
            human_returns.append(frame[["review_id","human_answer_label"]])
        votes = pd.concat(human_returns,ignore_index=True)
        consensus_rows = []
        for review_id,group in votes.groupby("review_id"):
            count = Counter(group.loc[group.human_answer_label.ne(""),"human_answer_label"])
            winners = sorted(label for label,n in count.items() if n==max(count.values())) if count else []
            consensus_rows.append({"review_id":review_id,"human_label":winners[0] if winners else "",
                                   "tie":len(winners)>1,"n_votes":sum(count.values())})
        calibration_items = human_key.merge(pd.DataFrame(consensus_rows),on="review_id",validate="one_to_one")
        def original_retained(reference,answer):
            if answer.upper()=="NOT_FOUND": return False
            r,a = scorer.normalize(reference),scorer.normalize(answer)
            return bool(r and (r in a or a in r)) or scorer.token_f1(reference,answer)>=.50
        def classify_human(reference,answer,fn):
            return "lost" if answer.upper()=="NOT_FOUND" else "retained" if fn(reference,answer) else "unsupported"
        for col,fn in [("original",original_retained),("final",scorer.retained)]:
            calibration_items[col] = [classify_human(r,a,fn) for r,a in calibration_items[["reference_answer","model_answer"]].itertuples(index=False,name=None)]
        assert calibration_items.original.eq(calibration_items.model_label).all()
        valid = calibration_items.human_label.isin(["retained","lost","unsupported"])&~calibration_items.tie
        strict = calibration_items[valid].copy()
        assert len(strict)==52
        labels = ["retained","lost","unsupported"]
        recall_rows,calibration_summary = [],[]
        for column,title in [("original","Ursprünglicher Scorer"),("final","Finaler Scorer")]:
            matrix = pd.crosstab(strict.human_label,strict[column]).reindex(index=labels,columns=labels,fill_value=0)
            matrix.index.name="Menschliche_Referenz"
            table(matrix.reset_index(),f"10_konfusion_{column}",f"10. Konfusionsmatrix: {title}",
                  "Nenner: dieselben 52 eindeutigen Kalibrierungsfälle, Zeilen menschliche Konsenslabels, "
                  "Spalten automatische Labels. Aggregation: ein Fall = eine Zählung; menschlicher Konsens "
                  "aus verfügbaren Reviewer-Stimmen. Fehlwerte: fehlende Stimmen nicht ergänzt; sechs "
                  "Antwort-Ties und zwei unclear-Konsensfälle ausgeschlossen. Keine unabhängigen Prüffälle.")
            for label in labels:
                correct,total = int(matrix.loc[label,label]),int(matrix.loc[label].sum())
                recall_rows.append({"Scorer":title,"Referenzklasse":label,"Korrekt":correct,"Gesamt":total,
                                    "Recall_%":100*correct/total if total else np.nan})
            calibration_summary.append({"Scorer":title,"Korrekt":int(np.trace(matrix)),"Fälle":len(strict),
                                        "Accuracy_%":100*np.trace(matrix)/len(strict),"Set":"Kalibrierungsfälle"})
        recall_table = pd.DataFrame(recall_rows)
        calibration_summary = pd.DataFrame(calibration_summary)
        table(recall_table,"10_recall","10. Je menschlicher Referenzklasse: korrekt / gesamt und Recall",
              "Nenner: Anzahl der Fälle in der jeweiligen menschlichen Referenzklasse, nicht Anzahl "
              "automatischer Vorhersagen. Aggregation: Diagonale der Konfusionsmatrix / Zeilensumme. "
              "Fehlwerte: gleiche Auswahl der 52 eindeutigen Fälle; Recall bei leerer Referenzklasse fehlend. "
              "Historisches unsupported ist eine Human-Referenzklasse, kein separates Quelllabel des Hauptexperiments.")
        table(calibration_summary,"10_kalibrierungsbasis","10. Auswertungsbasis der Scorerleistung",
              "Nenner: dieselben 52 Kalibrierungsfälle aus sechs Artikeln (von 60 ausgewählten Fällen). "
              "Aggregation: korrekte Fallklassifikationen / Fälle. Fehlwerte: Ties und unclear ausgeschlossen. "
              "Die Scorerregeln wurden an diesen Fällen entwickelt; diese Accuracy ist keine unabhängige "
              "Testleistung und validiert nicht die gesamte finale Faktgenerator-/QA-Pipeline.")
        calibration_items.to_csv(OUT/"kalibrierung_fallebene.csv",index=False)
    else:
        table(pd.DataFrame([{"Ergebnis":"fehlend","Grund":"Human-Validierungsdaten nicht mitgeliefert"}]),
              "10_kalibrierung_fehlend","10. QA-Kalibrierung",
              "Nenner: nicht verfügbar. Aggregation: nicht berechenbar. Fehlwerte: keine Validierungsdaten vorhanden.")

    def checked_export(relative):
        path = BASE/relative
        record_source(path,"Abgleich vorhandener Auswertung")
        return read(path)

    archived_labels = checked_export(f"hauptexperiment_qa_ergebnisse/{VERSION}/qa_labels_reproduziert.csv.gz")
    assert not archived_labels.duplicated(KEY).any() and len(archived_labels)==len(qa)
    aligned = qa[KEY+["label"]].merge(archived_labels[KEY+["label"]],on=KEY,validate="one_to_one",suffixes=("","_archiv"))
    assert aligned.label.eq(aligned.label_archiv).all()
    validation_checks.append({"Prüfung":"QA-Labels positionsgenau reproduziert","Anzahl":len(aligned),"Status":"bestanden"})

    archived_retention = checked_export(f"hauptexperiment_laengen_ergebnisse/{VERSION}/retention_und_drei_rangfolgen_24_gruppen.csv")
    for row in archived_retention.itertuples():
        for metric,old in [("R_obs","R_obs"),("R_20","R_zero"),("R_adj","R_adj")]:
            s = retention_stats[row.method,row.stage_target_words,metric]
            for field,suffix in [("mean","mean"),("low","ci_low"),("high","ci_high")]:
                assert np.isclose(s[field],getattr(row,f"{old}_{suffix}"),rtol=0,atol=1e-12)
            assert rank(row.method,row.stage_target_words,metric)==getattr(row,f"rank_{old}")
    validation_checks.append({"Prüfung":"Retentionsmittel, Intervalle und Ränge","Anzahl":24*3,"Status":"bestanden"})

    entity_archive = checked_export(f"hauptexperiment_verlust_ergebnisse/{VERSION}/retention_entitaetstyp_methoden_stufen_95ci.csv")
    entities_new = pd.DataFrame(entity_rows).merge(entity_archive,on=["method","stage_target_words","entity_type"],validate="one_to_one")
    for new,old in [("mean","estimate"),("low","ci_low"),("high","ci_high")]:
        assert np.allclose(entities_new[new],entities_new[old],rtol=0,atol=1e-12,equal_nan=True)
    validation_checks.append({"Prüfung":"Entitätsretention und Intervalle","Anzahl":len(entities_new),"Status":"bestanden"})

    fig_archive = checked_export(f"hauptexperiment_verlust_ergebnisse/{VERSION}/verlust_entitaetstyp_uebergaenge_95ci.csv")
    fig_archive = fig_archive[fig_archive.method_group.eq("Drei generative Verfahren gepoolt")]
    fig_aligned = fig14.merge(fig_archive,left_on=["Übergang","Entitätstyp"],right_on=["transition","entity_type"],validate="one_to_one")
    for new,old in [("Verlust_%","loss_percent"),("KI_lo_%","loss_ci_low_percent"),("KI_hi_%","loss_ci_high_percent")]:
        assert np.allclose(fig_aligned[new],fig_aligned[old],rtol=0,atol=1e-10,equal_nan=True)
    validation_checks.append({"Prüfung":"Abbildung-14-Punktwerte und Intervalle","Anzahl":len(fig_aligned),"Status":"bestanden"})

    metric_archive = checked_export(f"hauptexperiment_metriken_ergebnisse/{VERSION}/mittelwerte_24_gruppen_alle_metriken_95ci.csv")
    metric_aligned = pd.DataFrame(metric_rows).merge(metric_archive,on=["method","stage_target_words","metric"],validate="one_to_one",suffixes=("","_archiv"))
    for new,old in [("mean","mean_archiv"),("low","ci_low"),("high","ci_high"),("n","n_valid")]:
        assert np.allclose(metric_aligned[new],numeric(metric_aligned[old]),rtol=0,atol=1e-12,equal_nan=True)
    validation_checks.append({"Prüfung":"Sechs Metriken: Mittel, Intervalle und gültiges n","Anzahl":len(metric_aligned),"Status":"bestanden"})

    path_archive = checked_export(f"hauptexperiment_verlust_ergebnisse/{VERSION}/faktenlaufbahnen.csv")
    path_aligned = paths.merge(path_archive[PATHKEY+["first_category","persistence"]],on=PATHKEY,validate="one_to_one")
    assert path_aligned.first.eq(path_aligned.first_category.replace({"always_retained":"durchgehend","undetermined":"offen"})).all()
    assert path_aligned.loc[path_aligned.first.isin(["500","300","150"]),"later"].eq(
        path_aligned.loc[path_aligned.first.isin(["500","300","150"]),"persistence"].replace(
            {"all_later_non_retained":"durchgehend_nicht_Retained","observed_return":"später_Retained","persistence_undetermined":"offen"})).all()
    validation_checks.append({"Prüfung":"Erste Verluststufe und Folgeverläufe","Anzahl":len(path_aligned),"Status":"bestanden"})
    if calibration_items is not None:
        assert calibration_summary.Korrekt.tolist()==[32,41]
        validation_checks.append({"Prüfung":"Ursprünglicher/finaler Scorer: 32/52 und 41/52","Anzahl":52,"Status":"bestanden"})
    validation_checks.append({"Prüfung":"Spearman-Resampling mit Ties/Fehlwerten vs. scipy","Anzahl":5,"Status":"bestanden"})
    table(pd.DataFrame(validation_checks),"11_rechenpruefungen","Rechenprüfungen",
          "Nenner: jeweiliger Prüfumfang. Aggregation: vollständiger Schlüsselabgleich bzw. "
          "numerischer Vergleich mit absoluter Toleranz 10⁻¹² (Prozentwerte 10⁻¹⁰). "
          "Fehlwerte: bei numerischen Vergleichen nur identische fehlende Werte akzeptiert.")
    source_index = pd.DataFrame(sources)[["Datei","Rolle"]]
    table(source_index,"12_quellenindex","Quellenindex",
          "Nenner: tatsächlich gelesene Quellen einschließlich Manifeste und Abgleiche. Aggregation: "
          "eine Zeile pro Datei; vollständige SHA-256-Werte in quellen_sha256.csv. "
          "Fehlwerte: erforderliche Hauptquellen dürfen nicht fehlen; Validierungsquellen optional.")
    source_frame = pd.DataFrame(sources)
    source_frame.to_csv(OUT/"quellen_sha256.csv",index=False)
    software = {package:importlib.metadata.version(package) for package in ["numpy","pandas","scipy","nbformat","nbclient","nbconvert","ipykernel"]}
    manifest = {"version":VERSION,"n_articles":N,"n_outputs":len(articles),"n_qa":len(qa),
                "seed":SEED,"bootstrap_draws":B,"bootstrap_unit":"article", "interval":"95% percentile, linear quantiles",
                "paired":"one shared article index matrix across all methods, stages and facts",
                "spearman":"complete pairs within method-stage; average ranks recomputed in every draw",
                "retention_denominators":{"R_obs":"K/V","R_20":"K/20","R_adj":"(K/20)/(1+D)"},
                "length_penalty":"mean(R_20-R_adj)*100 percentage points",
                "ranking":"exact rational means; competition ranks for ties",
                "figure14_methods":FIG14_METHODS,"figure14_denominator":"retained at start and valid at both endpoints",
                "calibration":"same development cases, unique plurality excluding ties/unclear; no independent test",
                "software":{"python":platform.python_version(),**software},"tables":table_registry,
                "checks":validation_checks,"sources":sources}
    (OUT/"execution_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    pass  # Große Bootstrap-Zwischendatei wird nicht benötigt.
    print(f"Alle Rechenprüfungen bestanden. {len(table_registry)} vollständige Texttabellen.\n"
          f"Text: {TEXT_FILE}\nCSV und Provenienz: {OUT}")

    from IPython.display import display, Markdown
    missing = int(qa.technical_missing.sum())
    lines = [f"- **QA:** {missing:,} technische Fehlwerte von {len(qa):,} Positionen ({100*missing/len(qa):.4f} %).",
             f"- **Nennerwechsel:** {int(changes.Positionen_Nennerwechsel.sum())} von 24 Methodenrangpositionen, "
             f"{int(changes.Positionen_Nennerwechsel.gt(0).sum())} von vier Stufen betroffen.",
             f"- **Längenadjustierung:** {int(changes.Positionen_Längenadjustierung.sum())} von 24 Methodenrangpositionen, "
             f"{int(changes.Positionen_Längenadjustierung.gt(0).sum())} von vier Stufen betroffen.",
             f"- **Lokaler SummaC:** Ein fehlender Wert bei leerem Stufeneingang; die betroffene Gruppe und Korrelation haben 99 gültige Artikelpaare."]
    if calibration_items is not None:
        before,after = calibration_summary.iloc[0],calibration_summary.iloc[1]
        lines.append(f"- **Kalibrierung:** ursprünglich {int(before.Korrekt)}/52 ({before['Accuracy_%']:.4f} %), "
                     f"final {int(after.Korrekt)}/52 ({after['Accuracy_%']:.4f} %); Leistung auf denselben Kalibrierungsfällen.")
    lines.append("- Kopierbare Texttabellen sind im Notebook gespeichert; CSV-Dateien enthalten ungerundete Werte. "
                 "Die ausgewählten Spitzenvergleiche sind deskriptiv und keine unabhängig bestätigten Sieger.")
    display(Markdown("\n".join(lines)))
    with TEXT_FILE.open("a",encoding="utf-8") as stream:
        stream.write("Befunde\n"+"\n".join(lines)+"\n")

if __name__ == '__main__':
    from offline_guard import install
    install(allow_statistics_workers='--recompute-regression-bootstrap' in sys.argv)
    main(sys.argv[1], '--recompute-regression-bootstrap' in sys.argv, int(sys.argv[2]))
