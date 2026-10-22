# Statische Originalauswertung; Eingaben/Ausgaben relativ zur übergebenen Arbeitskopie.
# Keine LLM-Aufrufe. Standard: vorhandene Regressionscaches verwenden.
from pathlib import Path
import os,sys

def main(work, recompute_bootstrap=False, workers=1):
    WORK=__import__('pathlib').Path(work).resolve()
    __import__('os').chdir(WORK/('supplement' if 'human_validation_qa_faktenretention'[0].isdigit() else 'project/02_Auswertung'))
    from pathlib import Path
    from collections import Counter
    from itertools import combinations
    import hashlib, importlib.util, json, re, sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix, precision_recall_fscore_support
    from IPython.display import display, Markdown

    candidates = [Path.cwd(), Path.cwd() / "Notebooks und Grafiken"]
    BASE = next((p for p in candidates if (p / "human_validation_qa_daten/source_manifest.json").is_file()), None)
    if BASE is None:
        raise FileNotFoundError("Notebook im mitgelieferten Ordner starten; human_validation_qa_daten fehlt.")
    DATA = BASE / "human_validation_qa_daten"
    OUT = BASE / "human_validation_qa_grafiken"
    OUT.mkdir(exist_ok=True)
    LABELS = ["retained", "lost", "unsupported"]
    QUESTION_LABELS = ["good", "partial", "bad", "unclear"]
    ANSWER_LABELS = LABELS + ["unclear"]
    COLORS = {"retained": "#286a9f", "lost": "#a8afb8", "unsupported": "#c27c2c", "unclear": "#75658f"}
    plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "savefig.facecolor": "white"})
    pd.set_option("display.max_colwidth", 85)
    pd.set_option("display.max_rows", 24)
    def read(relative):
        return pd.read_csv(DATA / relative, keep_default_na=False)
    def export(table, name):
        table.to_csv(OUT / f"{name}.csv", index=False)
    def savefig(fig, name):
        fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
        plt.show()
        plt.close(fig)
    def bool_col(series):
        return series.astype(str).str.lower().isin(["true", "1"])
    def pct(value):
        return f"{value * 100:.1f} %".replace(".", ",")

    manifest = json.loads((DATA / "source_manifest.json").read_text())
    for entry in manifest["files"]:
        assert hashlib.sha256((DATA / entry["file"]).read_bytes()).hexdigest() == entry["sha256"], entry["file"]
    key = read("qa/internal_model_label_key.csv")
    assert len(key) == 60 and key.review_id.is_unique
    assert key[["article_id", "method", "stage_target_words", "canonical_fact_id"]].duplicated().sum() == 0
    returns = []
    audit_rows = []
    for path in sorted((DATA / "qa/returns").glob("reviewer_*_answers.csv")):
        frame = pd.read_csv(path, keep_default_na=False)
        assert len(frame) == 60 and frame.review_id.is_unique
        assert set(frame.review_id) == set(key.review_id)
        frame["reviewer_id"] = path.stem.removesuffix("_answers")
        for field, allowed in [("question_quality", QUESTION_LABELS), ("human_answer_label", ANSWER_LABELS)]:
            normalized = frame[field].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False)
            audit_rows.append({"Reviewer": frame.reviewer_id.iloc[0], "Feld": field,
                               "Leer": normalized.eq("").sum(), "Ungültig": (~normalized.isin(allowed + [""])).sum()})
            frame[field] = normalized.where(normalized.isin(allowed), pd.NA)
        returns.append(frame[["review_id", "reviewer_id", "question_quality", "human_answer_label", "overall_notes"]])
    annotations = pd.concat(returns, ignore_index=True).merge(
        key.drop(columns=["question_quality", "human_answer_label", "overall_notes"]),
        on="review_id", validate="many_to_one")
    annotations["complete"] = annotations[["question_quality", "human_answer_label"]].notna().all(axis=1)
    audit = pd.DataFrame(audit_rows)
    assert len(annotations) == 420 and annotations.complete.sum() == 204
    assert audit["Ungültig"].sum() == 0
    assert key.article_id.nunique() == 6 and key.canonical_fact_id.nunique() == 25
    assert key.model_label.value_counts().eq(20).all()
    assert key.method.value_counts().eq(20).all()
    assert key.stage_target_words.value_counts().eq(15).all()
    display(pd.DataFrame({"Dimension": ["QA-Fälle", "Artikel", "Einzigartige Fakten/Fragen", "Reviewer", "Mögliche Urteilszeilen", "Vollständige Urteilszeilen"],
                          "Anzahl": [len(key), key.article_id.nunique(), key.canonical_fact_id.nunique(), len(returns), len(annotations), annotations.complete.sum()]}))
    export(audit, "datenqualitaetspruefung")
    export(annotations, "rekonstruierte_einzelurteile")

    sample_rows = []
    for dimension in ["article_id", "method", "stage_target_words", "entity_type", "model_label"]:
        for value, count in key[dimension].value_counts().sort_index().items():
            sample_rows.append({"Dimension": dimension, "Ausprägung": str(value), "Fälle": count})
    sample = pd.DataFrame(sample_rows)
    display(sample)
    display(pd.crosstab(key.method, key.stage_target_words))
    export(sample, "stichprobenzusammensetzung")

    completion = annotations.groupby("reviewer_id").agg(
        erwartete_faelle=("review_id", "size"), vollstaendig=("complete", "sum"),
        fragenurteile=("question_quality", "count"), antworturteile=("human_answer_label", "count")).reset_index()
    completion["anteil_vollstaendig"] = completion.vollstaendig / completion.erwartete_faelle
    assert completion.vollstaendig.tolist() == [60, 2, 30, 7, 6, 39, 60]
    votes = annotations.pivot(index="reviewer_id", columns="review_id", values="human_answer_label").reindex(columns=key.review_id)
    coverage = votes.notna().astype(int)
    fig, (ax, heat) = plt.subplots(2, 1, figsize=(12, 6.6), height_ratios=[1, 1.1], layout="constrained")
    ax.barh(completion.reviewer_id, completion.vollstaendig, color="#286a9f")
    for i, row in completion.iterrows():
        ax.text(row.vollstaendig + .6, i, f"{row.vollstaendig}/60 ({row.anteil_vollstaendig:.1%})", va="center")
    ax.set(xlim=(0, 77), xlabel="Vollständige Zwei-Fragen-Bewertungen", title="Ungleichmäßige Beteiligung der sieben Reviewer")
    ax.invert_yaxis()
    heat.imshow(coverage, cmap=ListedColormap(["#eeeeee", "#286a9f"]), aspect="auto", vmin=0, vmax=1)
    heat.set(yticks=range(7), yticklabels=coverage.index, xticks=range(0, 60, 5),
             xticklabels=coverage.columns[::5], xlabel="QA-Fall (HV001–HV060)", title="Gültiges Antwortlabel je Reviewer und Fall; Grau = fehlend")
    savefig(fig, "01_beteiligung_und_missingness")
    display(completion.round(3))
    export(completion, "reviewer_vollstaendigkeit")
    valid_answer = annotations[annotations.human_answer_label.notna()]
    valid_question = annotations[annotations.question_quality.notna()]
    per_item_votes = valid_answer.groupby("review_id").size()
    display(per_item_votes.value_counts().sort_index().rename_axis("Gültige Stimmen pro Fall").rename("Anzahl Fälle").to_frame())
    display(Markdown(f"**Befund:** {annotations.complete.sum()}/{len(annotations)} vollständige Bewertungen ({pct(annotations.complete.mean())}); "
                     f"pro Fall {per_item_votes.min()}–{per_item_votes.max()} Antwortstimmen. Ausfälle sind anhand der Daten nicht als zufällig nachgewiesen."))

    def consensus_table(frame, field):
        records = []
        for review_id, group in frame.groupby("review_id", sort=True):
            labels = group[field].dropna().tolist()
            counts = Counter(labels)
            if not counts:
                records.append({"review_id": review_id, "label": pd.NA, "tie": False, "votes": 0,
                                "max_votes": 0, "strict_majority": False, "unanimous": False})
                continue
            max_votes = max(counts.values())
            winners = sorted(k for k, v in counts.items() if v == max_votes)
            records.append({"review_id": review_id, "label": winners[0], "tie": len(winners) > 1,
                            "votes": len(labels), "max_votes": max_votes,
                            "strict_majority": max_votes > len(labels) / 2,
                            "unanimous": len(counts) == 1})
        return pd.DataFrame(records)
    answer_consensus = consensus_table(annotations, "human_answer_label")
    question_consensus = consensus_table(annotations, "question_quality")
    archived = read("qa/analysis/majority_by_item.csv")
    for rebuilt, label_column, tie_column in [(answer_consensus, "majority_answer_label", "answer_label_tie"),
                                              (question_consensus, "majority_question_quality", "question_quality_tie")]:
        aligned = rebuilt.merge(archived[["review_id", label_column, tie_column]], on="review_id", validate="one_to_one")
        assert aligned.label.eq(aligned[label_column]).all()
        assert aligned.tie.eq(bool_col(aligned[tie_column])).all()
    items = key.merge(answer_consensus.rename(columns={"label": "human_label"}), on="review_id", validate="one_to_one")
    items = items.merge(question_consensus[["review_id", "label", "tie"]].rename(
        columns={"label": "human_question_quality", "tie": "question_tie"}), on="review_id", validate="one_to_one")
    items["legacy_eligible"] = items.human_label.isin(LABELS)
    items["strict_eligible"] = items.legacy_eligible & ~items.tie
    assert items.legacy_eligible.sum() == 58 and items.strict_eligible.sum() == 52
    assert items.tie.sum() == 6 and items.question_tie.sum() == 2
    qcounts = valid_question.question_quality.value_counts().reindex(QUESTION_LABELS, fill_value=0)
    assert qcounts.tolist() == [172, 19, 6, 7]
    quality = qcounts.rename_axis("Label").rename("Urteile").reset_index()
    quality["Anteil"] = quality.Urteile / len(valid_question)
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    bars = ax.barh(quality.Label, quality.Anteil * 100, color=["#286a9f", "#e4ad42", "#a8afb8", "#75658f"])
    for bar, row in zip(bars, quality.itertuples()):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{row.Urteile}/204 ({row.Anteil:.1%})", va="center")
    ax.set(xlim=(0, 105), xlabel="Anteil der gültigen Fragenqualitätsurteile (%)", title="Qualität der ausgewählten QA-Einheiten")
    ax.invert_yaxis()
    savefig(fig, "02_fragenqualitaet")
    display(quality.round(4))
    display(Markdown(f"**Befund:** 172/204 Urteile sind `good`, 191/204 `good` oder `partial`. "
                     f"Alle 60 historischen Konsenslabels lauten `good`; {question_consensus.tie.sum()} davon sind Ties. "
                     "58/58 eindeutige Fallkonsenslabels sind `good`. Die 60 Einheiten enthalten 25 einzigartige Fragen."))
    export(quality, "fragenqualitaet")
    export(question_consensus, "fragenqualitaet_konsens")

    def nominal_alpha(frame, field, allowed):
        coincidence = np.zeros((len(allowed), len(allowed)), dtype=float)
        usable_units = 0
        for _, group in frame.groupby("review_id"):
            values = group.loc[group[field].isin(allowed), field]
            m = len(values)
            if m < 2:
                continue
            count = values.value_counts().reindex(allowed, fill_value=0).to_numpy(dtype=float)
            coincidence += (np.outer(count, count) - np.diag(count)) / (m - 1)
            usable_units += 1
        n = coincidence.sum()
        if n < 2:
            return np.nan, usable_units
        marginal = coincidence.sum(axis=0)
        observed = (n - np.trace(coincidence)) / n
        expected = (n*n - np.sum(marginal*marginal)) / (n*(n-1))
        return (1-observed/expected if expected else np.nan), usable_units

    # Verifikation anhand des nominalen Beispiels mit zwei Beobachtern in Krippendorffs Originaltext.
    example = pd.DataFrame({"review_id": list(range(12))*2,
                           "x": list("aabb dcccedda".replace(" ", "")) + list("babbbccceddd")})
    example_alpha, _ = nominal_alpha(example, "x", list("abcde"))
    assert np.isclose(example_alpha, 0.6919642857142857), example_alpha

    reliability_rows = []
    for field, allowed in [("question_quality", QUESTION_LABELS), ("human_answer_label", ANSWER_LABELS)]:
        matches = total = 0
        unit_agreement = []
        for _, group in annotations.groupby("review_id"):
            pairs = list(combinations(group[field].dropna(), 2))
            if pairs:
                unit_matches = sum(a == b for a, b in pairs)
                matches += unit_matches; total += len(pairs)
                unit_agreement.append(unit_matches / len(pairs))
        alpha, usable = nominal_alpha(annotations, field, allowed)
        reliability_rows.append({"Dimension": field, "Übereinstimmende Paare": matches, "Gültige Paare": total,
                                 "Rohe Paarübereinstimmung": matches/total,
                                 "Fallmittel der Paarübereinstimmung": np.mean(unit_agreement),
                                 "Nominales Alpha": alpha, "Fälle für Alpha": usable})
    reliability = pd.DataFrame(reliability_rows)
    assert reliability["Übereinstimmende Paare"].tolist() == [184, 149]
    assert reliability["Gültige Paare"].tolist() == [264, 264]
    answer_alpha, _ = nominal_alpha(annotations, "human_answer_label", ANSWER_LABELS)
    answer_alpha_three, alpha_three_n = nominal_alpha(annotations, "human_answer_label", LABELS)
    display(reliability.round(4))
    display(Markdown(f"**Neu berechnet:** Antwortlabels: nominales Alpha = {answer_alpha:.3f} mit `unclear`; "
                     f"ohne `unclear` = {answer_alpha_three:.3f} ({alpha_three_n} auswertbare Fälle). "
                     "Die Fallaggregation erzeugt eine verwendbare Referenz, aber keine fehlerfreie Ground Truth."))
    export(reliability, "reliabilitaet")

    reviewers = sorted(annotations.reviewer_id.unique())
    pair_rows = []
    kappas = np.full((7,7), np.nan)
    overlaps = np.zeros((7,7), dtype=int)
    for i,j in combinations(range(7), 2):
        common = votes.loc[[reviewers[i], reviewers[j]]].T.dropna()
        n = len(common)
        a,b = common.iloc[:,0], common.iloc[:,1]
        combined = pd.concat([a,b])
        kappa = cohen_kappa_score(a,b) if n and combined.nunique() > 1 else np.nan
        kappas[i,j] = kappas[j,i] = kappa
        overlaps[i,j] = overlaps[j,i] = n
        pair_rows.append({"Reviewer A":reviewers[i],"Reviewer B":reviewers[j],"Gemeinsame Fälle":n,
                          "Rohe Übereinstimmung":a.eq(b).mean() if n else np.nan,"Cohen-Kappa (nominal)":kappa})
    fig, ax = plt.subplots(figsize=(8,6.8), layout="constrained")
    kappa_cmap = LinearSegmentedColormap.from_list("kappa", ["#c27c2c", "#ffffff", "#286a9f"])
    im = ax.imshow(np.ma.masked_invalid(kappas), cmap=kappa_cmap, vmin=-1, vmax=1)
    for i in range(7):
        for j in range(7):
            if i != j:
                value = kappas[i,j]
                ax.text(j,i,(f"{value:.2f}" if np.isfinite(value) else "n. d.")+f"\nn={overlaps[i,j]}",
                        ha="center",va="center",color="white" if abs(value) > .6 else "#222222",fontsize=10)
    ax.set(xticks=range(7),yticks=range(7),xticklabels=[f"R{i+1}" for i in range(7)],yticklabels=[f"R{i+1}" for i in range(7)],
           xlabel="Reviewer",ylabel="Reviewer",title="Paarweise Antwortlabel-Reliabilität; Zahl gemeinsamer Fälle")
    fig.colorbar(im,ax=ax,label="Ungewichtetes Cohen-Kappa")
    savefig(fig,"03_reviewer_kappa")
    pair_metrics = pd.DataFrame(pair_rows)
    export(pair_metrics,"reviewer_paarvergleich")
    strength = items.groupby(["votes","max_votes","tie","strict_majority"]).size().rename("Fälle").reset_index()
    display(strength)
    display(items.loc[items.tie,["review_id","article_id","human_label","votes","max_votes"]])
    export(items,"konsens_je_fall")

    comparable_rows = annotations[annotations.human_answer_label.isin(LABELS)].copy()
    row_matches = comparable_rows.model_label.eq(comparable_rows.human_answer_label).sum()
    assert len(comparable_rows) == 182 and row_matches == 106
    def metric_row(frame, prediction, name):
        y = frame.human_label
        p = frame[prediction]
        if not len(frame):
            return {"Auswertung":name,"Fälle":0,"Übereinstimmungen":0,"Accuracy":np.nan,"Kappa":np.nan,"Macro-F1":np.nan}
        return {"Auswertung":name,"Fälle":len(frame),"Übereinstimmungen":int(y.eq(p).sum()),
                "Accuracy":accuracy_score(y,p),"Kappa":cohen_kappa_score(y,p,labels=LABELS),
                "Macro-F1":f1_score(y,p,labels=LABELS,average="macro",zero_division=0)}
    baseline = pd.DataFrame([metric_row(items[items.legacy_eligible],"model_label","Historisch: unclear ausgeschlossen, Ties alphabetisch"),
                             metric_row(items[items.strict_eligible],"model_label","Primär: eindeutiges Drei-Klassen-Label")])
    display(baseline.round(4))
    assert baseline["Übereinstimmungen"].tolist() == [33,32]
    strict = items[items.strict_eligible].copy()
    majority_class = strict.human_label.mode().iloc[0]
    majority_baseline = strict.human_label.eq(majority_class).mean()
    display(Markdown(f"**Befund:** Einzelurteile: {row_matches}/182 = {pct(row_matches/182)}. "
                     f"Eindeutige Fälle: 32/52 = {pct(32/52)}. Die konstante Regel `{majority_class}` erreicht hier "
                     f"{pct(majority_baseline)}; Accuracy allein beschreibt die Klassenfehler deshalb unzureichend."))
    export(baseline,"urspruenglicher_scorer_kennzahlen")

    import qa_scorer_snapshot as scorer
    def original_retained(reference, answer):
        if answer.upper() == "NOT_FOUND":
            return False
        r,a = scorer.normalize(reference),scorer.normalize(answer)
        return bool(r and (r in a or a in r)) or scorer.token_f1(reference,answer) >= .50
    def first_patch_canonical(text):
        text = scorer.normalize(text)
        for source,target in [("u s a","united states"),("u s","united states"),("usa","united states"),("us","united states")]:
            text = re.sub(rf"\b{source}\b",target,text)
        decades = {"one":10,"two":20,"three":30,"four":40,"five":50,"six":60,"seven":70,"eight":80,"nine":90,"ten":100}
        for word,years in decades.items():
            text = re.sub(rf"\bover {word} decades?\b",f"more than {years} years",text)
            text = re.sub(rf"\b{word} decades?\b",f"{years} years",text)
        return text
    def first_patch_retained(reference,answer):
        if original_retained(reference,answer):
            return True
        if answer.upper() == "NOT_FOUND":
            return False
        r,a = first_patch_canonical(reference),first_patch_canonical(answer)
        if r and (r in a or a in r):
            return True
        return bool(0 < len(r.split()) <= 10 and len(a.split()) / len(r.split()) <= 4 and scorer.token_f1(r,a) >= .43)
    def classify(reference, answer, retained_function):
        return "lost" if answer.upper() == "NOT_FOUND" else "retained" if retained_function(reference,answer) else "unsupported"
    items["reconstructed_original"] = [classify(r,a,original_retained) for r,a in zip(items.reference_answer,items.model_answer)]
    assert items.reconstructed_original.eq(items.model_label).all()
    items["first_patch"] = [classify(r,a,first_patch_retained) for r,a in zip(items.reference_answer,items.model_answer)]
    items["calibrated"] = [classify(r,a,scorer.retained) for r,a in zip(items.reference_answer,items.model_answer)]
    versions = [("model_label","Ursprünglich"),("first_patch","Erste Kalibrierung (rekonstruiert)"),("calibrated","Generalisierte Kalibrierung (Snapshot)")]
    calibration_rows=[]
    for mask,name in [(items.legacy_eligible,"Historisch (n=58)"),(items.strict_eligible,"Eindeutig (n=52)")]:
        for column,title in versions:
            row=metric_row(items[mask],column,title);row["Vergleichsset"]=name;calibration_rows.append(row)
    calibration=pd.DataFrame(calibration_rows)
    assert calibration[calibration.Vergleichsset.eq("Historisch (n=58)")]["Übereinstimmungen"].tolist() == [33,37,42]
    assert calibration[calibration.Vergleichsset.eq("Eindeutig (n=52)")]["Übereinstimmungen"].tolist() == [32,36,41]
    calibrated_rows = comparable_rows.merge(items[["review_id","calibrated"]],on="review_id",validate="many_to_one")
    assert calibrated_rows.calibrated.eq(calibrated_rows.human_answer_label).sum() == 128
    display(calibration.round(4))
    export(calibration,"scorer_kalibrierung")
    fig,ax=plt.subplots(figsize=(11,4.8),layout="constrained")
    y=np.arange(3)
    for offset,set_name,color,hatch in [(-.18,"Historisch (n=58)","#a8afb8","//"),(.18,"Eindeutig (n=52)","#286a9f",None)]:
        group=calibration[calibration.Vergleichsset.eq(set_name)]
        bars=ax.barh(y+offset,group.Accuracy*100,height=.32,color=color,hatch=hatch,label=set_name)
        for bar,row in zip(bars,group.itertuples()):
            ax.text(bar.get_width()+.7,bar.get_y()+bar.get_height()/2,f"{row.Übereinstimmungen}/{row.Fälle} ({row.Accuracy:.1%})",va="center",fontsize=10)
    ax.set(yticks=y,yticklabels=["Ursprünglicher Scorer","Erste Kalibrierung","Generalisierte Kalibrierung"],xlim=(0,100),
           xlabel="Übereinstimmung mit aggregierten menschlichen Labels (%)",title="Kalibrierung auf denselben historischen Antworten")
    ax.invert_yaxis();ax.legend(loc="lower center",bbox_to_anchor=(.5,1.02),ncols=2);ax.set_title(ax.get_title(),pad=42)
    savefig(fig,"04_kalibrierung")
    display(Markdown("**Interpretation:** 9 zusätzliche Übereinstimmungen auf 52 eindeutigen Fällen (+17,3 Prozentpunkte). "
                     "Die Verbesserung ist deskriptiv und innerhalb der Kalibrierungsstichprobe gemessen. "
                     "Einzelurteile steigen von 106/182 auf 128/182 (58,2 % → 70,3 %); diese Zeilen sind nicht unabhängig."))

    strict=items[items.strict_eligible].copy()
    fig,axes=plt.subplots(1,2,figsize=(12,5.3),layout="constrained")
    for ax,column,title in zip(axes,["model_label","calibrated"],["Ursprünglicher Scorer","Kalibrierter Scorer"]):
        matrix=confusion_matrix(strict.human_label,strict[column],labels=LABELS)
        shares=matrix/matrix.sum(axis=1,keepdims=True)*100
        im=ax.imshow(shares,cmap="Blues",vmin=0,vmax=100)
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f"{matrix[i,j]}\n({shares[i,j]:.1f}%)",ha="center",va="center",color="white" if shares[i,j]>55 else "#222222")
        ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,
               yticklabels=[f"{label} (n={matrix[i].sum()})" for i,label in enumerate(LABELS)],xlabel="Automatisches Label",ylabel="Menschlicher Konsens",title=title)
        export(pd.DataFrame(matrix,index=LABELS,columns=LABELS).reset_index(names="Human"),f"konfusion_{column}")
    fig.colorbar(im,ax=list(axes),shrink=.7,label="Anteil innerhalb der menschlichen Klasse (%)")
    savefig(fig,"05_konfusion_vor_nach")
    for column,title in [("model_label","Original"),("calibrated","Kalibriert")]:
        group=strict[strict[column].eq("unsupported")]
        display(Markdown(f"**{title}:** {len(group)} automatische `unsupported`-Fälle; "
                         f"davon {group.human_label.eq('unsupported').sum()} menschlich ebenfalls `unsupported`. "
                         "Das ist eine bedingte Stichprobenaussage über das automatische Label, keine Sensitivität für sämtliche Halluzinationen."))
    class_rows=[]
    for column,title in [("model_label","Original"),("calibrated","Kalibriert")]:
        precision,recall,f1,support=precision_recall_fscore_support(strict.human_label,strict[column],labels=LABELS,zero_division=0)
        for label,p,r,f,n in zip(LABELS,precision,recall,f1,support):
            class_rows.append({"Scorer":title,"Human-Klasse":label,"Human-Fälle":n,"Precision":p,"Recall":r,"F1":f})
    class_metrics=pd.DataFrame(class_rows)
    display(class_metrics.round(4))
    export(class_metrics,"klassenmetriken")
    display(Markdown("**Messgrenze:** Retained-Recall steigt von 16/30 (53,3 %) auf 25/30 (83,3 %). "
                     "Die drei menschlichen `unsupported`-Fälle bleiben als `lost` eingestuft (Recall 0/3). "
                     "Alle sieben verbleibenden automatischen `unsupported`-Fälle haben menschlich `retained` oder `lost`. "
                     "Es ist damit weder hohe Precision noch hohe Sensitivität für menschliches `unsupported` belegt."))

    changes=items[items.model_label.ne(items.calibrated)].copy()
    changes["Korrektur_gegen_historischen_Konsens"]=changes.calibrated.eq(changes.human_label)
    display(changes[["review_id","article_id","canonical_fact_id","reference_answer","model_answer","model_label","calibrated","human_label","tie"]])
    assert len(changes)==10 and changes.Korrektur_gegen_historischen_Konsens.sum()==9
    export(changes,"alle_umklassifikationen")
    changed_strict=changes[changes.strict_eligible]
    display(Markdown(f"**Befund:** {len(changed_strict)} eindeutige Fälle werden korrigiert; sie verteilen sich auf "
                     f"{changed_strict.canonical_fact_id.nunique()} einzigartige Fakten aus {changed_strict.article_id.nunique()} Artikeln. "
                     "Die zehnte Änderung widerspricht dem alphabetisch ausgewählten Tie-Label und wird nicht als bestätigter Fehler oder Erfolg gezählt."))
    case_ids=["HV001","HV002","HV023","HV033","HV041","HV048"]
    case_examples=items[items.review_id.isin(case_ids)][["review_id","reference_answer","model_answer","model_label","calibrated","human_label","tie"]]
    export(case_examples,"qualitative_fallbeispiele")

    subgroups=[]
    for dimension in ["method","stage_target_words","entity_type","article_id"]:
        for value,group in strict.groupby(dimension):
            before=group.model_label.eq(group.human_label).sum()
            after=group.calibrated.eq(group.human_label).sum()
            subgroups.append({"Dimension":dimension,"Ausprägung":str(value),"Fälle":len(group),"Original korrekt":before,
                              "Kalibriert korrekt":after,"Original Accuracy":before/len(group),"Kalibriert Accuracy":after/len(group),
                              "Änderung (Prozentpunkte)":(after-before)/len(group)*100})
    subgroups=pd.DataFrame(subgroups)
    fig,axes=plt.subplots(1,3,figsize=(16,5.4),layout="constrained",width_ratios=[1.3,1,1.2])
    for ax,dimension,title in zip(axes,["method","stage_target_words","entity_type"],["Historische Methode","Zielwortzahl","Entitätstyp"]):
        group=subgroups[subgroups.Dimension.eq(dimension)].copy()
        if dimension=="stage_target_words":
            group=group.assign(order=group.Ausprägung.astype(int)).sort_values("order",ascending=False)
        y=np.arange(len(group))
        ax.plot(np.stack([group["Original Accuracy"],group["Kalibriert Accuracy"]])*100,np.stack([y,y]),color="#bbbbbb")
        ax.scatter(group["Original Accuracy"]*100,y,facecolors="white",edgecolors="#555555",s=65,label="Original",zorder=3)
        ax.scatter(group["Kalibriert Accuracy"]*100,y,color="#286a9f",marker="s",s=45,label="Kalibriert",zorder=3)
        ax.set(yticks=y,yticklabels=[f"{v} (n={n})" for v,n in zip(group.Ausprägung,group.Fälle)],xlim=(-3,103),xlabel="Übereinstimmung (%)",title=title)
        ax.invert_yaxis();ax.grid(axis="x",alpha=.15)
    axes[0].legend(loc="lower center",bbox_to_anchor=(.5,1.02),ncols=2,fontsize=10)
    for ax in axes: ax.set_title(ax.get_title(),pad=42)
    savefig(fig,"06_subgruppen")
    display(subgroups.round(4))
    export(subgroups,"subgruppen")

    full_reviewers=completion.loc[completion.vollstaendig.eq(60),"reviewer_id"].tolist()
    full_consensus=consensus_table(annotations[annotations.reviewer_id.isin(full_reviewers)],"human_answer_label")
    two_complete=items.drop(columns=["human_label","tie","votes","max_votes","strict_majority","unanimous"]).merge(
        full_consensus.rename(columns={"label":"human_label"}),on="review_id",validate="one_to_one")
    two_complete=two_complete[two_complete.human_label.isin(LABELS)&~two_complete.tie]
    sets={"Historisch: Ties alphabetisch":items[items.legacy_eligible],
          "Primär: Ties + unclear ausgeschlossen":strict,
          "Absolute Mehrheit + kein unclear":items[items.legacy_eligible & items.strict_majority],
          "Zusätzlich eindeutige gute Fragenqualität":strict[strict.human_question_quality.eq("good") & ~strict.question_tie],
          "Nur zwei vollständige Reviewer, deren Übereinstimmung":two_complete}
    sensitivity_rows=[]
    for name,group in sets.items():
        for column,title in [("model_label","Original"),("calibrated","Kalibriert")]:
            row=metric_row(group,column,name);row["Scorer"]=title;sensitivity_rows.append(row)
    sensitivity=pd.DataFrame(sensitivity_rows)
    display(sensitivity.round(4))
    weighting=[]
    for column,title in [("model_label","Original"),("calibrated","Kalibriert")]:
        group=strict.assign(correct=strict[column].eq(strict.human_label).astype(float))
        weighting.append({"Scorer":title,"Fallgleich gewichtet":group.correct.mean(),
                          "Artikelgleich gewichtet":group.groupby("article_id").correct.mean().mean(),
                          "Faktgleich gewichtet":group.groupby(["article_id","canonical_fact_id"]).correct.mean().mean()})
    weighting=pd.DataFrame(weighting)
    display(weighting.round(4))
    export(sensitivity,"konsens_sensitivitaet")
    export(weighting,"gewichtung_sensitivitaet")

    leave_out=[]
    for article in sorted(items.article_id.unique()):
        group=strict[strict.article_id.ne(article)]
        before=group.model_label.eq(group.human_label).mean();after=group.calibrated.eq(group.human_label).mean()
        leave_out.append({"Typ":"Artikel","Ausgeschlossen":str(article),"Fälle":len(group),"Original":before,"Kalibriert":after,"Delta (pp)":(after-before)*100})
    for reviewer in reviewers:
        rebuilt=consensus_table(annotations[annotations.reviewer_id.ne(reviewer)],"human_answer_label")
        group=items[["review_id","model_label","calibrated"]].merge(rebuilt.rename(columns={"label":"human_label"}),on="review_id",validate="one_to_one")
        group=group[group.human_label.isin(LABELS)&~group.tie]
        before=group.model_label.eq(group.human_label).mean();after=group.calibrated.eq(group.human_label).mean()
        leave_out.append({"Typ":"Reviewer","Ausgeschlossen":reviewer,"Fälle":len(group),"Original":before,"Kalibriert":after,"Delta (pp)":(after-before)*100})
    leave_out=pd.DataFrame(leave_out)
    display(leave_out.round(4))
    export(leave_out,"leave_one_out_stabilitaet")
    fig,ax=plt.subplots(figsize=(10,5.6),layout="constrained")
    y=np.arange(len(leave_out))
    ax.barh(y,leave_out["Delta (pp)"],color=["#286a9f" if t=="Artikel" else "#a8afb8" for t in leave_out.Typ])
    for i,row in leave_out.iterrows():
        ax.text(row["Delta (pp)"]+.4,i,f"{row['Delta (pp)']:+.1f} pp; n={row.Fälle}",va="center",fontsize=10)
    ax.set(yticks=y,yticklabels=leave_out.Typ+": "+leave_out.Ausgeschlossen,
           xlim=(0,max(leave_out["Delta (pp)"])+8),xlabel="Kalibriert minus Original (Prozentpunkte)",title="Stabilität beim Ausschluss einzelner Artikel oder Reviewer")
    ax.invert_yaxis();savefig(fig,"07_leave_one_out")

    participant_rows=[]
    for reviewer in reviewers:
        own=annotations[annotations.reviewer_id.eq(reviewer)&annotations.human_answer_label.isin(LABELS)].merge(
            items[["review_id","calibrated"]],on="review_id",validate="one_to_one")
        other_consensus=consensus_table(annotations[annotations.reviewer_id.ne(reviewer)],"human_answer_label")
        other=own.merge(other_consensus,on="review_id",validate="one_to_one")
        other=other[other.label.isin(LABELS)&~other.tie]
        participant_rows.append({"Reviewer":reviewer,"Modellvergleich n":len(own),
                                 "Original Accuracy":own.model_label.eq(own.human_answer_label).mean(),
                                 "Kalibriert Accuracy":own.calibrated.eq(own.human_answer_label).mean(),
                                 "Fremdkonsens n":len(other),"Übereinstimmung Fremdkonsens":other.label.eq(other.human_answer_label).mean()})
    participant_metrics=pd.DataFrame(participant_rows)
    display(participant_metrics.round(4))
    export(participant_metrics,"reviewer_modell_und_fremdkonsens")
    fig,ax=plt.subplots(figsize=(10,5.2),layout="constrained")
    y=np.arange(len(participant_metrics))
    for offset,column,color,hatch,label in [(-.17,"Original Accuracy","#a8afb8","//","Original"),
                                            (.17,"Kalibriert Accuracy","#286a9f",None,"Kalibriert")]:
        ax.barh(y+offset,participant_metrics[column]*100,height=.30,color=color,hatch=hatch,label=label)
    ax.set(yticks=y,yticklabels=[f"{r} (n={n})" for r,n in zip(participant_metrics.Reviewer,participant_metrics["Modellvergleich n"])],
           xlim=(0,100),xlabel="Übereinstimmung mit eigenen Drei-Klassen-Urteilen (%)",title="Reviewer–Modell-Übereinstimmung auf jeweils verfügbaren Fällen")
    ax.invert_yaxis();ax.legend(loc="lower center",bbox_to_anchor=(.5,1.02),ncols=2);ax.set_title(ax.get_title(),pad=42)
    savefig(fig,"09_reviewer_modell_uebereinstimmung")

    prompt_rows=[]
    for name,folder in [("Prompt v1","optimized_answer_prompt"),("Prompt v2","optimized_answer_prompt_v2")]:
        rerun=read(f"qa/analysis/{folder}/optimized_prompt_answers_vs_human_majority.csv")
        assert len(rerun)==60 and rerun.review_id.is_unique and set(rerun.review_id)==set(items.review_id)
        joined=items[["review_id","human_label","legacy_eligible","strict_eligible"]].merge(rerun,on="review_id",validate="one_to_one")
        joined["original_prediction"]=[classify(r,a,original_retained) for r,a in zip(joined.reference_answer,joined.new_answer)]
        assert joined.original_prediction.eq(joined.new_model_label).all()
        joined["calibrated_prediction"]=[classify(r,a,scorer.retained) for r,a in zip(joined.reference_answer,joined.new_answer)]
        historical=joined[joined.legacy_eligible]
        assert historical.original_prediction.eq(historical.human_label).sum()==(30 if name=="Prompt v1" else 34)
        prompt_rows.append({"Prompt":name,"Historisch Originalscorer n":len(historical),
                           "Historisch korrekt":historical.original_prediction.eq(historical.human_label).sum()})
        for column,title in [("original_prediction","Originalscorer"),("calibrated_prediction","Kalibrierter Scorer")]:
            selected=joined[joined.strict_eligible]
            prompt_rows.append({"Prompt":name,"Scorer":title,"Primär n":len(selected),
                               "Primär korrekt":selected[column].eq(selected.human_label).sum(),"Primär Accuracy":selected[column].eq(selected.human_label).mean()})
    general=read("qa/analysis/three_general_prompt_test/prompt_test_answers.csv")
    assert len(general)==180 and not general[["prompt_variant","review_id"]].duplicated().any()
    for name,group in general.groupby("prompt_variant"):
        assert len(group)==60 and set(group.review_id)==set(items.review_id)
        joined=items[["review_id","human_label","strict_eligible"]].merge(group,on="review_id",validate="one_to_one")
        for function,column in [(original_retained,"original_scorer_label"),(scorer.retained,"final_scorer_label")]:
            reconstructed=[classify(r,a,function) for r,a in zip(joined.reference_answer,joined.answer)]
            assert joined[column].eq(reconstructed).all()
        for column,title in [("original_scorer_label","Originalscorer"),("final_scorer_label","Kalibrierter Scorer")]:
            selected=joined[joined.strict_eligible]
            prompt_rows.append({"Prompt":name,"Scorer":title,"Primär n":len(selected),
                               "Primär korrekt":selected[column].eq(selected.human_label).sum(),"Primär Accuracy":selected[column].eq(selected.human_label).mean()})
    prompt_comparison=pd.DataFrame(prompt_rows)
    display(prompt_comparison)
    export(prompt_comparison,"prompt_scorer_vergleich")

    benchmark=read("qa_spaeterer_antworterabgleich/report/human_validation_subset.csv")
    assert len(benchmark)==120 and not benchmark[["answerer","review_id"]].duplicated().any()
    benchmark=benchmark.merge(items[["review_id","human_label","tie","strict_eligible","article_id","method","stage_target_words","canonical_fact_id","generated_question","reference_answer"]],
                              on="review_id",validate="many_to_one",suffixes=("_benchmark",""))
    assert benchmark.human_majority_label.eq(benchmark.human_label).all()
    assert bool_col(benchmark.human_label_tie).eq(benchmark.tie).all()
    for column in ["article_id","method","canonical_fact_id","reference_answer"]:
        assert benchmark[column+"_benchmark"].eq(benchmark[column]).all(),column
    assert benchmark.stage.eq(benchmark.stage_target_words).all()
    assert benchmark.question.eq(benchmark.generated_question).all()
    benchmark_rows=[]
    for answerer,group in benchmark.groupby("answerer"):
        for name,mask in [("Archiv: ohne Ties (inkl. unclear)",bool_col(group.comparable)),("Harmonisiert: ohne Ties und unclear",group.strict_eligible)]:
            selected=group[mask]
            matches=selected.model_label.eq(selected.human_label).sum()
            benchmark_rows.append({"Antworter":answerer,"Vergleichsset":name,"Fälle":len(selected),"Übereinstimmungen":matches,"Accuracy":matches/len(selected)})
    benchmark_agreement=pd.DataFrame(benchmark_rows)
    assert benchmark_agreement["Übereinstimmungen"].tolist()==[41,41,39,39]
    assert benchmark_agreement["Fälle"].tolist()==[54,52,54,52]
    display(benchmark_agreement.round(4))
    display(benchmark.loc[benchmark.human_label.eq("unclear"),["review_id","answerer","human_label","model_label","comparable","agreement"]])
    display(read("qa_spaeterer_antworterabgleich/report/human_agreement_bootstrap.csv"))
    export(benchmark_agreement,"spaeterer_antworterabgleich")

    overall=read("qa_spaeterer_antworterabgleich/report/overall.csv")
    bootstrap=read("qa_spaeterer_antworterabgleich/report/paired_article_bootstrap.csv")
    display(overall[["answerer","rows","retention_rate","loss_rate","reference_discordant_rate","literal_false_negative_rate","error_rows"]].round(4))
    display(bootstrap.round(4))
    assert overall.rows.eq(2400).all() and overall.error_rows.eq(0).all()
    export(overall,"gesamtbenchmark_kontext")
    export(bootstrap,"gesamtbenchmark_archivierte_intervalle")
    fig,ax=plt.subplots(figsize=(9,4.4),layout="constrained")
    x=np.arange(2)
    for offset,color,mask,label,hatch in [(-.18,"#a8afb8",benchmark_agreement.Vergleichsset.str.startswith("Archiv"),"Archiv: n=54","//"),
                                         (.18,"#286a9f",benchmark_agreement.Vergleichsset.str.startswith("Harmonisiert"),"Drei Klassen: n=52",None)]:
        group=benchmark_agreement[mask]
        bars=ax.bar(x+offset,group.Accuracy*100,width=.32,color=color,hatch=hatch,label=label)
        for bar,row in zip(bars,group.itertuples()):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{row.Übereinstimmungen}/{row.Fälle}\n{row.Accuracy:.1%}",ha="center",fontsize=10)
    ax.set(xticks=x,xticklabels=["GLM","Qwen3.8"],ylim=(0,105),ylabel="Übereinstimmung mit historischen Human-Labels (%)",
           title="Antworterabgleich auf wiederverwendeten Human-Fällen")
    ax.legend(loc="upper center",ncols=2)
    savefig(fig,"08_spaeterer_antworterabgleich")

    final_protocol=json.loads((DATA/"finale_pipeline/frozen_downstream_protocol.json").read_text())
    qa_manifest=json.loads((DATA/"finale_pipeline/qa_manifest.json").read_text())
    protocol_summary=pd.DataFrame({"Parameter":["Faktgenerator","Kandidaten je Artikel","Kanonische Fakten je Artikel","QA-Antworter","Exakter zusammenhängender Beleg","Erwartete QA-Einheiten","Bootstrap-Einheit","Antwortprotokoll"],
                                   "Wert":[final_protocol["fact_generator"],final_protocol["candidate_fact_pool_per_article"],final_protocol["canonical_facts_per_article"],
                                           final_protocol["qa_answerer"],final_protocol["qa_exact_contiguous_support_required"],final_protocol["expected_summary_qa_units"],
                                           final_protocol["bootstrap_unit"],qa_manifest["answer_prompt_protocol"]]})
    display(protocol_summary)
    assert final_protocol["expected_summary_qa_units"]==100*6*4*20
    export(protocol_summary,"finale_pipeline_abgrenzung")

    summary_values={
        "possible_reviews":len(annotations),"complete_reviews":int(annotations.complete.sum()),
        "question_good":int(qcounts["good"]),"question_good_or_partial":int(qcounts["good"]+qcounts["partial"]),
        "question_ratings":len(valid_question),"answer_ratings":len(valid_answer),"comparable_individual_ratings":len(comparable_rows),
        "original_row_matches":int(row_matches),"calibrated_row_matches":int(calibrated_rows.calibrated.eq(calibrated_rows.human_answer_label).sum()),
        "answer_alpha_nominal":float(answer_alpha),"answer_alpha_without_unclear":float(answer_alpha_three),
        "question_alpha_nominal":float(reliability.loc[reliability.Dimension.eq("question_quality"),"Nominales Alpha"].iloc[0]),
        "strict_items":len(strict),"strict_original_matches":int(strict.model_label.eq(strict.human_label).sum()),
        "strict_calibrated_matches":int(strict.calibrated.eq(strict.human_label).sum()),
        "strict_original_kappa":float(cohen_kappa_score(strict.human_label,strict.model_label,labels=LABELS)),
        "strict_calibrated_kappa":float(cohen_kappa_score(strict.human_label,strict.calibrated,labels=LABELS)),
        "strict_original_macro_f1":float(f1_score(strict.human_label,strict.model_label,labels=LABELS,average="macro")),
        "strict_calibrated_macro_f1":float(f1_score(strict.human_label,strict.calibrated,labels=LABELS,average="macro")),
        "absolute_majority_cases":int((items.legacy_eligible&items.strict_majority).sum()),
        "article_leave_out_delta_min_pp":float(leave_out.loc[leave_out.Typ.eq("Artikel"),"Delta (pp)"].min()),
        "article_leave_out_delta_max_pp":float(leave_out.loc[leave_out.Typ.eq("Artikel"),"Delta (pp)"].max())}
    (OUT/"gepruefte_kennzahlen.json").write_text(json.dumps(summary_values,ensure_ascii=False,indent=2))
    export(items,"vollstaendige_fallauswertung")
    wide = items[["review_id","article_id","canonical_fact_id","method","stage_target_words","entity_type","model_label","first_patch","calibrated","human_label","tie","strict_eligible"]].copy()
    for field in ["human_answer_label","question_quality"]:
        ratings=annotations.pivot(index="review_id",columns="reviewer_id",values=field)
        ratings.columns=[f"{r}_{field}" for r in ratings.columns]
        wide=wide.merge(ratings,on="review_id",validate="one_to_one")
    export(wide,"qa_fallmatrix_alle_reviewer")
    display(pd.Series(summary_values,name="Wert").to_frame().round(4))
    print("Alle Replikations- und Konsistenzprüfungen bestanden; Exporte:",OUT.name)

if __name__ == '__main__':
    from offline_guard import install
    install(allow_statistics_workers='--recompute-regression-bootstrap' in sys.argv)
    main(sys.argv[1], '--recompute-regression-bootstrap' in sys.argv, int(sys.argv[2]))
