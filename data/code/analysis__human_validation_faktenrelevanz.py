# Statische Originalauswertung; Eingaben/Ausgaben relativ zur übergebenen Arbeitskopie.
# Keine LLM-Aufrufe. Standard: vorhandene Regressionscaches verwenden.
from pathlib import Path
import os,sys

def main(work, recompute_bootstrap=False, workers=1):
    WORK=__import__('pathlib').Path(work).resolve()
    __import__('os').chdir(WORK/('supplement' if 'human_validation_faktenrelevanz'[0].isdigit() else 'project/02_Auswertung'))
    from pathlib import Path

    import pandas as pd

    CSV_PATH = Path("human_validation_faktenrelevanz_ergebnisse.csv")
    if not CSV_PATH.exists() and (Path("..") / CSV_PATH).exists():
        CSV_PATH = Path("..") / CSV_PATH
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"{CSV_PATH} wurde nicht gefunden. Lege Notebook und CSV in denselben Ordner."
        )

    human_validation = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    human_validation.head()

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    from IPython.display import display, Markdown
    from itertools import combinations
    from collections import Counter

    df = human_validation.copy()
    reviewers = [f"bewertung_teilnehmer_{i}" for i in range(1, 6)]
    labels = ["Peripheral", "Supporting", "Core"]
    colors = ["#a8afb8", "#e4ad42", "#286a9f"]
    articles = sorted(df.artikel_id.unique())
    assert len(df) == 168 and df.fakt_id.is_unique and not df.isna().any().any()
    assert df[reviewers + ["median", "bewertung_modell"]].isin([0, 1, 2]).all().all()
    assert df[reviewers].median(axis=1).eq(df["median"]).all()
    assert df.in_top_20_laut_modell.eq(df.rang_laut_modell.le(20)).all()
    assert df.groupby("artikel_id").in_top_20_laut_modell.sum().eq(20).all()

    FIG_DIR = Path("human_validation_grafiken")
    FIG_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold",
                         "savefig.facecolor": "white"})
    def save_figure(fig, name):
        for axis in fig.axes:
            if axis.get_legend() is not None:
                axis.set_title(axis.get_title(), pad=48)
        fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
        plt.show()


    def ordinal_alpha(ratings):
        """Ordinales Krippendorff-Alpha; vollständige, gleich große Ratergruppen."""
        counts = Counter(ratings.ravel().tolist())
        def distance(a, b):
            if a == b:
                return 0.0
            lo, hi = sorted((a, b))
            return (sum(counts[c] for c in range(lo, hi + 1))
                    - (counts[lo] + counts[hi]) / 2) ** 2
        observed = np.mean([distance(a, b) for unit in ratings
                            for i, a in enumerate(unit) for j, b in enumerate(unit) if i != j])
        n = ratings.size
        expected = sum(counts[a] * counts[b] * distance(a, b)
                       for a in counts for b in counts) / (n * (n - 1))
        return 1 - observed / expected

    def ndcg(values, original=False):
        values = np.asarray(values)
        discounts = np.log2(np.arange(2, min(20, len(values)) + 2))
        gains = 2 ** (values + 1) - 1 if original else values
        ideal = np.sort(gains)[::-1][:20]
        return (gains[:20] / discounts).sum() / (ideal / discounts).sum()

    article_rows = []
    for article, group in df.groupby("artikel_id"):
        ranked = group.sort_values("rang_laut_modell")
        top = ranked[ranked.in_top_20_laut_modell]
        core_total = group["median"].eq(2).sum()
        article_rows.append({"Artikel": article, "Kandidaten": len(group),
            "Core gesamt": core_total, "Core in Top 20": top["median"].eq(2).sum(),
            "Precision@20": top["median"].ge(1).mean(),
            "Core-Recall@20": top["median"].eq(2).sum() / core_total,
            "nDCG@20 (linear 0/1/2)": ndcg(ranked["median"]),
            "nDCG@20 (Original 1/3/7)": ndcg(ranked["median"], original=True),
            "Spearman Rang–Median": ranked[["rang_laut_modell", "median"]].corr(method="spearman").iloc[0, 1]})
    article_metrics = pd.DataFrame(article_rows).set_index("Artikel")
    top20 = df[df.in_top_20_laut_modell]
    alpha = ordinal_alpha(df[reviewers].to_numpy())
    accuracy = accuracy_score(df["median"], df.bewertung_modell)
    kappa = cohen_kappa_score(df["median"], df.bewertung_modell, weights="linear")
    macro_f1 = f1_score(df["median"], df.bewertung_modell, average="macro")
    unanimous = df[reviewers].nunique(axis=1).eq(1).sum()
    summary = pd.Series({"Fakten": len(df), "Artikel": len(articles), "Einzelurteile": df[reviewers].size,
        "Precision@20 (gepoolt)": top20["median"].ge(1).mean(),
        "Core-Recall@20 (gepoolt)": top20["median"].eq(2).sum() / df["median"].eq(2).sum(),
        "Core-Recall@20 (Artikelmittel)": article_metrics["Core-Recall@20"].mean(),
        "nDCG@20 (linear, Artikelmittel)": article_metrics["nDCG@20 (linear 0/1/2)"].mean(),
        "nDCG@20 (Original, Artikelmittel)": article_metrics["nDCG@20 (Original 1/3/7)"].mean(),
        "Modell–Mensch: exakte Übereinstimmung": accuracy,
        "Modell–Mensch: linear gewichtetes Kappa": kappa,
        "Modell–Mensch: Macro-F1": macro_f1,
        "Ordinales Krippendorff-Alpha": alpha,
        "Vollständig übereinstimmende Fakten": unanimous}, name="Wert")
    display(summary.to_frame().round(4))
    display(article_metrics.round(4))
    summary.to_csv(FIG_DIR / "kennzahlen.csv")
    article_metrics.to_csv(FIG_DIR / "kennzahlen_je_artikel.csv")
    # Abgleich mit den zuvor dokumentierten finalen Ergebnissen.
    assert np.isclose(alpha, 0.458867, atol=1e-6)
    assert np.isclose(kappa, 0.542044, atol=1e-6)
    assert np.isclose(article_metrics["nDCG@20 (Original 1/3/7)"].mean(), 0.925537, atol=1e-6)


    groups = {"Kandidatenpool (n=168)": df, "Top 20 (n=100)": top20,
              "Außerhalb Top 20 (n=68)": df[~df.in_top_20_laut_modell]}
    composition = pd.DataFrame({name: g["median"].value_counts().reindex(range(3), fill_value=0)
                                for name, g in groups.items()}).T
    fig, ax = plt.subplots(figsize=(10, 4.2), layout="constrained")
    left = np.zeros(len(composition))
    for category in range(3):
        percentages = composition[category].to_numpy() / composition.sum(axis=1).to_numpy() * 100
        ax.barh(composition.index, percentages, left=left, color=colors[category],
                edgecolor="white", label=labels[category])
        for i, (pct, n) in enumerate(zip(percentages, composition[category])):
            if 0 < pct < 8:
                ax.text(101, i, f"{pct:.1f}% (n={n})", ha="left", va="center", fontsize=10)
            elif pct >= 8:
                ax.text(left[i] + pct / 2, i, f"{pct:.1f}%\n(n={n})", ha="center", va="center",
                        color="white" if category == 2 else "#222222", fontsize=10 if pct >= 8 else 7)
        left += percentages
    ax.set(xlim=(0, 100), xlabel="Anteil der Fakten (%)", title="Menschliche Relevanz nach Modellselektion")
    ax.invert_yaxis()
    ax.legend(ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    save_figure(fig, "01_relevanz_selektion")
    display(Markdown("**Ergebnis:** 88 von 100 ausgewählten Fakten sind Core oder Supporting (88 %). "
        "Im Kandidatenpool sind es 112/168 (66,7 %), außerhalb der Top 20 24/68 (35,3 %). "
        "Die Top 20 enthalten 33 Core-, 55 Supporting- und 12 Peripheral-Fakten. "
        "Dieser Vergleich beschreibt die Selektionsleistung innerhalb des eingefrorenen Pools."))


    by_article = pd.crosstab(top20.artikel_id, top20["median"]).reindex(index=articles, columns=range(3), fill_value=0)
    fig, (ax, recall_ax) = plt.subplots(1, 2, figsize=(12, 4.8), width_ratios=[1.7, 1], layout="constrained")
    y = np.arange(len(articles))
    left = np.zeros(len(articles))
    for category in range(3):
        values = by_article[category].to_numpy() * 5
        ax.barh(y, values, left=left, color=colors[category], edgecolor="white", label=labels[category])
        for i, (value, n) in enumerate(zip(values, by_article[category])):
            if n:
                ax.text(left[i] + value / 2, i, str(n), ha="center", va="center",
                        color="white" if category == 2 else "#222222")
        left += values
    ax.set(yticks=y, yticklabels=articles, xlim=(0, 100), xlabel="Top-20-Anteil (%)",
           title="Zusammensetzung der Top 20")
    ax.invert_yaxis()
    ax.legend(ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=10)
    recall = article_metrics["Core-Recall@20"].to_numpy() * 100
    recall_ax.barh(y, recall, color=colors[2], height=0.55)
    for i, (value, row) in enumerate(zip(recall, article_metrics.to_dict("records"))):
        recall_ax.text(3, i, f"{int(row['Core in Top 20'])}/{int(row['Core gesamt'])} ({value:.1f}%)",
                       va="center", color="white")
    recall_ax.set(yticks=y, yticklabels=[], xlim=(0, 100), xlabel="Core-Recall@20 (%)",
                  title="Erfasste Core-Fakten")
    recall_ax.invert_yaxis()
    save_figure(fig, "02_auswahl_je_artikel")
    display(Markdown("**Ergebnis:** Precision@20 reicht von 65 % bis 100 %. Artikel `cnndm-train-141898` "
        "enthält sieben Peripheral-Fakten in den Top 20, erfasst jedoch alle neun Core-Fakten. "
        "Insgesamt werden 33/35 Core-Fakten erfasst (94,3 % gepoolt); der ungewichtete Artikelmittelwert "
        "des Core-Recall beträgt 93,8 %. Zwei Core-Fakten liegen außerhalb der Top 20."))


    confusion = pd.crosstab(df["median"], df.bewertung_modell).reindex(index=range(3), columns=range(3), fill_value=0)
    row_pct = confusion.div(confusion.sum(axis=1), axis=0) * 100
    annotations = np.array([[f"{confusion.iloc[i,j]}\n({row_pct.iloc[i,j]:.1f}%)"
                              for j in range(3)] for i in range(3)])
    fig, ax = plt.subplots(figsize=(7.2, 5.4), layout="constrained")
    sns.heatmap(row_pct, annot=annotations, fmt="", cmap="Blues", vmin=0, vmax=100,
                xticklabels=labels, yticklabels=[f"{labels[i]} (n={confusion.sum(axis=1).iloc[i]})" for i in range(3)],
                linewidths=1, cbar_kws={"label": "Anteil innerhalb der Human-Klasse (%)"}, ax=ax)
    ax.set(xlabel="Modellbewertung", ylabel="Menschlicher Median", title="Übereinstimmung der Relevanzlabels")
    ax.tick_params(axis="y", rotation=0)
    save_figure(fig, "03_modell_mensch_konfusion")
    display(Markdown(f"**Ergebnis:** Exakte Übereinstimmung: {accuracy:.1%} (112/168), "
        f"linear gewichtetes Kappa: {kappa:.3f}, Macro-F1: {macro_f1:.3f}. "
        "Das Modell stuft 19/35 menschliche Core-Fakten niedriger ein (16 Supporting, 3 Peripheral). "
        "Kein menschlicher Peripheral-Fakt wird als Core bewertet. Insgesamt unterschätzt das Modell "
        "48 Fakten und überschätzt acht; 112 stimmen überein. Das Ranking und die Top-20-Auswahl "
        "funktionieren daher besser als die exakte Drei-Klassen-Kalibrierung."))


    reviewer_counts = pd.DataFrame({r: df[r].value_counts().reindex(range(3), fill_value=0) for r in reviewers}).T
    pair_kappa = pd.DataFrame(np.nan, index=range(1, 6), columns=range(1, 6))
    for i, j in combinations(range(1, 6), 2):
        pair_kappa.loc[i, j] = pair_kappa.loc[j, i] = cohen_kappa_score(df[reviewers[i-1]], df[reviewers[j-1]], weights="linear")
    fig, (ax, k_ax) = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    left = np.zeros(5)
    for category in range(3):
        values = reviewer_counts[category].to_numpy() / len(df) * 100
        ax.barh(range(5), values, left=left, color=colors[category], edgecolor="white", label=labels[category])
        for i, value in enumerate(values):
            ax.text(left[i] + value/2, i, f"{value:.1f}%", ha="center", va="center", fontsize=10,
                    color="white" if category == 2 else "#222222")
        left += values
    ax.set(yticks=range(5), yticklabels=[f"Teilnehmer {i}" for i in range(1, 6)], xlim=(0, 100),
           xlabel="Anteil der 168 Bewertungen (%)", title="Individuelle Labelverteilungen")
    ax.invert_yaxis()
    ax.legend(ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=10)
    sns.heatmap(pair_kappa, mask=pair_kappa.isna(), annot=True, fmt=".2f", vmin=0, vmax=1,
                cmap="Blues", linewidths=1, cbar_kws={"label": "Linear gewichtetes Kappa"}, ax=k_ax)
    k_ax.set(xlabel="Teilnehmer", ylabel="Teilnehmer", title="Paarweise Übereinstimmung")
    k_ax.tick_params(axis="y", rotation=0)
    save_figure(fig, "04_teilnehmer_reliabilitaet")
    display(Markdown(f"**Ergebnis:** Ordinales Krippendorff-Alpha = {alpha:.3f}; "
        f"alle fünf Teilnehmer stimmen bei {unanimous}/168 Fakten ({unanimous/168:.1%}) überein. "
        "Der Core-Anteil reicht von 12,5 % (Teilnehmer 1) bis 39,9 % (Teilnehmer 5). "
        "Die paarweisen gewichteten Kappas liegen zwischen 0,234 und 0,434. "
        "Diese Streuung begrenzt die Interpretation des Medians als fehlerfreie Ground-Truth."))


    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    rank_matrix = df.pivot(index="artikel_id", columns="rang_laut_modell", values="median").reindex(index=articles, columns=range(1, 35))
    fig, ax = plt.subplots(figsize=(12, 4.2), layout="constrained")
    cmap = ListedColormap(colors)
    cmap.set_bad("white")
    ax.imshow(rank_matrix.to_numpy(), aspect="auto", cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3))
    ax.axvline(19.5, color="#222222", linewidth=2, linestyle="--")
    ax.set(yticks=range(5), yticklabels=articles,
           xticks=[0,4,9,14,19,24,29,33], xticklabels=[1,5,10,15,20,25,30,34],
           xlabel="Modellrang (1 = höchste Relevanz)", title="Menschlicher Median entlang des Modellrankings")
    ax.legend(handles=[Patch(facecolor=colors[i], label=labels[i]) for i in range(3)]
              + [Patch(facecolor="white", edgecolor="#999999", label="Kein Fakt")],
              ncols=4, loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=10)
    save_figure(fig, "05_relevanz_nach_modellrang")
    display(Markdown(f"**Ergebnis:** Mittleres nDCG@20 mit linearem Gain 0/1/2 = "
        f"{article_metrics['nDCG@20 (linear 0/1/2)'].mean():.3f}. "
        f"Das ursprüngliche nDCG mit Gain 1/3/7 beträgt {article_metrics['nDCG@20 (Original 1/3/7)'].mean():.3f}. "
        f"Die mittlere Spearman-Korrelation zwischen Modellrang und Human-Median beträgt "
        f"{article_metrics['Spearman Rang–Median'].mean():.3f}. "
        "Das negative Vorzeichen ist erwartungsgemäß: Ein kleiner Rang bedeutet hohe Relevanz, "
        "ein hoher Humanscore ebenfalls. Die beiden nDCG-Werte verwenden verschiedene Gain-Definitionen "
        "und sollten im Ergebnisteil nicht austauschbar verwendet werden."))


    participant_rows = []
    for number, column in enumerate(reviewers, 1):
        human = df[column]
        model = df["bewertung_modell"]
        participant_rows.append({
            "Teilnehmer": f"Teilnehmer {number}", "Fakten": len(df),
            "Identische Bewertungen": int(human.eq(model).sum()),
            "Exakte Übereinstimmung": accuracy_score(human, model),
            "Cohen-Kappa (linear gewichtet)": cohen_kappa_score(human, model, labels=[0, 1, 2], weights="linear"),
            "Macro-F1": f1_score(human, model, labels=[0, 1, 2], average="macro", zero_division=0),
            "KI niedriger": int(model.lt(human).sum()),
            "KI höher": int(model.gt(human).sum()),
            "Relevanter Anteil der Top 20": df.loc[df.in_top_20_laut_modell, column].ge(1).mean(),
        })
    participant_model_metrics = pd.DataFrame(participant_rows).set_index("Teilnehmer")
    assert participant_model_metrics[["Identische Bewertungen", "KI niedriger", "KI höher"]].sum(axis=1).eq(len(df)).all()
    display(participant_model_metrics.round(4))
    participant_model_metrics.to_csv(FIG_DIR / "teilnehmer_ki_uebereinstimmung.csv", encoding="utf-8-sig")

    fig, (agreement_ax, kappa_ax) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    y = np.arange(5)
    agreement = participant_model_metrics["Exakte Übereinstimmung"].to_numpy() * 100
    agreement_ax.barh(y, agreement, color=colors[2], height=0.6)
    for i, (value, count) in enumerate(zip(agreement, participant_model_metrics["Identische Bewertungen"])):
        agreement_ax.text(2, i, f"{value:.1f}% ({count}/168)", color="white", va="center")
    agreement_ax.axvline(accuracy * 100, color="#333333", linestyle="--", linewidth=1.3,
                         label=f"KI–Median: {accuracy:.1%}")
    agreement_ax.set(yticks=y, yticklabels=participant_model_metrics.index,
                     xlim=(0, 100), xlabel="Identische Kategorien (%)", title="Exakte Übereinstimmung mit der KI")
    agreement_ax.invert_yaxis()
    agreement_ax.legend(loc="lower right", fontsize=9)
    weighted_kappas = participant_model_metrics["Cohen-Kappa (linear gewichtet)"].to_numpy()
    kappa_ax.barh(y, weighted_kappas, color=colors[2], height=0.6)
    for i, value in enumerate(weighted_kappas):
        kappa_ax.text(0.02, i, f"{value:.3f}", color="white", va="center")
    kappa_ax.axvline(kappa, color="#333333", linestyle="--", linewidth=1.3, label=f"KI–Median: {kappa:.3f}")
    kappa_ax.set(yticks=y, yticklabels=[], xlim=(0, 1), xlabel="Linear gewichtetes Cohen-Kappa",
                 title="Zufallsbereinigte Übereinstimmung")
    kappa_ax.invert_yaxis()
    kappa_ax.legend(loc="lower right", fontsize=9)
    save_figure(fig, "06_teilnehmer_ki_uebereinstimmung")

    best_exact = participant_model_metrics["Exakte Übereinstimmung"].idxmax()
    best_kappa = participant_model_metrics["Cohen-Kappa (linear gewichtet)"].idxmax()
    lowest = participant_model_metrics["Exakte Übereinstimmung"].idxmin()
    display(Markdown(
        f"**Ergebnis:** Die exakte Übereinstimmung der einzelnen Teilnehmer mit der KI reicht von "
        f"{participant_model_metrics['Exakte Übereinstimmung'].min():.1%} bis "
        f"{participant_model_metrics['Exakte Übereinstimmung'].max():.1%}. "
        f"Die höchste exakte Übereinstimmung erreicht {best_exact}, die niedrigste {lowest}. "
        f"Das linear gewichtete Kappa reicht von {weighted_kappas.min():.3f} bis {weighted_kappas.max():.3f}; "
        f"den höchsten Wert erreicht {best_kappa}. "
        "Die gestrichelten Linien zeigen den Vergleich der KI mit dem menschlichen Median als Referenz. "
        "Der Median wird aus diesen fünf Teilnehmerurteilen gebildet und ist keine unabhängige weitere Bewertung. "
        "Eine hohe KI-Übereinstimmung bedeutet nicht automatisch eine bessere menschliche Bewertung."
    ))

if __name__ == '__main__':
    from offline_guard import install
    install(allow_statistics_workers='--recompute-regression-bootstrap' in sys.argv)
    main(sys.argv[1], '--recompute-regression-bootstrap' in sys.argv, int(sys.argv[2]))
