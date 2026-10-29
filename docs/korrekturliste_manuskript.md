# Korrekturliste zum Abgabemanuskript

Grundlage: `2910_Masterarbeit.docx`, vom Autor bestätigt, geprüft am 29.10.2026.
Das Manuskript wurde ausschließlich gelesen. Seitenzahlen stammen aus den
gespeicherten Verzeichnissen; Abschnitt und Textzitat sind die maßgebliche
Fundstelle. Zahlenbelege: [Manuskriptabgleich](abgleich_manuskript.md) und
[797 numerische Einzelprüfungen](belege_abgleich_tabellen.csv).

## Numerische Korrekturen

| Fundstelle | Ist-Text / Ist-Wert | Vorschlag | Beleg |
|---|---|---|---|
| Tabelle 16, C.2, S. 81; Auto-CoT − Few-Shot-CoT, 300 Wörter, untere Grenze | `4,89` | **4,98** | `results/02_faktenretention/vier_gepaarte_kontraste_pp_95ci.csv`: 4,9803289473684185; unabhängige Rohdaten-Nachrechnung im Abgleich. Die Grenze 4,89 gilt für 500 Wörter. |
| Tabelle 18, D.1, S. 84; gleicher Kontrast/Stufe, unadjustiert unten | `4,89` | **4,98** | `results/06_laengenregression/haupttabelle_16_originale_und_laengenkontrollierte_kontraste.csv`, `original_ci_low_pp`. Die ursprüngliche Vermutung der Aufgabenbeschreibung ist für diesen Repo-Stand umgekehrt. |
| Tabelle 18; Auto-CoT − Few-Shot-CoT, 150 Wörter, „Unadjustiert, Supportartikel“ | `3,91` | **3,97** | Dieselbe CSV, `raw_same_articles_pp=3.969860548807917`; ungerundeter Wert auch in den Einzelbelegen. Keine Änderung des längenkontrollierten Kontrasts. |
| Kapitel 4.3.2, Absatz „Gegenüber der beobachteten Retention …“, S. 48/49 | „Die gepaarten Abstände zwischen den führenden generativen Verfahren … 0,08pp [-1,95; 2,08] bei 500“ | „Bei 500 Zielwörtern betrug der adjustierte Abstand **Zero-Shot-CoT minus Auto-CoT 0,08 pp [−1,95; 2,08]**. Diese beiden Verfahren waren dort nicht die führenden generativen Verfahren.“ Die folgenden stufenspezifischen Kontraste jeweils ausdrücklich benennen. | `berichtete_adjustierte_abstaende_100_98.csv`. Automate-CoT und Zero-Shot-CoT führen bei 500; deskriptiver Abstand 5,643294 pp. Dafür ist in dieser Tabelle kein Intervall ausgewiesen. |
| Kapitel 4.1.1, Absatz „Die Teilnehmer unterschieden sich …“, S. 41 | „Cohen-Kappa-Werte lagen zwischen 0,234 und 0,434“ | „… zwischen **0,234 und 0,435**.“ | `results/01_menschliche_validierung/relevanz_kappa_paare.csv`; P3/P5 ist das Maximum und wird auch in Abb. 20 als 0,435 gezeigt. |

## Begriffliche Präzisierungen und Reproduzierbarkeit

| Fundstelle | Ist-Text | Vorschlag | Begründung |
|---|---|---|---|
| Kapitel 4.2.1, Absatz „Lead-k erreichte …“, S. 43 | „Bei allen fünf generativen Verfahren tritt der größte Rückgang bereits auf der ersten Kompressionsstufe mit 500 Wörtern auf.“; später „Der nächstgrößte Rückgang …“ | „Auf der ersten beobachteten Zusammenfassung (500 Zielwörter) wurden je nach generativem Verfahren 26,5–44,9 % der Ausgangsfakten erstmals nicht als retained bewertet. **Unter den gemessenen Stufenübergängen** war der mittlere Rückgang von 150 auf 75 Zielwörter am größten. Ein Rückgang vom Original zur 500-Wort-Stufe wurde mangels Original-QA-Baseline nicht gemessen.“ | Anteile erstmaliger Nicht-Retained-Urteile sind in Notebook 04 belegt; gemessene mittlere Retentionsänderungen in Notebook 02. Sie haben unterschiedliche Definitionen. |
| Kapitel 5.1, Absatz „Bei allen generativen Verfahren …“, S. 55/56 | „größte Rückgang der mittleren Faktenretention bei der ersten Kompression“; „nächststärkste Rückgang“ | Dieselbe Unterscheidung wie in Kapitel 4.2.1 übernehmen: erster Nichtnachweis bei 500 versus stärkster gemessener Übergang 150→75. | Keine Original-QA-Baseline. Kapitel 5.1, F1-Schlussabsatz, und Kapitel 7 verwenden bereits die vorsichtigere Aussage über erstmalige Nichtnachweise. |
| Tabelle 19, D.2, S. 85, „Mindestunterstützung“; ergänzend Kapitel 3.9.2 | „≥20 Artikel und ≥20 Beobachtungen je Verfahren im Überlappungsbereich“ | „≥20 Beobachtungen je Verfahren im gemeinsamen 5.–95.-Perzentilbereich und ≥20 **lokal gestützte gepaarte Artikel**; an beiden Referenzwortzahlen je Artikel mindestens fünf Ausgaben **jedes** Verfahrens im Band **max(5 Wörter, 10 % der Wortzahl)**.“ | Notebook 06, `support_for`. Kapitel 3.9.2 nennt fünf Ausgaben, definiert aber die Bandbreite nicht. |
| Kapitel 4.1.2 und 4.4.1/Tabelle 7 | `22 von 1.999` versus `21 von 1.407` | Nach Tabelle 7 ergänzen: „Diese Bezugsgruppe umfasst eindeutige erste Nicht-Retained-Urteile vor der 75-Wort-Stufe. Die Konsistenzprüfung in 4.1.2 erfasst dagegen alle Fakten mit mindestens einem gültigen Paar unter den sechs möglichen Stufenpaaren; daher 22/1.999 statt 21/1.407.“ | `lead_k_rueckwechsel_bezugsgruppen.csv`. Beide Zahlen sind korrekt und bleiben erhalten. |
| Kapitel 3.6.1, Absatz „Anschließend erfolgt …“ | „Daraufhin werden Fragen auf Duplikate geprüft und entfernt.“ | „Duplikate werden anhand identischer **normalisierter Fakttexte** ausgeschlossen.“ | `src/faktenretention/facts.py:173`: Duplikatschlüssel ist `norm_fact`, kein normalisierter Fragetext. Tabelle 12 beschreibt dies bereits korrekt. |
| Kapitel 3.4.1, Absatz „Das Durchlaufen …“ | „… wird dieser einfach auf die nötige Textlänge der Stufe abgeschnitten“ (Bezug auf Ursprungstext) | „Die ersten k Wörter werden mit derselben Wortzählregel aus der Vorstufe übernommen. Wegen der verschachtelten Präfixe ist dies inhaltlich identisch zum direkten Abschneiden des Originaltexts.“ | Die Refactor-Pipeline reicht bei Lead-k tatsächlich die Vorstufe weiter; `text_rules.lead_k` und Notebook 04 prüfen die Gleichheit. Kein Ergebnisunterschied. |
| Tabelle 10, A.2, S. 74 | Thinking-Budgets sind genannt, Reflection-Interval fehlt | Optional eigene Spalte/Fußnote: „Reflection-Interval bei aktivem Wrapper: 512 Tokens. Hauptversuch Zero-Shot: 1.024 Thinking-Tokens; Kandidaten und Automate-Optimierung: 512. Die Angaben sind Promptinstruktionen.“ | `seed_raw` hat allgemein Defaults 512/128, die Aufrufer setzen Reflection ausdrücklich auf 512; der Hauptversuch setzt Thinking auf 1.024. Ohne aktive CoT-Verpackung beträgt das Budget 0. |
| Anhang A.1, „Digitaler Anhang“, S. 74 | „… sämtlichen Code für das Reproduzieren … ebenso wie sämtliche Daten …“ | „Der digitale Anhang enthält den Code und die dokumentierten Daten zur Offline-Reproduktion der Auswertung. Neue Modellläufe sind eingeschränkt reproduzierbar; vollständige Cluster-Embeddings und QA-Retry-Zwischenantworten fehlen. Die Modellmetrikinferenz wird bei der Offline-Auswertung nicht wiederholt.“ | README und `docs/modelle_und_beschaffung.md`; keine vollständig rekonstruierbare historische Inferenz behaupten. |
| Tabelle 26, F, S. 91 | `data/` vorhanden, aber `docs/` und die `data/`-Unterordner fehlen | `docs/`, `data/human_qa/` (mit `returns/` und `reviewer_app/`), `data/methoden/` und `data/model_metadata/` (mit `manifests/`, `blobs/`, `provenance/`) ergänzen. `data/prepared/` als lokal erzeugt kennzeichnen; LICENSE und CITATION.cff ergänzen. | Tatsächliche Repo-Struktur; `docs/` steht bereits in der bisherigen README, die Unterordner wurden nun ergänzt. |
| Anhang A.1/F, nach finaler Freigabe | Repository-URL ohne feste Codefassung | Finalen Commit-Hash und vorgeschlagenen Tag **v1.0-abgabe** nennen; Tag erst nach Übernahme und Prüfung der Korrekturen erstellen. | Exakte Befehle stehen in der README. Es wurde kein Tag/Release angelegt und nichts gepusht. |

## Erledigte Verdachtsfälle ohne nötige Zahlenkorrektur

- **Anhang E.1 existiert** und enthält Tabelle 21. Der Fließtext verweist auf
  diesen Abschnitt; eine fehlende „Tabelle E.1“ wird nicht referenziert.
  Die missverständliche frühere Repo-Zuordnung zum Kendall-Export wurde korrigiert.
- Kapitel **5, 7, Zusammenfassung und Abstract enthalten keine 0,08-pp-Angabe**
  zu den führenden Verfahren. Die problematische Formulierung steht in 4.3.2.
- Beide Lead-k-Zahlen sind korrekt; vorgeschlagen wird nur eine Erklärung.
- Notebook 02 wurde im Repo textlich korrigiert; das Manuskript blieb unverändert.

## Offen / vom Autor zu klären

Die vorgeschlagenen Manuskriptkorrekturen selbst übernehmen oder begründet
verwerfen. Anschließend alle Verzeichnisse aktualisieren und die hier nach
Abschnitt/Tabelle lokalisierten Stellen gegen das finale PDF prüfen. Im
gespeicherten Anhangsverzeichnis steht E.1 auf S. 86, im Tabellenverzeichnis
Tabelle 21 auf S. 87; eine neue PDF-Paginierung war hier nicht verfügbar.
