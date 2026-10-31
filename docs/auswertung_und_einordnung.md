# Auswertungsgrundlagen und Hinweise

Diese Ergänzung zur [README](../README.md) beschreibt den Datenstand,
die wissenschaftlichen Definitionen und Hinweise zum Manuskript.
Der historische [Manuskriptabgleich](abgleich_manuskript.md) bezieht sich auf
`2910_Masterarbeit.docx` vom 29.10.2026. Ein später lokal vorliegendes
PDF-Manuskript wurde an ausgewählten Korrekturstellen nachgeprüft; beide
Manuskripte sind keine versionierten Bestandteile dieses Repositories. Der
Abgleich unterscheidet historische und spätere Befunde.

## Datenstand und Kennzahlen

`data/quellen.json` ordnet Quellen, SHA-256, Datenstand und Parameterbelege zu.
Die fünf Hauptdateien enthalten Artikelmetadaten, kanonische Fakten,
Zusammenfassungen, QA-Antworten und gespeicherte Modellmetriken.
`generierungszustaende.jsonl.gz` enthält die Generierungszustände und die
gespeicherten rekursiven Eingaben ab der zweiten Stufe; Originalartikel und
Highlights sowie die ersten Eingabetexte wurden durch Platzhalter ersetzt.
Verwendet wird ausschließlich `aktuell_20260930`: die ursprünglichen vier Verfahren
plus die später ersetzten Plain-/Few-Shot-CoT-Ketten vom 30. September, abgeschlossen
am 1. Oktober. Alte und neue Fassungen werden nicht gepoolt.

- 100 Artikel × 6 Verfahren × 4 Stufen = 2.400 Ausgaben; 2.000 feste Ausgangsfakten.
- 48.000 QA-Positionen = 47.804 gültige Urteile + 196 technische Fehlwerte.
  Die 40 regelbasierten NOT_FOUND-Urteile zweier leerer Ausgaben sind gültige Verlusturteile.
- Beobachtete Retention ist K/V, fester Nenner K/20. Adjustierung:
  `(K/20)/(1+D)` mit `D=(Unter-+Überschreitung des Wortfensters)/Zielwortzahl`.
- Artikel sind die Einheit gepaarter Vergleiche und der 10.000 Bootstrap-Ziehungen,
  Seed 32. Die 95-%-Perzentilintervalle sind nicht für Mehrfachvergleiche korrigiert.
- Regression: mindestens 20 Beobachtungen je Verfahren im gemeinsamen 5.–95.-Perzentilbereich
  und 20 gepaarte, lokal gestützte Artikel. Lokal mindestens fünf Ausgaben **jedes**
  Verfahrens im Band `max(5 Wörter, 10 % der betrachteten Wortzahl)`.

Weitere notwendige Quellen:

- `human_qa/` und `menschliche_relevanz_quellen.zip`: pseudonymisierte Rückgaben,
  Fallmetadaten, Auswahlregel und Bewertungsunterlagen. Kalibrierung auf denselben
  Fällen ist keine unabhängige Validierung; die finalen 24-Kandidaten-Fakten wurden
  nicht nochmals menschlich validiert. Entitätstypen sind automatisch zugewiesen.
- `demonstrationen.zip`: Metadaten der Auswahlartikel (240 Training, 30 Validierung,
  100 Test; zusätzlich der ursprüngliche 200er-Trainingsstand als Herkunftsbeleg),
  Kandidaten, Auswahl-/Pruningurteile und 5.753 verwendete Optimierungscaches.
  `02_demonstrations.py` reproduziert daraus 600 Update-Verluste und vier finale
  Validierungsverluste, ohne neue Generierungen oder Optimierung.
- `modellpilot.zip`: Metadaten zu fünf Artikeln, Modellantworten, Versuche und Parameter des
  Neunmodelle-Vorexperiments zu Tabelle 14. Dies ist eine andere Untersuchung als
  der entfernte zusätzliche LLM-Scorervergleich.
- `methodenmaterial.zip`: ursprüngliche Methodenbilder 1–4, Bewertungsmasken 17/18
  und Manuskripttabellen 3/4 und 9–14. Die Masken zeigen Originaltext als Bild;
  ihre Weitergabe bleibt ungeklärt. Keine Analysedatenquelle.
- `methoden/`: ursprüngliche Parameter und Belege der späteren Ersetzung;
  `prompts/`: zwölf tatsächlich eingesetzte Demonstrationsprompts und deren Zuordnung.

## Gespeicherte Modellbewertungen und Bootstrap-Anpassungen

Die Notebooks werten gespeicherte BERTScore- und SummaC-Bewertungen aus.
Ihre ursprüngliche Modellinferenz wird dadurch nicht erneut ausgeführt.
Notebook 06 verwendet standardmäßig den mitgelieferten Regressionscache.
Mit `RECOMPUTE_BOOTSTRAP=True` werden alle 10.000 Anpassungen neu geschätzt
und der neue Cache unter `results/` gespeichert.

Die Auto-CoT-Clusterprojektion in Notebook 03 stellt die historisch
gespeicherten 2D-Positionen neu dar. Die ursprünglichen Embeddings und
Cluster werden dabei nicht neu berechnet.

## Zuordnung zum Manuskript

Das lokal geprüfte PDF führt 23 Abbildungen. Für Abbildungen 1, 2, 4, 17 und
18 liegen historische Bilder in `data/methodenmaterial.zip`; sie sind keine
von den Auswertungsnotebooks erzeugten Ergebnisdateien. Abbildung 3 ist dort
ebenfalls historisch enthalten und wird aus der gespeicherten Projektion in
Notebook 03 neu dargestellt. Die Archivbilder 17/18 enthalten Quelltext als
Bild; siehe den Rechtehinweis in der README.

| Abbildung | Zugehörige Datei beziehungsweise Archivmitglied |
|---|---|
| 1 | `data/methodenmaterial.zip`: `manuskript_image4.png` |
| 2 | `data/methodenmaterial.zip`: `manuskript_image5.png` |
| 3 | `data/methodenmaterial.zip`: `manuskript_image6.png`; [neu dargestellte Clusterprojektion](../results/03_textlaenge/autocot_cluster_projection.png), [Projektionsquelle](../reference/figure_01_autocot_cluster_projection.svg) |
| 4 | `data/methodenmaterial.zip`: `manuskript_image7.png` |
| 5–7 | [Relevanzkonfusion](../results/01_menschliche_validierung/abb_05_relevanzkonfusion.png), [Relevanzranking](../results/01_menschliche_validierung/abb_06_relevanzranking.png), [QA-Kalibrierung](../results/01_menschliche_validierung/abb_07_qa_kalibrierung.png) |
| 8–10 | [Retentionsverlauf](../results/02_faktenretention/abb_08_retentionsverlauf.png), [Artikelverteilungen](../results/02_faktenretention/abb_09_artikelverteilungen.png), [gepaarte Kontraste](../results/02_faktenretention/abb_10_gepaarte_kontraste.png) |
| 11–14 | [Wortfenster](../results/03_textlaenge/abb_11_wortfenster.png), [Wortlängen](../results/03_textlaenge/abb_12_wortlaengen.png), [Längenadjustierung](../results/03_textlaenge/abb_13_laengenadjustierung.png), [Matching 300/75](../results/03_textlaenge/abb_14_matching_300_75.png) |
| 15–16 | [Erste Nicht-Retained-Stufe](../results/04_verlustmuster/abb_15_erste_nicht_retained_stufe.png), [bedingte Entitätsverluste](../results/04_verlustmuster/abb_16_bedingte_entitaetsverluste.png) |
| 17–18 | `data/methodenmaterial.zip`: `manuskript_image20.png` und `manuskript_image21.png` |
| 19–20 | [Relevanzurteile](../results/01_menschliche_validierung/abb_19_relevanzurteile.png), [Relevanzkappa](../results/01_menschliche_validierung/abb_20_relevanzkappa.png) |
| 21–23 | [Matching 500](../results/03_textlaenge/abb_21_matching_500.png), [Matching 150](../results/03_textlaenge/abb_22_matching_150.png), [Entitätsretention](../results/04_verlustmuster/abb_23_entitaetsretention.png) |

Die Zuordnung bezeichnet die jeweilige thematische Quelle; sie ist kein neuer
vollständiger Bild- oder Pixelabgleich mit dem PDF. Der ältere dokumentierte
Abgleich von Abbildungen 5–16 und 19–23 ist in
[abgleich_manuskript.md](abgleich_manuskript.md) erhalten.

| Notebook | Abbildungen und Tabellen |
|---|---|
| 01 – Menschliche Validierung | Abb. 5–7, 19/20; Tab. 5/6 |
| 02 – Faktenretention | Abb. 8–10; Tab. 15/16/20 |
| 03 – Textlänge | Abb. 11–14, 21/22; Tab. 17; Methodenabbildung Auto-CoT |
| 04 – Verlustmuster | Abb. 15/16/23; Tab. 7, 21–25 |
| 05 – Weitere Metriken | Tab. 8; Kendall-τ-b-Export `anhang_E_1_kendall_tau_b.csv` ohne gleichnamige Tabelle im aktuellen Manuskript; dessen E.1 enthält Tab. 21 |
| 06 – Längenregression | Tab. 18/19; Auto-CoT–Plain bei 75 Wörtern |

## Grenzen und Hinweise zum Manuskript

Die Tests prüfen positionsgenaue QA-Labels, Nenner,
Artikelbootstrap-Referenzen und die Experimentphasen ohne echte Modellaufrufe.
Historisch fehlen vollständige Cluster-Embeddings und ein vollständiges Archiv der QA-Retry-Zwischenantworten;
diese werden nicht nachträglich erfunden. Gespeicherte BERTScore-/SummaC-Werte werden
statistisch ausgewertet; ihre ursprüngliche Inferenz wird dadurch nicht neu validiert.

Der [PDF-Stichprobenabgleich](abgleich_manuskript.md#ergänzung-später-vorliegendes-pdf)
ergänzt den früheren DOCX-Befund:

- Anhang E.1 existiert und enthält Tabelle 21 zur Entitätsverteilung. Eine
  „Tabelle E.1“ wird im aktuellen Manuskript nicht referenziert. Der historisch
  benannte Kendall-Export aus Notebook 05 ist davon zu unterscheiden.
- Das spätere PDF nennt für Auto-CoT–Few-Shot-CoT bei **300 Wörtern** in den
  Tabellen 16 und 18 die korrigierte Intervalluntergrenze **4,98 pp**; die
  Repo-Nachrechnung ergibt 4,9803289473684185 pp. **4,89** gehört zur
  500-Wort-Stufe. Für die Supportartikel bei **150 Wörtern** nennt Tabelle 18
  nun **3,97 pp**; der Repo-Wert beträgt 3,969861 pp.
- Notebook 02 beschreibt jetzt ausschließlich Änderungen zwischen gemessenen
  Kompressionsstufen. Eine Original-QA-Baseline liegt nicht vor. Entsprechende
  Aussagen über den größten Rückgang vom Original stehen noch in Kapitel 4/5.
- Das spätere PDF nennt für die führenden generativen Verfahren Automate-CoT
  und Zero-Shot-CoT bei 500 Wörtern nun den adjustierten Abstand **5,64 pp**
  [3,12; 8,16]. Die zuvor beanstandeten **0,08 pp** gehörten zu einem anderen
  Vergleich.
- Die Lead-k-Angaben **21/1.407** und **22/1.999** haben unterschiedliche,
  in Notebook 04 reproduzierte Bezugsgruppen. Das Manuskript beschreibt beide
  Gruppen an verschiedenen Stellen; ein expliziter Querverweis wäre hilfreich.

Fundstellen, weitere Abweichungen und Änderungsvorschläge stehen in der
[Korrekturliste](korrekturliste_manuskript.md). Wissenschaftliche Rechenregeln,
Rohdaten und gespeicherte Caches wurden bei diesem Abgleich nicht geändert.
