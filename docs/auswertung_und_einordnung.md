# Auswertungsgrundlagen und Hinweise

Diese Ergänzung zur [README](../README.md) beschreibt den Datenstand,
die wissenschaftlichen Definitionen und Hinweise zum Manuskript.
Grundlage ist das vom Autor benannte Abgabemanuskript
`2910_Masterarbeit.docx` (Stand 29.10.2026). Es wurde lokal für den
[Manuskriptabgleich](abgleich_manuskript.md) eingesehen, wird aber nicht
mitversioniert. Der Abgleich nennt den Prüfstand und die Grenzen der Prüfung.

## Datenstand und Kennzahlen

`data/quellen.json` ordnet Quellen, SHA-256, Datenstand und Parameterbelege zu.
Die fünf Hauptdateien enthalten Artikel, kanonische Fakten, Zusammenfassungen,
QA-Antworten und gespeicherte Modellmetriken. `generierungszustaende.jsonl.gz`
enthält die tatsächlichen rekursiven Eingaben und Generierungszustände.
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
  Originalfälle, Auswahlregel und Bewertungsunterlagen. Kalibrierung auf denselben
  Fällen ist keine unabhängige Validierung; die finalen 24-Kandidaten-Fakten wurden
  nicht nochmals menschlich validiert. Entitätstypen sind automatisch zugewiesen.
- `demonstrationen.zip`: tatsächliche Auswahlartikel (240 Training, 30 Validierung,
  100 Test; zusätzlich der ursprüngliche 200er-Trainingsstand als Herkunftsbeleg),
  Kandidaten, Auswahl-/Pruningurteile und 5.753 verwendete Optimierungscaches.
  `02_demonstrations.py` reproduziert daraus 600 Update-Verluste und vier finale
  Validierungsverluste, ohne neue Generierungen oder Optimierung.
- `modellpilot.zip`: fünf Artikel, Originalantworten, Versuche und Parameter des
  Neunmodelle-Vorexperiments zu Tabelle 14. Dies ist eine andere Untersuchung als
  der entfernte zusätzliche LLM-Scorervergleich.
- `methodenmaterial.zip`: ursprüngliche Methodenbilder 1–4, Bewertungsmasken 17/18
  und unveränderte Manuskripttabellen 3/4 und 9–14. Keine Analysedatenquelle.
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
Historisch fehlen vollständige Cluster-Embeddings und alle QA-Retry-Zwischenantworten;
diese werden nicht nachträglich erfunden. Gespeicherte BERTScore-/SummaC-Werte werden
statistisch ausgewertet; ihre ursprüngliche Inferenz wird dadurch nicht neu validiert.

Der aktuelle Abgleich ersetzt die Hinweise zum früheren Manuskriptstand:

- Anhang E.1 existiert und enthält Tabelle 21 zur Entitätsverteilung. Eine
  „Tabelle E.1“ wird im aktuellen Manuskript nicht referenziert. Der historisch
  benannte Kendall-Export aus Notebook 05 ist davon zu unterscheiden.
- Tabellen 16 und 18 nennen für die untere unadjustierte Intervallgrenze
  Auto-CoT–Few-Shot-CoT bei **300 Wörtern 4,89 pp**. Die aktuelle Repo-Auswertung,
  `tests/referenzwerte.csv` und eine unabhängige Nachrechnung ergeben jedoch
  **4,9803289473684185 pp**, gerundet **4,98**. **4,89** gehört zur **500-Wort-Stufe**.
- Tabelle 18 nennt für Auto-CoT–Few-Shot-CoT bei **150 Wörtern** auf den
  Supportartikeln unadjustiert **3,91 pp**; der Repo-Wert beträgt **3,969861 pp**,
  gerundet **3,97**.
- Notebook 02 beschreibt jetzt ausschließlich Änderungen zwischen gemessenen
  Kompressionsstufen. Eine Original-QA-Baseline liegt nicht vor. Entsprechende
  Aussagen über den größten Rückgang vom Original stehen noch in Kapitel 4/5.
- Der berichtete adjustierte 500-Wort-Abstand von **0,08 pp** gehört zu
  **Zero-Shot-CoT–Auto-CoT**. Die führenden generativen Verfahren sind dort
  Automate-CoT und Zero-Shot-CoT mit **5,64 pp** Abstand.
- Die Lead-k-Angaben **21/1.407** und **22/1.999** haben unterschiedliche,
  in Notebook 04 reproduzierte Bezugsgruppen. Das Manuskript beschreibt beide
  Gruppen an verschiedenen Stellen; ein expliziter Querverweis wäre hilfreich.

Fundstellen, weitere Abweichungen und Änderungsvorschläge stehen in der
[Korrekturliste](korrekturliste_manuskript.md). Wissenschaftliche Rechenregeln,
Rohdaten und gespeicherte Caches wurden bei diesem Abgleich nicht geändert.
