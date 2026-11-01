# Datenstand und Reproduzierbarkeit

Die Prüfung `python scripts/validate_wissenschaftliche_integritaet.py` verwendet
die mitgelieferten Daten und führt keine Modellinferenz aus.

## Daten und Auswahl

- **100 Artikel**, sechs Methoden und vier Stufen ergeben **2.400 Zusammenfassungen**;
  die **2.000 kanonischen Fakten** umfassen genau 20 Fakten pro Artikel.
- Die **48.000 QA-Positionen** enthalten 47.804 gültige Bewertungen und 196
  technische Fehlwerte. Technische Fehlwerte werden nicht als Faktenverlust
  klassifiziert; 40 `NOT_FOUND`-Bewertungen aus zwei leeren Ausgaben sind gültig.
- Die vier ursprünglich beibehaltenen Methoden umfassen 400 Artikel-Verfahren-Zeilen.
  Plain und Few-Shot-CoT wurden mit je 100 Zeilen neu erzeugt. Das gilt auch für
  die QA-Daten: 32.000 historische und je 8.000 Positionen der Reruns. Die
  [Ergebnis-](../data/methoden/result_merge_manifest.json),
  [QA-Merge-Manifeste](../data/methoden/qa_merge_manifest.json) und das
  [Protokoll](../data/methoden/protocol.json) dokumentieren die Zusammenführung.
- Die zwölf finalen Demonstrationsprompts stimmen mit ihren eingefrorenen
  Hashwerten überein. Auswahlbelege und Artikel-Hashes dokumentieren getrennte
  Trainings-, Validierungs- und Testdaten. Die erhaltenen Auswahlunterlagen
  zeigen keinen Rückgriff auf finale Testergebnisse; Vorgänge außerhalb dieser
  Artefakte lassen sich damit nicht ausschließen.
- Die 1.800 Übergänge zwischen Zusammenfassungsstufen stimmen mit den
  gespeicherten Eingaben überein. Die ursprünglichen Artikel sind in den
  Datentabellen entfernt; der erste Übergang kann damit nicht anhand des
  Originaltextes überprüft werden.

## Auswertung und Grenzen

Die beobachtete Retention ist K/V, mit K erhaltenen Fakten und V gültigen
Urteilen; K/20 verwendet den festen Nenner. Gepaarte Vergleiche und Bootstrap
verwenden den Artikel als Einheit. Die ursprüngliche QA-Baseline für den
Originalartikel fehlt. Die erste 500-Wort-Stufe zeigt daher Nichtnachweise,
keinen direkt gemessenen Rückgang gegenüber dem Original.

Duplikate werden anhand normalisierter Fakttexte erkannt. Die Längenregression
verwendet für die lokale Unterstützung das Band `max(5 Wörter, 10 %)` an beiden
beobachteten Wortzahlen. Die Lead-k-Angaben 21/1.407 und 22/1.999 beziehen sich
auf unterschiedliche Gruppen: erstere auf eindeutige erste Nicht-Retained-Stufen
vor 75 Wörtern, letztere auf Fakten mit mindestens einem gültigen Stufenpaar.

Der Abgleich des Manuskripts vom 29. Oktober 2026 ist in den
[numerischen Einzelbelegen](belege_abgleich_tabellen.csv) dokumentiert. Ein
späteres PDF wurde an ausgewählten Stellen nachgeprüft; die Abbildungen sind
unten ihren Quellen zugeordnet. Ein vollständiger erneuter Abgleich des PDFs
wurde nicht durchgeführt.

Die Notebooks werten gespeicherte BERTScore- und SummaC-Werte aus; sie führen
die ursprüngliche Inferenz nicht erneut aus. Notebook 06 nutzt standardmäßig
10.000 gespeicherte Bootstrap-Modellanpassungen. Ursprüngliche Auto-CoT-
Embeddings und QA-Retry-Zwischenantworten sind nicht vollständig archiviert.
Der ältere unveränderte historische Merge-Bestand liegt nicht im Anhang;
dessen Datei-Hashes lassen sich daraus nicht erneut byteweise überprüfen.

Die isolierte Ausführung der sechs Notebooks reproduzierte 56 Ergebnistabellen.
Im Vergleich ausgenommen sind zwei Laufzeitfelder; beim numerisch bestimmten
Kovarianzrang ist eine gemeldete Abweichung um eins zulässig. Sonstige Werte
werden mit `1e-10` verglichen. Ein neuer Drei-Artikel-Lauf durchlief die sechs
Verfahren und vier Stufen; BERTScore und SummaC wurden danach berechnet. Eine
vollständige neue Konstruktion aller Demonstrationen wurde nicht validiert.

## Manuskriptabbildungen und Ergebnisdateien

Abbildungen 1–4 sowie 17–18 sind historische Methodenbilder im Archiv. Abbildung
3 wird zusätzlich aus gespeicherten Projektionskoordinaten neu gezeichnet. Die
übrigen Dateien sind Ausgaben der Notebooks. Die Zuordnung bezieht sich auf die
thematische Quelle und ist kein vollständiger Pixelabgleich mit dem Manuskript.

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
