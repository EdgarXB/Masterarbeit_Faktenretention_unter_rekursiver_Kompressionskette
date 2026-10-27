# Notebook-Einstieg und Wartung vereinfacht

Stand: 27. Oktober 2026. Zusätzliche Überarbeitung auf
`codex/reproducible-notebooks`, Ausgangscommit
`7cf76f6dd0a16952c016da038ad69597737acf18`. Kein Push, Merge oder Release.

## Bedienung

Nach der Installation genügt einmal `python 05_analyze.py --prepare`. Danach ein
fachliches Notebook mit dem Analysekernel öffnen und **Run All** ausführen.
`DATA_DIR`, `OUTPUT_DIR` und `TABLES` stehen unmittelbar unter dem Titel. Standard:
vorbereitete Eingaben unter `data/prepared`, neue Tabellen und Grafiken unter
`results/notebooks/tables` und `results/notebooks/figures`.

Entfallen sind die manuell gesetzte `FAKTENRETENTION_WORK`, `workspace.json`, ein
vorausgehender Analyse-Runner-Lauf und das manuelle Ausführen von Vorgängern für
die historische Einzelauswertung. Eigene absolute Eingabe- und Ausgabeordner sind
in derselben Zelle einstellbar; die einmalige Vorbereitung unterstützt `--data-dir`.
Wer eigene Vorgängertabellen verwenden möchte, kann `TABLES` auf deren `tables`
setzen. Der automatisierte Runner berechnet seine Vorgänger stets selbst.

## Aktiver Code und Herkunft

- Notebook 11 benötigt jetzt 03, 04 **und 07**; Notebook 06 benötigt 03, 04 **und 05**.
  Lead-k hat keine Analysevorgänger. Beim Tabellenexport entfallen die unnötigen
  Vorgänger 01 und 02. Die übrigen Dateizugriffe wurden gegen die Abhängigkeiten geprüft.
- `notebook_context.py` enthält nur Pfadprüfung, Offline-Sperre und Inlinegrafiken;
  die Kontextklasse, Laufzugehörigkeitsdatei und Umgebungsvariable sind entfernt.
- `analyses.json` nennt weiterhin jede erwartete Datei statisch. Wiederholte
  Verzeichnis-, Notebook- und Referenzpfade werden durch gemeinsame Konventionen
  ersetzt: 597 statt 1.849 Zeilen. Erwartete Ausgaben stammen nicht aus einem
  Verzeichnis-Scan der tatsächlich entstandenen Dateien.
- Historische Pfade werden nur in `frozen_data.prepared_path` übersetzt. Die aktive
  Struktur unterscheidet Eingaben, Vorgängertabellen, Caches und neue Tabellen/Grafiken.
  Der Runner entpackt keine zweite Referenzkopie und keine tausenden unbenötigten
  historischen Experimentdateien. Der zusätzliche vollständige Altbaum-Export
  `01_prepare.py --working-copy` entfällt ebenfalls. Abgeleitete Vorgängermanifeste erhalten übersetzte
  Herkunftspfade; die Originalmanifeste und Originalbytes bleiben in den Archiven.
- Methoden, Stufen, Anzeigenamen, Kontraste und Wortfenster verwenden `protocol.py`.
  Die verbliebenen identischen Wortzähler verwenden `text_rules.count_words`;
  unterschiedliche statistische Schätzer bleiben unverändert im jeweiligen Notebook.
  CSV-/JSONL-Zugriffe und identische Round-Trip-Tabellenleser sind in `io.py` gebündelt.
- Die 44 Dateien der Baseline wurden vollständig bytegeprüft in
  `archive/baseline_ad7eb1a.zip` zusammengefasst. Die Mitgliedsprüfsummen sind separat
  verzeichnet. Tests lesen gezielte Quellen aus dem ZIP, ohne Altcode zu importieren.
- Eine kompakte `data/code_provenance.json` ersetzt die doppelte Pfadliste und rund
  2,4 MB wiederholte Patchkopien. Originalherkunft und Originalprüfsummen bleiben
  nachvollziehbar; ursprüngliche Portabilitätsdiffs liegen im Baseline-ZIP und die
  Refaktorierungsdiffs in der unveränderten Git-Historie.
- Die vollständige Regression-Bootstrap-Neuberechnung ist aus Notebook 11 entfernt.
  Nur vorhandene fingerprintgeprüfte Caches werden gelesen. Bei fehlendem oder
  unpassendem Cache stoppt das Notebook mit einer konkreten Anleitung.

## Tatsächlich ausgeführte Prüfungen

Die vollständige Testsuite besteht mit **32 Tests**. Dazu gehören neue Prüfungen
für die Dateizugriffe aller Notebooks gegen die Abhängigkeiten, die Trennung von
Eingaben und neuen Ausgaben, den direkten Vorbereitungshinweis und die vollständige
Baseline-Sicherung. Scorer-, Text-, Prompt- und Statistikregeln werden weiterhin
gegen bytegleiche historische Quellen beziehungsweise deren AST geprüft.

| Prüfweg | Ergebnis | Beleg |
|---|---|---|
| Vollständiger Runner-Lauf, frische Kernel und neu berechnete Vorgänger | 12/12 erfolgreich, 290 Tabellen-/Inventarvergleiche, 65 pixelgleiche PNGs | [Runner](validation/simplification-runner.json) |
| Alle zwölf Notebooks direkt, Standardparameter unverändert | 12/12 erfolgreich, dieselben 290 Vergleiche und 65 pixelgleichen PNGs | [Direkt](validation/simplification-direct.json) |
| Frischer Einzellauf `--only 11_statistische_laengenkontrolle` | 03, 04, 07, 11 erfolgreich | [Einzellauf 11](validation/simplification-single11.json) |
| Frischer Einzellauf `--only 06_weitere_metriken` | 03, 04, 05, 06 erfolgreich | [Einzellauf 06](validation/simplification-single06.json) |
| Frischer Einzellauf `--only 08_tabellenexport` | 03, 04, 05, 06, 08 erfolgreich; ohne 01 und 02 | [Einzellauf 08](validation/simplification-single08.json) |
| Frischer Einzellauf `--only 09_lead_k_kontrolle` | ausschließlich 09 erfolgreich | [Einzellauf 09](validation/simplification-single09.json) |
| Notebook 11 direkt mit getrennten eigenen Daten-/Ausgabeordnern und Arbeitsverzeichnis `/tmp` | erfolgreich, Tabellen und PNGs referenzgleich | [Eigene Pfade](validation/simplification-custom-paths.json) |
| Bestehende und erweiterte Testsuite | 32/32 erfolgreich | [Testprotokoll](validation/simplification-tests.log) |

Der direkte Weg wurde mit gewöhnlichem `nbclient.NotebookClient` ausgeführt,
**ohne** Runner-Ausführung und **ohne** `FAKTENRETENTION_*`-Variablen. Jeder Kernel
startete im Notebookverzeichnis; die Parameterzellen blieben unverändert. Beim
zusätzlichen Pfadtest wurden ausschließlich `DATA_DIR` und `OUTPUT_DIR` geändert.
Für die Ergebnisprüfung wurden dieselben unveränderten Referenz-ZIPs verwendet.
Die gespeicherten aktiven Notebooks enthalten die tatsächlich direkt ausgeführten
Tabellen und eingebetteten Grafiken.

Zusätzlich wurden alle bisherigen statischen Referenzzuordnungen erhalten und die
Vollständigkeit von **105 PDF-/SVG-Exporten** geprüft. Das sind die bisherigen 101
Exporte plus vier bereits erzeugte Exporte der längengematchten Einzelmatrizen,
die zuvor nicht im Inventar standen. Es wurden keine Prüfungen entfernt und keine
Toleranzen erweitert: Tabellen bleiben bei `rtol=0, atol=1e-12`, PNGs pixelgenau.
Das Exportinventar von Notebook 06 führt den ausdrücklich wiederverwendeten
ROUGE-Eingang weiterhin mit, obwohl er nun getrennt im Eingabeordner liegt.

`python 01_prepare.py` bestätigte alle **6.288 Quellen** und die Bestandszahlen;
`python -m faktenretention.frozen_data --checksums` bestätigte alle **47**
unveränderten Originalprüfsummen. Alle **44 Baseline-Mitglieder** wurden zusätzlich
gegen ihre separaten Prüfsummen geprüft. Das Paketmanifest wurde an die aktuelle
Struktur angepasst; ursprüngliche Messdaten, Prompts und Referenzarchive wurden
nicht geändert.

Der Notebook-Einstieg wurde als HTML exportiert und im vorhandenen Headless-Chromium
visuell geprüft. Außerdem wurde die Grafik zur Retention über tatsächlicher Wortlänge
betrachtet. Für alle 65 PNGs liegt zusätzlich der vollständige Pixelvergleich vor.
Es wurde kein manueller Klicktest in einer VS-Code- oder JupyterLab-Oberfläche ausgeführt.

## Verbleibende Grenzen

Die dokumentierte Linux-Analyseumgebung und die passenden Paketversionen bleiben
für die Regressionscache-Fingerprints erforderlich. Ein Notebook-Frontend ist
zusätzlich zum Kernel nötig. Im direkten Standardweg sind Vorgängertabellen
historische Eingaben; die vollständige Neuberechnung einschließlich Vorgängerprüfung
liefert der Runner. Die ROUGE-/BERTScore-/SummaC-Messwerte und LLM-Urteile bleiben
historisch eingefroren. Es gab keine echten LLM-Aufrufe, Gewichtdownloads, neuen
Optimierungen oder vollständigen Regression-Bootstrap-Neuberechnungen.
