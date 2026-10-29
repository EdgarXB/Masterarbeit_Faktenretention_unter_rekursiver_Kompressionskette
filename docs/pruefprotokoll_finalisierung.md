# Prüfprotokoll der Abgabe-Finalisierung

Prüfdatum: **29.10.2026**. Ausgangscommit:
`78bcdea` (`Dokumentiere Experimentabläufe und genaue Modellbeschaffung`).
Arbeitsbranch: **`abgabe-finalisierung`**. Keine Modellläufe, kein Push,
kein Tag und kein Release. Das vom Autor bereitgestellte Manuskript
`2910_Masterarbeit.docx` wurde nicht verändert oder versioniert.

## Umfang und Dateiliste

- `README.md`: Reproduktionsgrenzen, Voraussetzungen, Quickstart, CI-Umfang,
  Herkunft/Lizenzangaben, KI-Hinweis, Zitierung, vorgeschlagene Releasebefehle.
- `docs/auswertung_und_einordnung.md`: bestätigter Manuskriptstand und anhand
  des aktuellen Abgleichs berichtigte Hinweise.
- `docs/modelle_und_beschaffung.md`, `requirements-summac.txt`: ausdrücklich
  ungetestete dritte Installationsumgebung und tatsächlich beobachtete Grenzen.
- `docs/abgleich_manuskript.md`, `docs/belege_abgleich_tabellen.csv`:
  Zuordnung aller angeforderten Tabellen/Abbildungen und 797 Zahlenprüfungen.
- `docs/korrekturliste_manuskript.md`: Fundstellen, Ist-Text und Vorschläge.
- `docs/abgleich_code_parameter.md`: rein lesender Parameterreview.
- `notebooks/02_faktenretention.ipynb`, `notebooks/05_weitere_metriken.ipynb`:
  ausschließlich Markdown zur Original-QA-Baseline bzw. zur E.1-Zuordnung.
- `LICENSE`, `CITATION.cff`: vom Autor gewählte MIT-Lizenz für eigenen Code
  sowie Zitiermetadaten (Edgar Berngardt, Hochschule Düsseldorf, 2026).
- `pyproject.toml`: vom Autor freigegebene Mindestversion `>=3.13`.
- `.gitignore`: lokales Abgabemanuskript und typische lokale `.env`-Dateien.
- Dieses Prüfprotokoll und `docs/pruefbelege_finalisierung.json`.

Die angefragten Entscheidungen wurden beantwortet: Manuskript
`2910_Masterarbeit.docx`, Code-Lizenz MIT, Python-Mindestversion `>=3.13`.
Python 3.13.15 war bereits installiert; mit der Analyseumgebung wurden die
nachfolgenden Prüfungen ausgeführt. Die Mindestversion ist eine konservative
Festlegung auf Basis dieses geprüften Stands, kein Nachweis, dass jede höhere
Python-Version oder jede andere Plattform bereits geprüft wäre.

## Tests und Notebooks vor und nach der Änderung

| Prüfung | Vorher | Nachher |
|---|---|---|
| `.venv-analysis/bin/python -m unittest discover -s tests -v` | **24 Tests, OK** (2,203 s) | **24 Tests, OK** (2,077 s) |
| `01_menschliche_validierung.ipynb` | erfolgreich | erfolgreich |
| `02_faktenretention.ipynb` | erfolgreich | erfolgreich |
| `03_textlaenge.ipynb` | erfolgreich | erfolgreich |
| `04_verlustmuster.ipynb` | erfolgreich | erfolgreich |
| `05_weitere_metriken.ipynb` | erfolgreich | erfolgreich |
| `06_laengenregression.ipynb` | erfolgreich | erfolgreich |

Notebookaufruf je Datei (in frischem Kernel):

```bash
.venv-analysis/bin/python -m jupyter nbconvert --execute --to notebook --output-dir /tmp/abgabe-finalisierung/before --ExecutePreprocessor.timeout=900 notebooks/01_menschliche_validierung.ipynb
```

Dieselbe Form wurde für alle sechs Dateien, nach der Änderung mit dem
Ausgabeordner `/tmp/abgabe-finalisierung/after`, ausgeführt. Die Tabellen und
Grafiken wurden dabei wie vorgesehen nach `results/` geschrieben. Die ausgeführten
Notebookkopien liegen unter `/tmp`; die versionierten Notebooks behalten ihre
ursprünglichen Codezellen und gespeicherten Outputs. Im JSON-Diff der beiden
bearbeiteten Notebooks ändern sich nur die genannten Markdown-Zellen.

Umgebungsschalter: `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. Der erste Versuch innerhalb der
Sandbox scheiterte an deren Socket-Sperre beim Kernelstart. Mit freigegebenen
lokalen Jupyter-Sockets liefen alle Notebooks vor und nach den Änderungen.
Dies war eine Ausführungsbeschränkung der Sandbox, kein Fehler einer Notebookzelle.

Zusätzlich nachher erfolgreich: `05_analyze.py --prepare` (36 Quellen),
`01_prepare.py` (36 Quellen), `02_demonstrations.py` (gespeicherte Auswahl/
Optimierungsprotokolle, **0 LLM-Aufrufe**), `git diff --check`,
Notebookformatvalidierung und Vergleich aller Codezellen/gespeicherten Outputs
gegen die gesicherten Originalnotebooks. TOML wurde geparst; `CITATION.cff`
wurde als YAML geparst und die Kernfelder geprüft. Eine vollständige Prüfung
gegen das externe CFF-JSON-Schema wurde nicht ausgeführt.

Die eingebetteten Manuskriptbilder der angeforderten Abbildungen wurden
betrachtet. Der quantitative Bildabgleich bleibt bei Kurven/Boxplots ein
Stichprobenabgleich, siehe Manuskriptabgleich. Die Notebookgrafiken wurden
inhaltlich nicht bearbeitet; ihre resultierenden PNG-Dateien stimmen bytegenau
mit dem Ausgangsstand überein. Eine neue gerenderte HTML-Ansicht der zwei
geänderten Markdown-Passagen wurde nicht visuell geprüft.

## Vergleich der Ergebnisse und geschützten Daten

Vor der ersten Prüfung wurden Dateilisten und SHA-256 der vorhandenen Dateien
unter `data/`, `results/` und `notebooks/` gesichert. Nach der letzten Ausführung
wurde jede vorhandene Ergebnisdatei erneut geprüft.

- **377 vorhandene Dateien unter `data/` unverändert**, einschließlich lokaler
  vorbereiteter Eingaben, Bootstrap-Caches und aller Modellmanifeste/Hashes.
- **47.142 vorhandene Dateien unter `results/` geprüft**, einschließlich
  ignorierter lokaler Arbeitsbestände; keine neue Datei gegenüber dem Snapshot.
  Davon sind **74 Dateien versioniert, darunter 56 CSVs**.
- Nach beiden Notebookserien wichen ausschließlich zwei Laufzeitfelder vom
  ursprünglichen Dateistand ab:
  `C_modelldiagnostik.csv:fit_seconds` und
  `bootstrap_anpassungsdiagnostik.csv:seconds_this_run` in Notebook 06.
- Alle anderen Spalten dieser Diagnose-CSVs wurden exakt verglichen und
  blieben identisch. Dies sind gemessene Rechenzeiten, keine wissenschaftlichen
  Ergebnisse. Die nachher gemessenen Zeiten stehen in
  [pruefbelege_finalisierung.json](pruefbelege_finalisierung.json).
- Nach dem Vergleich wurden nur diese zwei von der Ausführung überschriebenen
  Diagnose-CSVs aus dem zuvor sauberen Ausgangscommit wiederhergestellt.
  **Der finale Inhalt aller `results/`-Dateien entspricht bytegenau dem Snapshot.**
  Es wurde kein Ergebnis neu festgelegt und kein Cache überschrieben.
- Auch die unveränderte Manuskriptdatei wurde abschließend gegen ihren zu Beginn
  dokumentierten SHA-256 geprüft.

Der Zahlenabgleich mit dem Manuskript fand **3 abweichende Datenzellen unter
797 Einzelprüfungen** (Tab. 16 einmal, Tab. 18 zweimal); daraus wird kein
Gesamtgenauigkeitsmaß der Arbeit abgeleitet. Weitere begriffliche Unterschiede
und Dokumentationslücken stehen separat in der Korrekturliste.

## SummaC-Probe ohne neue Inferenz

Eine frische Testumgebung unter `/tmp` wurde angelegt. Der Installationsversuch
mit `uv pip sync --offline` und dem bestehenden Paketcache scheiterte am fehlenden
Wheel `nvidia-nvjitlink==13.4.92`. Deshalb wurden die weiteren Installationsschritte
und die Inferenz in dieser Umgebung nicht ausgeführt. `uv pip check` in der
bestehenden historischen Metrikumgebung bestätigt den Konflikt zwischen SummaC
0.0.4 und Hugging-Face-Hub 1.28.0. Paketmetadaten wurden nur gelesen.
Der neue Zusatz `requirements-summac.txt` ist ausdrücklich **ungetestet**.
Die bereits vorhandene CPU-Provenienz wurde nicht als neue Prüfung ausgegeben.

## Repo-Hygiene und Secrets

Geprüft wurde der anfängliche versionierte Bestand von **198 Dateien**, dabei
auch **5.805 Textmitglieder** in den ZIP-Archiven sowie entpackte Inhalte der
Gzip-Quellen. Binäre Bilder/PDFs/Modellcache wurden nicht als Text durchsucht.
Prüfmuster: typische OpenAI-/HF-/GitHub-/AWS-Schlüssel, Private-Key-Kopfzeilen,
lange wörtliche Zuweisungen an API-Key/Token/Password/Secret sowie absolute
Home-/Windows-Benutzerpfade und der lokale Benutzername. Das ist eine
Mustersuche, kein vollständiger Beweis der Abwesenheit beliebiger Geheimnisse.

**Keine Schlüssel-/Private-Key-/Credential-Zuweisung nach diesen Mustern gefunden.**
Absolute Benutzerpfade wurden in folgenden elf Dateien/Archivmitgliedern
gefunden; acht davon enthalten den lokalen Benutzernamen:

| Datei / Archivmitglied | Befund |
|---|---|
| `data/model_metadata/provenance/sources.json` | 9 Originalpfade |
| `data/model_metadata/provenance/environment_20260824.json` | 4 Home-Pfade |
| `data/modellpilot.zip::manifest.json` | 1 Home-Pfad |
| `data/demonstrationen.zip::auto_cot_selection/auto_cot_selection_report.json` | 1 Home-Pfad |
| `data/demonstrationen.zip::automate_cot_pruning/automate_cot_pruned_pools.json` | 1 Home-Pfad |
| `data/demonstrationen.zip::automate_cot_pruning/automate_cot_pruning_audit.jsonl` | 1 Home-Pfad |
| `data/demonstrationen.zip::candidate_rollouts/manifest.json` | 1 Home-Pfad |
| `data/demonstrationen.zip::candidate_rollouts/stage_candidates.jsonl` | 1 Home-Pfad |
| `data/demonstrationen.zip::data/split_manifest.json` | 5 Home-Pfade |
| `data/demonstrationen.zip::data/split_manifest_v14_200.json` | 5 Home-Pfade |
| `data/demonstrationen.zip::validation_rollouts/manifest.json` | 1 Home-Pfad |

Diese Originalbelege bleiben gemäß Aufgabenregel unverändert. Autorennennung
und Pseudonyme der Bewertenden sind von geheimen Zugangsdaten zu unterscheiden.
Die Prüfung ersetzt keinen Nachweis einer Veröffentlichungseinwilligung.

Größte versionierte Dateien: `data/demonstrationen.zip` **16.253.766 Bytes**,
`data/bootstrap_linear.npz` **7.317.202 Bytes**,
`data/generierungszustaende.jsonl.gz` **2.040.307 Bytes**,
`notebooks/03_textlaenge.ipynb` **1.917.044 Bytes**. Keine versionierte Datei
überschritt 100 MiB. Die lokale Größe von `results/` (etwa 1,8 GiB) stammt auch
aus ignorierten Arbeitsbeständen; sie wurden weder hinzugefügt noch entfernt.

`.gitignore` deckt virtuelle Umgebungen, `data/prepared/`, lokale Ergebnisläufe
und Modellressourcen ab. Ergänzt wurden nur das konkrete lokale Manuskript und
lokale `.env`-Dateien. Ignorierte Bestände werden nicht dadurch aus der
Git-Historie entfernt; es war keine Historienbereinigung Gegenstand der Aufgabe.

## Offen / vom Autor zu klären

| Punkt | Optionen |
|---|---|
| Manuskriptkorrekturen | Vorschläge übernehmen oder eine abweichende Entscheidung mit Begründung dokumentieren; danach finalen PDF-Stand prüfen. |
| Menschliche Bewertungen | Weitergabe-Einwilligung und Lizenz belegen oder vor öffentlicher Veröffentlichung den zulässigen Umfang festlegen; keine Daten wurden entfernt. |
| CNN/DailyMail 3.0.0 | Konkrete Weitergabebedingungen belegen; die Datensatzkarte nennt in ihrem Lizenzabschnitt ausdrücklich 1.0.0. |
| Historische Benutzerpfade | Originalbelege beibehalten oder separat einen nachvollziehbaren bereinigten Veröffentlichungsbestand planen; Originale und Hashes hier unverändert. |
| SummaC | Alle Wheels für eine spätere frische Offline-Installation bereitstellen und diese gesondert validieren, oder die dokumentierte Inferenzgrenze beibehalten. |
| Abgabeversion | Nach finaler Freigabe Tag/Release `v1.0-abgabe` selbst ausführen oder zunächst nur den finalen Commit-Hash in Anhang A.1/F nennen. Befehle stehen in der README. |
