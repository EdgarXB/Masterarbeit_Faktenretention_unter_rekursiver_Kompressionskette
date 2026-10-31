# Bestandsprüfung des digitalen Anhangs

Prüfstand: 31. Oktober 2026. Der Arbeitsstand enthält auch die noch nicht
committeten Änderungen zur Entfernung von Originaltexten und Rechnerpfaden.
Diese Prüfung bewertet den aktuellen Arbeitsbaum, nicht die Git-Historie.
Nachtrag: Bei der späteren visuellen Zuordnung wurden Originaltext und ein
Originalauszug in den Maskenabbildungen `manuskript_image20.png` und
`manuskript_image21.png` innerhalb von `data/methodenmaterial.zip` erkannt.
Die damalige Integritäts- und Textprüfung hat Bildinhalte nicht erfasst.

## Entscheidungen zu Daten und Ergebnissen

| Bestand | Funktion und Entscheidung |
|---|---|
| `data/demonstrationen.zip` | Behalten. Die 5.772 Mitglieder umfassen Ausgangs- und Kandidatendaten, Auswahl- und Pruningbelege sowie 5.753 Cache-Einträge. `02_demonstrations.py` prüft daraus 600 Update-Verluste und vier finale Validierungsverluste. Die 5.753 Cache-Einträge sind keine frei austauschbaren temporären Dateien. |
| `data/methodenmaterial.zip` | Behalten. Sechs eingebettete Abbildungen und neun Tabellen dokumentieren Methodik und Bewertungsmasken; sie sind historische Manuskriptbelege, keine Eingabe der statistischen Notebooks. |
| `data/modellpilot.zip` | Behalten. Manifest, 18 Bedingungs- und 63 Stufenkennzahlen, Versuche und Ausgaben dokumentieren das Neunmodelle-Vorexperiment. Die Abschlusszahlen und gerundeten Plain-ROUGE-L-Werte stimmen mit Tabelle 14 des lokal vorliegenden PDF-Manuskripts überein; Kapitel 3.7 verweist auf dieses Experiment zur Auswahl des Summarizers. |
| `data/menschliche_relevanz_quellen.zip` und `data/human_qa/` | Behalten. Bewertungsunterlagen, Rückgaben, Auswahl- und Zuordnungsschlüssel belegen die menschliche Validierung. |
| `data/generierungszustaende.jsonl.gz` | Behalten. `faktenretention.data.length_values` prüft damit die rekursiven Ausgaben und Wortzahlen. Die entfernte erste Eingabe ist nur durch Metadaten belegt. |
| `data/artikel.csv.gz`, `fakten.csv.gz`, `zusammenfassungen.csv.gz`, `qa_antworten.csv.gz`, `modellmetriken.csv` | Behalten. Diese fünf gemeinsamen Eingaben werden durch `05_analyze.py --prepare` und die Auswertung verwendet. Artikel- und Highlight-Felder sind bereinigt; generierte Zusammenfassungen und Lead-k-Auszüge bleiben erhalten. |
| `data/bootstrap_linear.npz` und `data/bootstrap_linear.json` | Behalten. Notebook 06 verwendet den Cache standardmäßig; Manifest und `bootstrap_cache.py` prüfen Bindung und Fingerprint. |
| `data/methoden/`, `data/model_metadata/`, `data/models.json`, `data/MODEL_INVENTORY.json` und `data/resources.example.json` | Behalten. Sie dokumentieren Parameter, Modellzuordnung, Herkunft oder die Einrichtung zusätzlicher Modellressourcen. |
| `prompts/` | Unverändert behalten. Die zwölf Prompts und ihre Manifeste belegen die eingesetzten Promptbanken; die zwölf Hashwerte des Bankmanifests stimmen. Die 500-Wörter-Prompts enthalten weiterhin Originalartikel als Demonstrationen. |
| `results/` | Alle 74 versionierten Dateien behalten. 64 sind in Notebook-Code direkt benannt, acht weitere in Dokumentation und die zwei übrigen Diagnosedateien werden in Notebook 06 durch formatierte Dateinamen exportiert. Dateien dienen Ergebnis- oder Diagnosezwecken. |
| `reference/figure_01_autocot_cluster_projection.svg` | Behalten. Notebook 03 liest die Referenzdatei zur Darstellung der gespeicherten Projektion. |
| `notebooks/` | Alle sechs Notebooks und ihre gespeicherten Ausgaben behalten. Notebookausgaben wurden nicht pauschal geleert. |

## Integrität, Redundanz und Verweise

- Alle vier ZIP-Dateien bestanden die CRC-Integritätsprüfung, alle sechs
  versionierten Gzip-Dateien ließen sich vollständig lesen. Innerhalb der
  ZIPs wurden keine bytegleichen Mitglieder gleicher Größe und CRC gefunden.
- Unter den versionierten Dateien gibt es drei Paare mit identischem SHA-256:
  jeweils ein Ollama-Modellmanifest unter `data/model_metadata/manifests/` und
  der zugehörige Herkunftsbeleg unter `data/model_metadata/provenance/`. Die
  Kopien haben unterschiedliche Rollen und bleiben erhalten.
- Die 36 in `data/quellen.json` erfassten Quellen und die zwölf Promptdateien
  im Bankmanifest bestehen ihre Prüfsummenprüfung. Das Demonstrationsaudit
  läuft ohne Modellaufrufe. Die neun Tabellen und der Quellenvermerk im
  Methodenarchiv ergänzen die sechs Bilder als Manuskriptbelege; das Archiv
  wurde nicht als Recheneingabe vorausgesetzt.
- Markdown-Links in README, Dokumentation und Notebook-Markdown-Zellen wurden
  auf vorhandene lokale Ziele geprüft; es wurden keine fehlenden Ziele gefunden.
  Historische Quellenbezeichnungen und Prüfsummen in Archivmanifesten sind
  Herkunftsangaben und keine Zusage, dass unbereinigte Originaldateien
  mitgeliefert werden. Zwei bereinigte Demonstrationsmanifeste zeigen mit ihren
  relativen Eingabereferenzen auf vorhandene ZIP-Mitglieder.
- `.gitignore` schließt `data/prepared/`, `.venv-analysis/`,
  `results/runs/` und `results/notebooks/` aus. Lokal vorhandene ignorierte
  Analyse- und Prüfläufe sind keine Bestandteile des versionierten Anhangs.
  Darunter liegen größere Laufdiagnostiken und Notebookexporte; ihre
  individuelle Entbehrlichkeit wurde nicht für jede Datei nachgewiesen und
  sie wurden daher nicht gelöscht. Das nicht versionierte PDF-Manuskript
  wurde nur lesend zum Abgleich verwendet.

Kein versioniertes Archiv oder Ergebnis wurde bei dieser Bestandsprüfung
gelöscht. Das Fehlen eines direkten Codeverweises allein belegt weder
Redundanz noch fehlenden wissenschaftlichen Nutzen. Die vorhandenen
Originalartikel in unveränderten Prompts und frühere Git-Versionen bleiben
separat zu berücksichtigen.
