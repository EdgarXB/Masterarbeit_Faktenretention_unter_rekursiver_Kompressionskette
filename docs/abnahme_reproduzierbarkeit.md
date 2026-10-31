# Praktische Reproduzierbarkeitsabnahme

Prüfstand: 31. Oktober 2026. Getestet wurde eine isolierte Kopie des Analyse-
und Datenstands einschließlich der noch nicht committeten Änderungen. Diese
Dokumentation und ihr README-Verweis wurden nach dem Lauf ergänzt. Der
ursprüngliche Arbeitsbaum und seine Notebookausgaben wurden nicht überschrieben.

## Umgebung und Prüfungen

Für den Test wurde eine neue virtuelle Umgebung mit Python 3.13.15 erstellt,
`requirements.txt` mit `uv pip sync` installiert und das Projekt mit
`uv pip install --no-deps -e .` eingebunden. Die 76 Abhängigkeiten wurden
offline aus einem bereits vorhandenen lokalen Paketcache installiert; auch die
Python-Installation war lokal vorhanden. Eine Erstinstallation aus dem Internet
wurde damit nicht geprüft. Die virtuelle Umgebung hatte keinen Zugriff auf die
Pakete einer bestehenden virtuellen Umgebung.

| Prüfung | Ergebnis |
|---|---|
| `05_analyze.py --prepare` | Erfolgreich aus den mitgelieferten Quellen. |
| `python -m unittest discover -s tests -v` | 30 Tests bestanden. |
| `scripts/validate_wissenschaftliche_integritaet.py` | Erfolgreich; Details im [Integritätsprotokoll](pruefung_wissenschaftliche_datenintegritaet.md). Der optionale Vergleich mit lokal ignorierten historischen Daten war in der isolierten Kopie nicht verfügbar. |
| `01_prepare.py` | 36 Quellen anhand ihrer Prüfsummen geprüft. |
| `02_demonstrations.py` | Historisches Auswahl- und Cacheaudit erfolgreich. |
| Sechs Notebooks | Jedes Notebook erfolgreich mit eigenem Jupyter-Kernel durch `nbconvert --execute` ausgeführt; ausgeführte Notebookkopien lagen außerhalb des Repository-Arbeitsbaums. |
| Vergleich der Ergebnistabellen | 56 von 56 neu geschriebenen Referenztabellen stimmen in Schema, Zeilenfolge und fachlichen Werten überein; numerische Toleranz `1e-10`. |
| Git-Arbeitsbaum | `git diff --check` ohne Befund. Der Status enthält die bereits bestehenden und für diese Prüfung erstellten, noch nicht committeten Änderungen; durch den isolierten Notebooklauf wurden keine Ergebnisdateien im ursprünglichen Arbeitsbaum verändert. |

Die Sandbox blockierte zunächst das Erstellen lokaler Sockets für Jupyter;
der Notebooklauf wurde anschließend in der isolierten Kopie mit Zugriff auf
die für die Kernelkommunikation erforderlichen lokalen Sockets ausgeführt.
Modell- und Datenzugriffe blieben für den Lauf im Offline-Modus.

## Automatischer Ergebnisvergleich

`scripts/compare_analysis_results.py` vergleicht alle CSV-Dateien eines
Referenz-Ergebnisverzeichnisses mit gleichnamigen neu erzeugten Tabellen.
Fehlende Tabellen, veränderte fachliche Werte, Spalten oder Zeilen führen zu
einem Fehler. Nur `fit_seconds` in `C_modelldiagnostik.csv` und
`seconds_this_run` in `bootstrap_anpassungsdiagnostik.csv` sind ausgenommen,
weil sie die Dauer des jeweiligen Laufs messen. Die Ausnahme betrifft keine
anderen Spalten. Der Vergleich prüft Tabellenwerte, nicht die Pixel der PNGs.

Der Offline-CI-Workflow sichert die Referenzergebnisse vor dem Notebooklauf,
entfernt in seinem temporären Checkout die alten Tabellen, führt die sechs
Notebooks getrennt aus und vergleicht anschließend die neu geschriebenen
Tabellen. Zwei Unit-Tests prüfen, dass fehlende Tabellen und fachliche
Änderungen erkannt werden und die definierte Laufzeitausnahme funktioniert.
Der geänderte Workflow wurde lokal nachvollzogen; ein neuer GitHub-Actions-Lauf
für den noch nicht committeten Arbeitsstand liegt nicht vor.
