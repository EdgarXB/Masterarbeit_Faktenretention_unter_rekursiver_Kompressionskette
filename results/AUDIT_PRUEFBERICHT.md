# Geprüfte Auditkorrekturen — 23.10.2026

## Erfolgreich geprüft

- **15 Tests bestanden** in einer frisch aus `requirements-analysis.lock.txt` aufgebauten Umgebung. Alle 76 Pakete wurden offline aus vorhandenem Paketcache installiert; `uv pip check` bestätigt konsistente Abhängigkeiten. Tatsächliche Plattform: Linux x86_64, CPython 3.13.15. Exakte Bibliotheksversionen stehen in `data/analysis_environment.json`. Der Cache-Vorabtest akzeptiert diesen Stand und lehnt simulierte abweichende Python-, NumPy- und statsmodels-Versionen verständlich ab.
- **47 bestehende Dateien mit unveränderten SHA-256-Prüfsummen**, darunter alle direkt gelieferten eingefrorenen Messdaten/Prompts, Originalarchive und vier unveränderte wissenschaftliche Kernmodule. Auch die vorhandenen 6.288 logischen Originaldateien bestehen ihre Inventarprüfung. Die Prüfwerte wurden nicht an Git-Zeilenendenkonvertierungen angepasst.
- **Drei frische Git-Clones** mit `core.autocrlf=false`, `input` und `true`: sämtliche 47 Bestandsdateien bleiben byteidentisch, die vollständige Datenprüfung und der frühe Umgebungstest bestehen in jedem Clone. `.gitattributes` schützt eingefrorene Inhalte vor Konvertierung; Quellcode verwendet LF.
- **48.000 QA-Labels einschließlich Grounding/Fehlwertstatus**, 2.400 Artikelzellen, 24 primäre Retentionsmittelwerte und ihre 95-%-Intervalle, 24 längenadjustierte Mittelwerte und Intervalle sowie **16 gepaarte Methodenvergleiche** stimmen innerhalb 10⁻¹² mit den eingefrorenen Referenzen überein. Zusätzlich reproduziert die statische Original-QA-Auswertung aus einem frischen Clone **25/25 Tabellen**. Die vollständigen zwölf Detailanalysen wurden im Rahmen dieser gezielten Auditkorrektur nicht nochmals ausgeführt.
- **Doppelte dataset_index-Werte** werden vor Generierung abgelehnt: einschließlich numerisch gleicher Schreibweisen, Kollisionen mit automatisch vergebenen Indizes und Duplikaten außerhalb eines ausgewählten Limits. Auch der ursprüngliche Kandidaten-CLI wird vor `generate_stage` geprüft. Doppelte Kandidaten-IDs, mehrdeutige Artikel-/Stufenschlüssel und widersprüchliche Artikel-Hashes werden vor der Modellauswertung abgefangen.
- **Wiederaufnahme nach simuliertem Abbruch**: vorbereitete Trainings- und Validierungsartikel werden bei identischer Quelldatei, identischem Inhalt und belegter Konfiguration wiederverwendet. Datei-Hash und Änderungszeit bleiben unverändert; das Anfragekonto wird nicht zurückgesetzt. Veränderte Quellen, Vorbereitungsoptionen und Geräte führen zum Abbruch. Die originalen Modellparameter und Optimierungs-/Auswahlregeln bleiben erhalten.
- **README-Links geprüft**: fünf zentrale Tabellen und fünf Abbildungen sind als bytegleiche Kopien der bestehenden Referenzarchive verlinkt, mit belegter Zuordnung zu Kapitel 4.2–4.5. Der veraltete Verweis auf Text-/Wrapperfunktionen in `03_generate.py` zeigt jetzt auf `data/code/summary_rules.py`.
- Syntaxprüfung aller Python-Dateien. Die wissenschaftlichen Scorer-, Statistik-, Bootstrap-Worker- und Text-/Prompt-Kernmodule sind bytegleich zum Ausgangsstand. In angepassten Originalmodulen kamen nur vorgeschaltete Prüfungen hinzu. Die Änderungen der ausführbaren Originalkopien stehen als zusätzliche Audit-Diffs in `data/code_provenance.json`.
- **GitHub-Veröffentlichung geprüft**: Branch `fix/audit-reproduzierbarkeit` hochgeladen und [Pull Request #1](https://github.com/EdgarXB/Masterarbeit_Faktenretention_unter_rekursiver_Kompressionskette/pull/1) gegen `main` erstellt. Die Anmeldung erfolgte über die GitHub CLI ohne sudo; der zuvor blockierte Schreibzugriff der Anbindung wurde dadurch umgangen.


Maschinenlesbarer Nachweis: [audit_checks.json](audit_checks.json). Testprotokoll: [audit_tests.log](audit_tests.log). **0 echte LLM-Aufrufe, 0 Modelldownloads, keine neuen Experimente.**

## Verbleibende Einschränkungen

- Linux x86_64 ist die getestete Ausführungsumgebung. Linux unter WSL2 ist als Windows-Weg dokumentiert, aber WSL2 wurde nicht separat getestet. Native Windows- und macOS-Kompatibilität wird nicht zugesagt.
- Eine echte Modellinferenz und ein echter Optimierungs-Neustart wurden nicht getestet; Wiederaufnahme und Schnittstellen wurden ausschließlich mit simulierten Phasen geprüft. Frühere Vorbereitungsstände ohne ausreichenden Konfigurationsbeleg werden nicht stillschweigend übernommen.
- Für die Regressionscaches ist die exakte Versionskombination notwendig. Der unveränderte Originalfingerprint prüft zusätzlich Daten, Designmatrix, Punktkoeffizienten und Worker; andere Hardware/BLAS kann trotz gleicher Versionen einen Cachekonflikt verursachen. Es gab keine 20.000 neuen Bootstrap-Fits.
- Der öffentliche Seed-Dateieintrag stimmt mit dem ursprünglichen Weight-Hash überein. Für das ursprüngliche QA-Hauptgewicht und Gemma ist der exakte öffentliche Bezug nicht lückenlos nachgewiesen. Die dazugehörige Dokumentation verwendet keine Ersatzmodelle und behauptet keine exakte Reproduktion mit anderen Gewichten. Siehe [MODELLBEZUG](../data/MODELLBEZUG.md).
- Das vollständige Manuskript ist nicht Teil des Repositorys. Die Kapitelzuordnung wurde an der vorhandenen Arbeitsfassung geprüft; es werden keine ins Leere führenden Manuskriptlinks angeboten.

## Notwendige manuelle Schritte

- Vor einer echten Experimentwiederholung die fehlenden exakten Originalblobs beschaffen und deren Digests prüfen. Für Offline-Ergebnisreproduktion sind sie nicht erforderlich.
- Einen Release ausschließlich nach [RELEASE_CHECKLIST.md](../RELEASE_CHECKLIST.md) manuell veröffentlichen: geprüfte Commit-SHA auswählen, Release-Unveränderlichkeit einschalten, Version taggen, ZIP/Prüfsumme an einen Entwurf anhängen, erst anschließend veröffentlichen. **Kein Release und kein Release-Tag wurden automatisch erstellt.**
