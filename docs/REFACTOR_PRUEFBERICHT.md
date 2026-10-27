# Notebook-Umstrukturierung: Prüfung vom 26.10.2026

Ausgangspunkt: GitHub-`main`, Commit
`ad7eb1a86039dbc44651e4e2cdb1dc358c3d0f14`. Umsetzung in einer unabhängigen lokalen
Arbeitskopie auf `codex/reproducible-notebooks`. Die bereits vorhandene Arbeitskopie
auf `fix/audit-reproduzierbarkeit` wurde nicht geändert. Kein Push, Merge oder Release.

## Ergebnis und Architektur

Die zwölf historischen Analysen liegen als normale, ausgeführte Notebooks unter
`notebooks/`, mit erhaltenen fachlichen Erklärungen, expliziten Pfaden, dokumentierten
Abhängigkeiten und insgesamt 65 eingebetteten PNGs. Alle Codezellen bleiben unter
100 Zeilen. Die Originalnotebooks bleiben bytegetreu im vorhandenen Referenz-ZIP.
Neue Laufdaten haben ein eigenes Ergänzungsnotebook.

Das Paket `src/faktenretention/` trennt Dateizugriff, Textregeln, QA-Scoring, Statistik,
Tabellenaufbereitung, Experimentphasen, Transport/Anfragebudget und Notebook-Prüfung.
Die zwölf aktiven Python-Analysekopien und der überholte Pilot-CLI sind nach
erfolgreichem Ergebnisabgleich archiviert. Der gemeinsame Tabellenexport übernimmt
bereits geprüfte Labels, Retentionsmittel/-intervalle, Entitätsquotientenschätzer und
Sekundärmetrikmittel. Seine zusätzlichen gepaarten, gepoolten und Rang-Auswertungen
bleiben erhalten. Unterschiedliche Schätzer wurden nicht pauschal vereinheitlicht.

Scorer, Textregeln und der fingerprintrelevante Bootstrap-Worker sind auch im aktiven
Paket byteidentisch mit dem Ausgangsstand. Weitere extrahierte Funktionen werden
gegen ihre Original-AST geprüft. Shared Cachekeys und Demonstrationsparameter
werden gemeinsam durch Offline-Audit und Optimierungsmodul verwendet. Die
Linux-Anfragesperre wird von reinen Analyseimporten nicht benötigt.

## Tatsächlich ausgeführte Prüfungen

| Prüfung | Ergebnis |
|---|---|
| Unveränderte Ausgangsversion | 15 bestehende Tests bestanden |
| Abschließende Testsuite | **28 Tests bestanden**, 7,655 s; [Protokoll](validation/tests.log) |
| Historische Hauptauswertung | **12/12 vollständig mit jeweils frischem Kernel ausgeführt** |
| CSV-Vergleiche | **289 statistische Tabellen und ein Artefaktinventar bestanden** |
| Numerische Abweichung | Maximum **2,842170943040401e-14**, unveränderte absolute Toleranz `1e-12` |
| Referenz-PNGs | **65/65 pixelidentisch** |
| Weitere Grafikexporte | **101 PDF-/SVG-Dateien vollständig vorhanden** |
| Unveränderte Bestandsprüfsummen | **47/47 bestätigt**, einschließlich archivierter früherer Codepfade |
| Quellen- und Datenintegrität | **6.288 logische Quellen**, 100 Artikel, 2.000 Fakten, 2.400 Artikel-Verfahren-Stufen, 48.000 QA-Positionen; 196 technische Fehlwerte und zwei leere Ausgaben bestätigt |
| Demonstrationsaudit | Alle 5.753 gespeicherten Generierungscaches und die gespeicherten Kombinationsverluste ohne neue Optimierung geprüft |
| Neue Laufdaten | Zusätzliches Mini-Fixture mit **22 simulierten und null echten Requests**, separate QA-Neuauswertung und Ergänzungsnotebook erfolgreich |
| Paketinstallation | Editable-Installation in eigener Analyseumgebung; `uv pip check`: 77 kompatible Pakete einschließlich des lokalen Pakets |
| Import-/Pfadverhalten | Reine Analyseimporte ohne Experimentmodule und bei simuliert fehlendem `fcntl`; Aufruf und Import außerhalb des Repository-Arbeitsverzeichnisses bestanden |
| Einstiegsskripte | Hilfe aller fünf Wrapper, Analyse-Liste/Umgebungsprüfung, Offline-Datenprüfung, Demonstrationsaudit, QA-Auswertung und historische/neue Notebook-Ausführung geprüft |
| Darstellungsartefakte | Alle zwölf Notebooks als HTML exportiert; drei zentrale PNGs visuell geprüft |

Der [maschinenlesbare Prüfbericht](validation/notebooks.json) enthält alle 355
CSV-/PNG-Vergleiche und die zusätzlichen Grafikexportprüfungen. Die vollständige
Ausführung wurde nach Entfernen der alten aktiven Kopien wiederholt. Eine danach
behobene reine Anzeigeanpassung der Matrixgrafiken wurde mit beiden erforderlichen
Vorgängern erneut ausgeführt; ihr [gesonderter Bericht](validation/matrix-display.json)
ist erhalten. Die zusätzliche SVG-Datei der kombinierten Matrix ist ausdrücklich im
Inventar enthalten. Numerische Identität und PNG-Identität wurden anschließend
erneut geprüft; die Vergleichslogik verwendet auch numerische Identitätsschlüssel.

Der Notebook-Runner startet erwartete Ergebnisse ohne vorab kopierte Referenztabellen.
Fehlende CSVs, PNGs oder zusätzliche Grafikexporte, leere Vergleichslisten, getrennte
Einzel-/Gesamtläufe, Teilausführung und abhängiges Überspringen sind gezielt getestet.
Originale und neue Ausgaben werden nicht anhand von Änderungszeiten unterschieden.
Bewusst wiederverwendet: historische ROUGE-Nachberechnung, die zwei vollständigen
Regressionsbootstrap-Caches mit unverändertem Worker und historische ergänzende
Scorer-Metadaten. Die 20.000 Regression-Bootstrap-Anpassungen wurden nicht neu gerechnet.

Die visuell geprüften Abbildungen sind der QA-Retentionsverlauf, Retention gegenüber
Längenabweichung sowie originale gegenüber längenkontrollierten Kontrasten. Es wurden
keine Layout- oder Inhaltsabweichungen gegenüber den Referenz-PNGs gefunden. Ein
Browserrenderer stand nicht zur Verfügung; daher wird keine vollständige visuelle
Browserprüfung aller HTML-Seiten behauptet. Die Quellen, eingebetteten PNGs und
HTML-Exporte wurden strukturell geprüft. Anzeigeausgaben in versionierten Notebooks
verwenden für Rechnerpfade `${PROJECT_ROOT}` bzw. `${HOME}`; Messdaten und Roh-Exporte
wurden dadurch nicht verändert.

## Ressourcenschutz

Arbeit ausschließlich in eigener Arbeitskopie und eigener Analyseumgebung.
Rechenläufe wurden seriell, niedrig priorisiert und mit einem BLAS-/OpenMP-Thread
ausgeführt; die vollständigen Notebook-Läufe zusätzlich mit CPU-Affinität auf einen
Kern und 8 GiB Adressraumlimit. Keine GPU-Nutzung, keine echten LLM-Aufrufe, keine
Gewichtdownloads und keine neue echte Optimierung. Laufende Dienste, Container,
Modellumgebungen und globale Systemeinstellungen wurden nicht verändert.

Der bereits laufende `VLLM::EngineCore` mit PID 6564 lief bei der Abschlusskontrolle
weiter, mit unverändert 89.726 MiB GPU-Speicherbelegung. Rund 109 GiB RAM waren
weiter verfügbar. Das ist eine Abschlussbeobachtung und keine Garantie über jede
zwischenzeitliche Latenz eines anderen Prozesses.

## Verbleibende Grenzen

Der unveränderte historische Analyse-Lock wurde tatsächlich installiert und geprüft.
Der neue Experiment-Core-Lock wurde lediglich anhand von Paketmetadaten für Linux
x86_64/Python 3.13 aufgelöst. Die umfangreichen Modellpakete wurden nicht installiert
und Modellqualität/GPU-Laufzeit nicht getestet. Der Core-Lock gilt ausdrücklich nur
für `--without-model-metrics` und nicht für neue Demonstrationskonstruktion.

Der historische Vollstand ist konkret widersprüchlich: SummaC 0.0.4 benötigt
`huggingface-hub<=0.17.0`, Transformers 5.15.1 dagegen `>=1.5.0,<2.0`.
[Resolvernachweis](experiment-resolution-full.txt) und
[Umgebungsgrenzen](umgebungen.md) benennen das offen; keine historischen Pins wurden
zum Verbergen des Konflikts abgeändert. Fünf ursprüngliche Demonstrationsgrafiken
bleiben historische Referenzen und werden nicht als neue Notebook-Exporte ausgegeben.

Der Whitespace-Check ist für die bearbeiteten Dateien sauber; der bytegetreu übernommene Scorer behält seine ursprüngliche zusätzliche Leerzeile am Dateiende und wird hierfür ausdrücklich ausgenommen.

Die Linux-CI wurde angelegt, aber ohne Push nicht auf GitHub gestartet. Die lokalen
entsprechenden Prüfungen sind oben belegt. Historische wissenschaftliche Grenzen aus
[`data/DATEN_UND_METHODIK.md`](../data/DATEN_UND_METHODIK.md) bleiben unverändert gültig.
