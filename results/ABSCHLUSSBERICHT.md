# Abschlussbericht: eigenständig ausführbarer digitaler Anhang, Version 2

## Kritischer Audit und behobene Lücken

Die vorherige Fassung erfüllte die vollständige Experimentausführung nicht. Die Korrektur verwendet nun statische Kopien der tatsächlichen Forschungsfunktionen; eine Archivbeschreibung ersetzt die Ausführung nicht mehr.

| Einstieg | Vorher | Jetzt |
|---|---|---|
| 01_prepare.py | Bestand prüfen/Archiv entpacken | zusätzlich eigene CSV laden, Originalfilter 795–805 Wörter, Highlights, Duplikate, Reihenfolge, normalisierte Artikel-IDs und Provenienz |
| 02_demonstrations.py | nur historische Prompts und Optimierungscaches prüfen | zusätzlich explizite Originalpipeline: gemeinsame Kandidaten, Gates, MiniLM/KMeans, SummaC-Pruning, VR-PGE/Adam, finale Auswahl/Reihenfolge, gemeinsame Few-Shot-Auswahl |
| 03_generate.py | begrenzte Summary-Demonstration | Mini- oder separater Vollmodus: originale 24→20 Faktenauswahl, sechs Methoden, rekursive vier Stufen, QA und Metriken; gemeinsames Anfragebudget vor jedem POST |
| 04_evaluate.py | gespeicherte QA nur klassifizieren | originale QA-Generierung/Grounding/Retry-Regeln für neue Tabellen sowie Originalscorer; semantische Metriken mit ausdrücklich bereitgestellten lokalen Ressourcen |
| 05_analyze.py | Notebookquelltext zur Laufzeit ändern | zwölf statische .py-Auswertungen, Originalscorer/-Statistikimporte; gemeinsame Artikelstatistik für neue/fixe Daten; optionale originale Regression-Bootstrap-Neuberechnung |

Die Phasenverbindung steht in `data/code/research_pipeline.py`; das ist prozedurale Dateiverarbeitung, kein neues Framework. Das wissenschaftliche Verhalten stammt aus den Originalmodulen. Die Transportfunktion reserviert zentral vor jedem echten Request das dauerhafte Budget. Nach Erreichen des Budgets wird auch innerhalb historischer Retry-Schleifen abgebrochen. Aufrufparameter bleiben explizit; es gibt keine automatische Kandidatenentwicklung.

## Ausgeführte Prüfungen

- Exakt **0 echte LLM-Generierungsrequests**, 0 Gewichtdownloads, keine GPU-Modelltests, keine echte neue Kandidatengenerierung oder Optimierung.
- Vollständiger Offline-Neulauf aller zwölf statischen Auswertungen: **289/289 statistische Tabellen** bei absoluter Toleranz 10⁻¹²; zusätzlich eine erwartete Änderung des Dateiinventars (vier nicht erneut gespeicherte große Bootstrap-Zwischenarrays).
- **65 neu gerenderte PNGs pixelidentisch** zu den Originalen; 70 ursprüngliche Abbildungen bleiben archiviert. Neu erzeugt wurden außerdem PDF-/SVG-Ausgaben. Fünf rein historische Referenzabbildungen wurden nicht als neu erzeugt ausgegeben.
- Gemeinsame neue/fixe Auswertung: dieselben 2.400 Artikel-Verfahren-Stufen stimmen in 11 Zähler-, Nenner-, Retentions- und Statusfeldern überein; alle 24 primären Bootstrapmittelwerte und ihre Intervallgrenzen stimmen ebenfalls überein.
- Sechs Schnittstellentests mit ausdrücklich simulierten Antworten und gesperrter echter Netzverbindung: vollständiger Mini-Ablauf mit 22 simulierten Requests, Wiederaufnahme ohne neue Requests, Abbruch vor Anfrage 31 einschließlich Fehlversuchen, unzulässige Mini-Konfigurationen, alle sechs Verfahren/vier Stufen mit 501 simulierten Requests sowie Konstruktionssteuerung mit den originalen expliziten Parametern. Fehlende Runtime und falsche Modelldigests führen im separaten Test vor jeder Generierung zum Abbruch. Der sechste Test ist der separate Konstruktionskontrolltest; der Mini-Test umfasst Wiederaufnahme in derselben Testfunktion.
- Syntaxprüfung aller Python-Dateien. Die genauen Laufprotokolle und maschinenlesbaren Resultate liegen unter `results/`.

Ein Mock-Test prüft Dateiformate, Kontrollfluss und Ressourcengrenzen, nicht die Qualität oder tatsächliche Ausführbarkeit externer Modellgewichte in einer neuen Laufzeit. Eine echte End-to-End-Modellreproduktion ist wegen der verbindlichen Null-Aufruf-Vorgabe nicht Teil dieser Prüfung.

## Ressourcen und historische Grenzen

Die normale Analyse liest die beiden vorhandenen Regression-Bootstrap-Caches nach vollständiger Fingerprintprüfung. Der originale Neuberechnungscode ist jetzt statisch vorhanden und ausdrücklich aktivierbar, wurde aber nicht mit 20.000 neuen Fits ausgeführt. Die übrigen gespeicherten Modellmetriken werden offline wiederverwendet. Neue Experimentdaten können ROUGE, BERTScore und SummaC aus den expliziten lokalen Ressourcen neu berechnen.

Externe Modellgewichte, Ollama und gegebenenfalls CUDA sind erforderlich, aber nicht beigefügt. Aliasse, Manifeste, Digests, bekannte Bezugskennungen und exakte Revisionen sind dokumentiert. Für Gemma ist kein vollständiger Bezugsweg belegt; heutige Verfügbarkeit wird nicht behauptet. Ersatzmodelle gelten nicht als äquivalent. Die HF-Revisionserkennung lokaler Ordner ist eine Versionsdeklaration; nur der SummaC-Head und die Ollama-Manifeste haben hier ausdrücklich überprüfte exakte Digests.

Neue Daten erhalten die gemeinsame Artikelstatistik mit unveränderten wissenschaftlichen Regeln. Historische Humanratings, die Vorher-/Nachher-Versionssensitivität und die auf das ursprüngliche 100-Artikel-Panel zugeschnittenen statistischen Längenkontrollmodelle bleiben als historische Zusatzanalysen getrennt. Sie werden nicht als neu erhobene Daten oder neu angepasste Modelle ausgewiesen.

Die unabhängige Portabilitätsprüfung führte alle fünf Einstiegsskripte, darunter zwei ausgewählte Detailanalysen, aus einer verschobenen Paketkopie aus; Originalprojekte, frühere Paketordner und Netzwerk waren per Audit-Hook gesperrt. Die 25 QA-Tabellen und 14 LLM-Vergleichstabellen stimmten erneut überein. `results/portabilitaet.json` dokumentiert die sechs Befehle (03 ist im Standard ausschließlich Bestandsprüfung).

Die ursprünglichen Referenzdaten bleiben bytegleich in den Archiven erhalten. Darin gespeicherte historische Herkunftstexte können ursprüngliche Pfade enthalten; kein solcher Pfad wird zur Laufzeit als Eingabe geöffnet. In den ausführbaren Dateien und aktiven Inventaren gibt es keine absoluten Benutzerpfade. Der vollständige Änderungsnachweis für die statischen Originalmodule steht in `data/code_provenance.json` (historische Pfadangaben im Diff relativiert). Ursprungsdateien wurden nicht verändert.

Die ursprünglichen Studiendesigngrenzen bleiben bestehen: deterministische Auswahl, Kalibrierung auf denselben Validierungsfällen, spätere Plain-/Few-Shot-Ersetzung, Mehrfachvergleiche, eingeschränkter Längensupport und fehlende historische Zwischenartefakte. Einzelheiten: `data/DATEN_UND_METHODIK.md`.

## Dateien und Codeumfang

Vier Hauptunterordner: `data`, `prompts`, `results`, `figures`; fünf reguläre Einstiegsskripte. Zusätzlich sind statische Originalmodule und konkret benötigte Pfad-, Budget- und Statistikfunktionen unter `data/code` enthalten. Ihre höhere Zahl ist erforderlich, damit der Anhang das Experiment tatsächlich ausführen kann und die langen ursprünglichen wissenschaftlichen Analysen nicht in fünf riesigen Dateien versteckt werden. Archivierte historische Quellen und Notebooks sind reine Referenzen.

| Einstieg | Physische Zeilen | Nichtleer, ohne reine Kommentare |
|---|---:|---:|
| 01_prepare.py | 27 | 21 |
| 02_demonstrations.py | 169 | 147 |
| 03_generate.py | 47 | 41 |
| 04_evaluate.py | 38 | 32 |
| 05_analyze.py | 144 | 134 |

Insgesamt 38 physisch enthaltene ausführbare Python-Dateien einschließlich Originalmodule und eines Testskripts, 14,711 physische Zeilen. Davon sind die zwölf statischen historischen Auswertungen ausdrücklich mitgezählt. Keine verkürzte Startskriptzahl wird als Gesamtumfang ausgegeben.
