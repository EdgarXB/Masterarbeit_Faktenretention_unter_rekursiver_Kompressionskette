# Wissenschaftliche Datenintegrität: Prüfprotokoll

Prüfstand: 31. Oktober 2026. Die Prüfung betrifft den aktuellen Arbeitsbaum
einschließlich der noch nicht committeten Bereinigungen. Sie untersucht den
eingefrorenen Datenbestand und startet keine Modellgenerierung. Der ausführbare
Prüfweg ist `python scripts/validate_wissenschaftliche_integritaet.py`; er
erstellt seine vorbereiteten Eingaben in einem temporären Verzeichnis und
gibt Befunde und Demonstrations-IDs als JSON aus. Voraussetzung ist die in
der README beschriebene Analyseumgebung.

## Ergebnis und Geltungsbereich

Die finalen Daten, die geprüften Haupt- und Sensitivitätsrechnungen und die
erzeugten Ergebnistabellen sind innerhalb der nachstehend genannten Grenzen
konsistent. Die Prüfung beweist weder die inhaltliche Wahrheit der Fakten noch
die fachliche Richtigkeit jeder einzelnen maschinellen Antwort. Insbesondere
lassen die entfernten Originalartikel und die fehlenden unveränderten
historischen Merge-Eingaben keinen lückenlosen Nachweis aller früheren
Verarbeitungsschritte allein aus dem digitalen Anhang zu.

## Vollständigkeit und rekursive Ketten

- 100 eindeutige Artikel, 2.000 eindeutige Fakten, genau 20 je Artikel,
  600 eindeutige Artikel-Verfahren-Zeilen und 2.400 eindeutige Stufen. Jede
  der sechs Methoden enthält jeden Artikel auf allen vier Stufen.
- 48.000 eindeutige QA-Positionen entsprechen Artikel × Verfahren × Stufe ×
  kanonischem Fakt; Artikel-ID, Datensatzindex und Artikel-Hash stimmen zwischen
  den geprüften Eingabedateien überein.
- Alle 2.400 gespeicherten Stufenausgaben wurden mit der entsprechenden
  Zusammenfassung verglichen. Bei allen 1.800 Übergängen 500→300, 300→150 und
  150→75 ist der gespeicherte Eingang exakt die vorangegangene Ausgabe.
  Für die 600 Eingänge Original→500 sind Artikel-Hash und Wortzahl gespeichert,
  die Originaltexte wurden aber entfernt: Ein direkter Text- oder Hashabgleich
  der Eingänge ist daher für diese Stufe nicht mehr möglich.

## Ersetzte Plain- und Few-Shot-Läufe

Die [Ergebnis-](../data/methoden/result_merge_manifest.json) und
[QA-Merge-Manifeste](../data/methoden/qa_merge_manifest.json) bleiben erhalten.
Die Prüfung der tatsächlichen Zeilen bestätigt 400 historische und je 100 neue
Plain- und Few-Shot-Artikel-Verfahren-Zeilen. Die 600 Zustände weisen für die
beiden neuen Verfahren durchgängig die Rerun-Signatur des
[Protokolls](../data/methoden/protocol.json) und die passende Herkunft auf;
die vier anderen Verfahren tragen die historische Signatur.

In den QA-Daten sind entsprechend 32.000 Positionen den vier historischen
Verfahren und je 8.000 den beiden Reruns zugeordnet. Für jede Methode sind es
dieselben 100 Artikel, und jede QA-Zeile trägt die erwarteten Experiment-,
Batch-, Artikel- und Stufenkennzeichen. In den finalen QA-Daten wurden keine
alten Plain-/Few-Shot-Batchkennzeichen gefunden.

Ein zusätzlicher positionsweiser Vergleich mit dem **nur lokal vorhandenen,
ignorierten** historischen Stand in `data/prepared/qa/original_frozen/` ergab
keine Änderungen an den 1.600 gespeicherten Textausgaben der 400 beibehaltenen
Zusammenfassungszeilen und keine
Änderungen der Antwort-, Beleg-, Status- und Fehlerfelder der 32.000
beibehaltenen QA-Positionen. Alle 800 Plain-/Few-Shot-Stufenausgaben änderten
sich; bei 10.846 der 16.000 QA-Positionen änderte sich mindestens eines dieser
vier Felder. Identische Antworten allein belegen weder alte noch neue Herkunft.
Dieser lokale Zusatzvergleich ist auf einer frischen Kopie nicht verfügbar.
Die in den Manifesten angegebenen historischen Datei-Hashes lassen sich ohne
die ursprünglichen unveränderten Merge-Dateien nicht erneut byteweise belegen.

## Prompts, Auswahl und Datentrennung

Alle zwölf finalen Promptdateien stimmen mit den SHA-256-Werten des
Bankmanifests überein; die vier Few-Shot-Prompts und deren Manifest stimmen
zusätzlich mit den Werten im Rerun-Protokoll überein. Für jede der zwölf
Prompt-Stufen wurden die fünf ausgewählten Kandidaten samt Reihenfolge und
Ausgabe gegen die gespeicherte Auswahl geprüft. Die Eingaben der Stufen
300/150/75 stimmen mit den Kandidatendaten überein; bei Stufe 500 stimmen die
Artikel-Hashes der Prompt-Eingaben mit den Auswahlbelegen überein.

Die archivierten Splits enthalten 240 Trainings-, 30 Validierungs- und 100
Testartikel. Es gibt keinen paarweisen Überlapp ihrer gespeicherten
Artikel-Hashes. Die 100 Testschlüssel stimmen mit den finalen Artikeln überein.
Alle 60 ausgewählten Demonstrationspositionen lassen sich einem
Trainingsartikel und seiner Kandidaten-ID zuordnen; die 120 archivierten
Validierungskandidaten gehören zum Validierungssplit. Das Demonstrationsaudit
rekonstruiert 600 Update-Verluste und vier finale Validierungsverluste anhand
der gespeicherten Caches. Die Positionen und Auswahlunterlagen zeigen keinen
Rückgriff auf finale Testergebnisse. Das schließt nicht mit absoluter Sicherheit
aus, dass außerhalb der erhaltenen Artefakte einmal Testinformationen
eingesehen oder Einfluss genommen haben.

Die unveränderten 500-Wörter-Prompts enthalten weiterhin Originalartikel als
Demonstrationen; ihre Entfernung wurde ausdrücklich zurückgestellt.

## QA-Nenner, gepaarte Vergleiche und Sensitivität

| Kategorie | QA-Positionen |
|---|---:|
| Retained | 25.074 |
| Lost | 17.558 |
| Reference-discordant | 5.172 |
| Technisch fehlend | 196 |
| Gesamt | 48.000 |

Die 196 technischen Fehlwerte sind als `technical_missing` klassifiziert und
stehen weder unter `lost` noch unter den anderen inhaltlichen Labels. Alle
47.804 übrigen Positionen haben genau eines der drei inhaltlichen Labels.
Die 40 QA-Positionen zweier leerer Ausgaben sind regelbasierte NOT_FOUND-Fälle,
keine technischen QA-Fehlwerte. Aus den Positionslabels wurden K, V, K/V und
K/20 für alle 2.400 Artikel-Verfahren-Stufen unabhängig aggregiert. Die
24 gespeicherten Nennervergleiche stimmen. Gegen den nur lokal vorhandenen
archivierten QA-Scorerstand stimmen alle 48.000 Positionslabels überein.

Alle 24 Hauptmittelwerte und 16 gepaarten Hauptkontraste samt 95-%-Intervallen
wurden aus den Artikelwerten mit 10.000 artikelweisen Bootstrap-Ziehungen und
Seed 32 unabhängig nachgerechnet und stimmen bis auf `1e-12`. Jede Paarung
verwendet dieselben Artikel-IDs. Die 24 Gruppen × drei Kennzahlen der
Längenanalyse und die fünf veröffentlichten adjustierten Vergleiche wurden
ebenfalls nachgerechnet. Die 98er-Sensitivität entfernt die Artikel 40 und 91
aus **beiden** verglichenen Methoden und verwendet sonst dieselbe finale
Datenbasis. Notebook 06 verwendet für den Regressionsbootstrap artikelweise
Häufigkeitsgewichte und prüft den gespeicherten Cache gegen Eingaben und
Fingerprint; die 10.000 Modellanpassungen wurden nicht neu durchgeführt.

## Ergebnisdateien und Ausführung

Alle sechs Notebooks wurden in einer isolierten Repository-Kopie jeweils in
einem eigenen Python-Prozess ausgeführt; die fehlenden Jupyter-Sockets dieser
Umgebung wurden dafür nicht benötigt. Alle 74 versionierten Ergebnisdateien
wurden neu geschrieben. In den 56 CSV-Dateien stimmen alle fachlichen Spalten
mit den versionierten Ergebnissen überein (numerische Toleranz `1e-10`); die
gemessenen Laufzeitspalten `fit_seconds` und `seconds_this_run` variieren. Die
18 erzeugten PNGs stimmen in dieser Umgebung pixelweise mit den versionierten
Abbildungen überein. Alle 28 Unit-Tests sowie Quellen- und Demonstrationsaudit
bestanden. Die isolierte Notebookausführung überprüft die Codezellen, ersetzt
aber keine separate fachliche Begutachtung aller Aussagen im Manuskript.
