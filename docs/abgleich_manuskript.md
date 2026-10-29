# Abgleich des Abgabemanuskripts mit dem Repository

Prüfstand: **29.10.2026**, vom Autor benannte Datei **`2910_Masterarbeit.docx`**.
SHA-256 der unverändert gelesenen Datei:
`4b4282f4260532db367f742ef55c3fecab5c9b47276670a79be71c515ae2ba04`.
Der Dateiname und Stand wurden vom Autor bestätigt. Das lokale Manuskript wird
nicht mitversioniert. Grundlage der Zuordnung ist
[auswertung_und_einordnung.md](auswertung_und_einordnung.md).

Die DOCX-Textabsätze, Tabellen und eingebetteten Abbildungsbilder wurden gelesen.
Seitenangaben folgen den gespeicherten Verzeichnissen der DOCX; ohne PDF-Export
wurde keine neue Paginierung behauptet. Abschnitt, Tabellen-/Abbildungsnummer
und Textzitate sind die maßgeblichen Fundstellen. Tabellenwerte wurden mit der
vor Änderungen vollständig ausgeführten Offline-Auswertung abgeglichen.
Rundungen werden in der jeweils gedruckten Präzision verglichen. „Übereinstimmend“
bezieht sich bei Tabellen auf die unten genannten Zahlen, bei Abbildungen auf
die ausdrücklich genannten Zahlen/Bildmerkmale; es bedeutet keine Pixelidentität.

## Tabellen

Alle numerischen Datenzellen der Tabellen 5–8, 15–18 und 20–25 wurden verglichen;
Tabelle 19 wurde semantisch gegen Spezifikation und Code geprüft. Die
[Einzelbelege](belege_abgleich_tabellen.csv) enthalten jede numerische Prüfung,
Manuskriptwert, ungerundeten Repo-Wert und Status. SD/Quartile der Tabelle 17
werden ergänzend aus den ursprünglichen Ausgabelängen berechnet, weil der
CSV-Export nicht alle diese Kennwerte enthält. Alle Quellenpfade unten sind
relativ zu `results/`, sofern nicht anders bezeichnet.

| Abschnitt/Tabelle/Abbildung im Manuskript | Quelle im Repo (Notebook/Datei) | Wert im Manuskript | Wert aus Repo-Auswertung | Status |
|---|---|---|---|---|
| Tabelle 5, 4.1.1, S. 39 | [01_menschliche_validierung/tabelle_5_top20_relevanz.csv](../results/01_menschliche_validierung/tabelle_5_top20_relevanz.csv) | Top 20 gesamt: 33 Core, 55 Supporting, 12 Peripheral | 33 / 55 / 12; auch alle fünf Artikelzeilen | übereinstimmend |
| Tabelle 6, 4.1.2, S. 42 | [01_menschliche_validierung/scorer_kalibrierung.csv](../results/01_menschliche_validierung/scorer_kalibrierung.csv) | 32/52; 36/52; 41/52 (61,5 / 69,2 / 78,8 %) | Identische Werte im eindeutigen Vergleichsset n=52 | übereinstimmend |
| Tabelle 7, 4.4.1, S. 52 | [04_verlustmuster/persistenz_nach_erster_stufe_95ci.csv](../results/04_verlustmuster/persistenz_nach_erster_stufe_95ci.csv) | Lead-k 1.407; 1.384 / 21 / 2 | Identisch; alle sechs Verfahren und Prozentanteile | übereinstimmend |
| Tabelle 8, 4.5, S. 54 | [05_weitere_metriken/tabelle_8_75_woerter.csv](../results/05_weitere_metriken/tabelle_8_75_woerter.csv) | Auto-CoT: 115,1 Wörter; QA 39,1; QA adj. 26,9; SummaC 0,263 | Identisch in Manuskriptpräzision; SummaC-Spalte ist source_summac (Originalartikel) | übereinstimmend |
| Tabelle 15, C.1, S. 80 | [02_faktenretention/retention_mittelwerte_95ci.csv](../results/02_faktenretention/retention_mittelwerte_95ci.csv) | Lead-k bei 500: 75,6 [72,8; 78,3] %; Auto-CoT bei 75: 39,1 [36,4; 41,7] % | Identisch; alle 24 Mittelwerte und 48 Intervallgrenzen | übereinstimmend |
| Tabelle 16, C.2, S. 81 | [02_faktenretention/vier_gepaarte_kontraste_pp_95ci.csv](../results/02_faktenretention/vier_gepaarte_kontraste_pp_95ci.csv) | Auto-CoT − Few-Shot-CoT, 300: untere Grenze 4,89 pp | 4,9803289473684185 pp → 4,98; übrige Zahlen übereinstimmend | abweichend |
| Tabelle 17, C.3, S. 82 | [03_textlaenge/laengeneinhaltung_24_gruppen.csv](../results/03_textlaenge/laengeneinhaltung_24_gruppen.csv) | Plain bei 500: 209,31 Wörter; SD 47,305; Q1 177,00; Q3 233,50 | Identisch; für SD und Quartile zusätzlich Roh-Wortzahlen aus length_values geprüft | übereinstimmend |
| Tabelle 18, D.1, S. 84 | [06_laengenregression/haupttabelle_16_originale_und_laengenkontrollierte_kontraste.csv](../results/06_laengenregression/haupttabelle_16_originale_und_laengenkontrollierte_kontraste.csv) | Auto–Few, 300: unadjustierte Untergrenze 4,89; bei 150: unadjustierter Supportabstand 3,91 pp | 4,98 bzw. 3,97 pp; übrige Zahlen einschließlich kontrollierter Intervalle übereinstimmend | abweichend |
| Tabelle 19, D.2, S. 85 | [06_laengenregression/tabelle_19_spezifikation.csv](../results/06_laengenregression/tabelle_19_spezifikation.csv) | Binomiales GLM, Logit; Artikel-Fixed-Effects; lineare Länge je Stufe; 5.–95. Perzentil; ≥20 | Grundspezifikation übereinstimmend; lokale ≥5-Ausgaben-Regel und Bandbreite fehlen in Tab. 19 (Kap. 3.9.2 nennt ≥5 ohne Bandbreite) | übereinstimmend (Grundspezifikation); Ergänzung offen |
| Tabelle 20, D.3, S. 86 | [02_faktenretention/tabelle_20_minimale_labelaenderungen.csv](../results/02_faktenretention/tabelle_20_minimale_labelaenderungen.csv) | Few–Plain bei 75: 14 Änderungen; Zero–Plain bei 300: 348 | Identisch; alle 16 Kontraste, Mindestzahlen und Prozentanteile | übereinstimmend |
| Tabelle 21, E.1, S. 87 | [04_verlustmuster/tabelle_21_entitaetsverteilung.csv](../results/04_verlustmuster/tabelle_21_entitaetsverteilung.csv) | NUMBER 526 Fakten / 96 Artikel; EVENT 80 / 53 | Identisch; alle acht Typen und fünf numerischen Spalten | übereinstimmend |
| Tabelle 22, E.2, S. 89 | [04_verlustmuster/tabelle_22_entitaetsretention_500.csv](../results/04_verlustmuster/tabelle_22_entitaetsretention_500.csv) | NUMBER, Lead-k: 80,4 % | 80,418251 %; alle 48 Zellen übereinstimmend | übereinstimmend |
| Tabelle 23, E.2, S. 89 | [04_verlustmuster/tabelle_23_entitaetsretention_300.csv](../results/04_verlustmuster/tabelle_23_entitaetsretention_300.csv) | NUMBER, Lead-k: 51,8 % | 51,809524 %; alle 48 Zellen übereinstimmend | übereinstimmend |
| Tabelle 24, E.2, S. 90 | [04_verlustmuster/tabelle_24_entitaetsretention_150.csv](../results/04_verlustmuster/tabelle_24_entitaetsretention_150.csv) | NUMBER, Lead-k: 33,1 % | 33,079848 %; alle 48 Zellen übereinstimmend | übereinstimmend |
| Tabelle 25, E.2, S. 90 | [04_verlustmuster/tabelle_25_entitaetsretention_75.csv](../results/04_verlustmuster/tabelle_25_entitaetsretention_75.csv) | NUMBER, Lead-k: 17,5 % | 17,490494 %; alle 48 Zellen übereinstimmend | übereinstimmend |

## Abbildungen

Die eingebetteten DOCX-Bilder wurden separat betrachtet und mit den neu erzeugten
Grafiken, Exporttabellen und Notebookberechnungen abgeglichen. Bei Kurven und
Boxplots ist dies ein visueller Abgleich mit Zahlenstichproben, keine vollständige
Rückgewinnung der Bilddaten. Bei Matrizen sind die genannten beschrifteten Zellen
geprüft. Abweichende Farben, Auflösungen oder Matrixlayouts ändern keine Werte.

| Abschnitt/Tabelle/Abbildung im Manuskript | Quelle im Repo (Notebook/Datei) | Wert im Manuskript | Wert aus Repo-Auswertung | Status |
|---|---|---|---|---|
| Abbildung 5, 4.1.1, S. 40 | Notebook 01; [abb_05_relevanzkonfusion.png](../results/01_menschliche_validierung/abb_05_relevanzkonfusion.png); kennzahlen.csv; data/menschliche_faktenrelevanz.csv | Konfusionsdiagonale 50 / 46 / 16; n=168 | Diagonale 112/168; Accuracy 66,7 %; Kappa 0,542 | übereinstimmend (genannter Prüfumfang) |
| Abbildung 6, 4.1.1, S. 41 | Notebook 01; [abb_06_relevanzranking.png](../results/01_menschliche_validierung/abb_06_relevanzranking.png); data/menschliche_faktenrelevanz.csv; kennzahlen_je_artikel.csv | Fünf Artikel, Ränge 1–34, Grenze Top 20 | Artikel- und Rangzuordnung stimmen; Top-20-Verteilung siehe Tab. 5 | übereinstimmend (genannter Prüfumfang) |
| Abbildung 7, 4.1.2, S. 42 | Notebook 01; [abb_07_qa_kalibrierung.png](../results/01_menschliche_validierung/abb_07_qa_kalibrierung.png); konfusion_model_label.csv; konfusion_calibrated.csv | Retained: 16 → 25; Lost: 16; Unsupported: 0 | Identische Konfusionszellen; Bezugsgruppen 30 / 19 / 3 | übereinstimmend (genannter Prüfumfang) |
| Abbildung 8, 4.2.1, S. 43 | Notebook 02; [abb_08_retentionsverlauf.png](../results/02_faktenretention/abb_08_retentionsverlauf.png); retention_mittelwerte_95ci.csv | Lead-k 500: 75,6 %; Auto-CoT 75: 39,1 % | Verläufe und Intervalle visuell konsistent mit vollständig geprüfter Tab. 15 | übereinstimmend (genannter Prüfumfang) |
| Abbildung 9, 4.2.1, S. 44 | Notebook 02; [abb_09_artikelverteilungen.png](../results/02_faktenretention/abb_09_artikelverteilungen.png); notebooks/02_faktenretention.ipynb (Artikelaggregation) | 75-Wort-Mediane generativ 30–40 % | Boxplots mit vier Stufen und sechs Verfahren; Mediane 30 / 35 / 35 / 40 / 35 % in generativer Methodenreihenfolge | übereinstimmend (genannter Prüfumfang) |
| Abbildung 10, 4.2.2, S. 45 | Notebook 02; [abb_10_gepaarte_kontraste.png](../results/02_faktenretention/abb_10_gepaarte_kontraste.png); vier_gepaarte_kontraste_pp_95ci.csv | Zero–Plain bei 75 negativ; Auto–Few alle Stufen positiv | Richtungen und grafisch ablesbare Intervalle stimmen; Präzisionsabweichung 300-Wort-Untergrenze in Tab. 16 gesondert aufgeführt | übereinstimmend (genannter Prüfumfang) |
| Abbildung 11, 4.3.1, S. 46 | Notebook 03; [abb_11_wortfenster.png](../results/03_textlaenge/abb_11_wortfenster.png); laengeneinhaltung_24_gruppen.csv | Lead-k 100 Treffer je Stufe; Plain 150: 39 / 22 / 39 | Identische Unter-/Treffer-/Überschreitungshäufigkeiten; Auto 500: 1 Treffer; Automate: 5 | übereinstimmend (genannter Prüfumfang) |
| Abbildung 12, 4.3.1, S. 47 | Notebook 03; [abb_12_wortlaengen.png](../results/03_textlaenge/abb_12_wortlaengen.png); laengeneinhaltung_24_gruppen.csv; Roh-Wortzahlen | Plain 500 um 209; Auto 75 um 115 Wörter | Vier Boxplotfelder/Wortfenster; Verteilungen visuell konsistent mit Tab. 17; zwei leere Zero-Ausgaben sichtbar | übereinstimmend (genannter Prüfumfang) |
| Abbildung 13, 4.3.2, S. 48 | Notebook 03; [abb_13_laengenadjustierung.png](../results/03_textlaenge/abb_13_laengenadjustierung.png); retention_und_drei_rangfolgen_24_gruppen.csv | 500: Plain 54,60 → 36,15 %; Automate 73,30 → 63,11 % | Identische Beschriftungen; grafischer fester Nenner K/20 ist von Tab. 15 (K/V) zu unterscheiden | übereinstimmend (genannter Prüfumfang) |
| Abbildung 14, 4.3.3, S. 50 | Notebook 03; [abb_14_matching_300_75.png](../results/03_textlaenge/abb_14_matching_300_75.png); matrixzellen_120_gerichtete_vergleiche.csv | 300: Zero–Few +4,96 pp, n=21; 75: Auto–Plain +3,89 pp, n=18 | Identisch; beide Grafiken zeigen das untere Dreieck; CSV enthält beide gerichteten Vergleiche | übereinstimmend (genannter Prüfumfang) |
| Abbildung 15, 4.4.1, S. 51 | Notebook 04; [abb_15_erste_nicht_retained_stufe.png](../results/04_verlustmuster/abb_15_erste_nicht_retained_stufe.png); erste_nicht_retained_stufe_95ci.csv | 500: Lead 24,3; Plain 44,9; Auto 28,4; Automate 26,5 % | Identisch; durchgehend retained: Zero 27,2 / Auto 35,9 %; Original ist keine gemessene QA-Stufe | übereinstimmend (genannter Prüfumfang) |
| Abbildung 16, 4.4.2, S. 53 | Notebook 04; [abb_16_bedingte_entitaetsverluste.png](../results/04_verlustmuster/abb_16_bedingte_entitaetsverluste.png); verlust_entitaetstyp_uebergaenge_95ci.csv | DATE_TIME: 18,7 → 42,5 % bei Übergängen auf 150/75 | Identische gepoolte bedingte Anteile (Zero / Auto / Automate); acht Typen und drei Übergänge | übereinstimmend (genannter Prüfumfang) |
| Abbildung 19, B.3, S. 79 | Notebook 01; [abb_19_relevanzurteile.png](../results/01_menschliche_validierung/abb_19_relevanzurteile.png); relevanzurteile_je_person.csv | Core P1–P5: 12,5 / 20,2 / 35,1 / 14,3 / 39,9 % | Identisch; je Person n=168 | übereinstimmend (genannter Prüfumfang) |
| Abbildung 20, B.4, S. 79 | Notebook 01; [abb_20_relevanzkappa.png](../results/01_menschliche_validierung/abb_20_relevanzkappa.png); relevanz_kappa_paare.csv | Matrix: Minimum 0,234; Maximum 0,435 (P3/P5) | Identisch gerundet; Fließtext nennt dagegen 0,434 als Maximum | übereinstimmend (genannter Prüfumfang) |
| Abbildung 21, C.4, S. 83 | Notebook 03; [abb_21_matching_500.png](../results/03_textlaenge/abb_21_matching_500.png); matrixzellen_120_gerichtete_vergleiche.csv | Zero–Auto +0,33 pp, n=29; Auto–Automate +0,46 pp, n=16 | Identisch; leere Vergleichszellen für Plain–Auto/Automate | übereinstimmend (genannter Prüfumfang) |
| Abbildung 22, C.4, S. 83 | Notebook 03; [abb_22_matching_150.png](../results/03_textlaenge/abb_22_matching_150.png); matrixzellen_120_gerichtete_vergleiche.csv | Plain–Lead +17,81 pp, n=19; Auto–Zero −3,63 pp, n=16 | Identisch; kleine Gruppen gestrichelt markiert | übereinstimmend (genannter Prüfumfang) |
| Abbildung 23, E.2, S. 88 | Notebook 04; [abb_23_entitaetsretention.png](../results/04_verlustmuster/abb_23_entitaetsretention.png); retention_entitaetstyp_methoden_stufen_95ci.csv | NUMBER n=526/96 Artikel; EVENT n=80/53; acht Verlaufsfelder | Typengrößen identisch; Verläufe und Intervalle visuell konsistent mit Tab. 21–25; keine exakte Digitalisierung der Kurven | übereinstimmend (genannter Prüfumfang) |

## Gezielte Klärung der bekannten Punkte

### Intervallgrenze Auto-CoT minus Few-Shot-CoT bei 300 Wörtern

Die Aufgabenbeschreibung und die frühere Repo-Doku vermuteten **4,89** als
Rohdatenwert. Das ist für den vorliegenden Daten-/Codestand nicht bestätigt:
Tabellen 16 und 18 des Manuskripts drucken beide **4,89**, die Auswertung ergibt
jedoch **4,9803289473684185** (95-%-Intervall **[4,98; 9,32]** bei Mittelwert
**7,12 pp**). Bei **500 Wörtern** lautet die untere Grenze hingegen
**4,893530701754386**, gerundet **4,89**. Auch `tests/referenzwerte.csv` enthält
diese stufenspezifische Unterscheidung.

Unabhängige Nachrechnung: gültige rohe QA-Antworten der beiden Verfahren auf
300 filtern, eingefrorene `retained`-Regel anwenden, je Artikel K/V bilden,
Artikel numerisch aufsteigend ordnen, dann Auto minus Few in Prozentpunkten.
Mit `np.random.default_rng(32).integers(0,100,size=(10000,100))` Artikel ziehen
und 2,5-/97,5-%-Quantile der mittleren Differenzen berechnen. Mittelwert:
**7,1192982456140355 pp**, Untergrenze **4,9803289473684185 pp**. Diese Prüfung
verwendet nicht `statistics.estimate`. Keine Daten oder Referenzwerte wurden geändert.

### Existiert E.1? Alle Textfundstellen

- **Kapitel 4.4.2**, Absatz beginnend „Ebenso wurde die Faktenretention …“:
  „Die vollständige Verteilung der Ausgangsfakten nach Entitätstyp ist in
  Anhang E.1 dokumentiert.“ Der Verweis passt zur vorhandenen **Tabelle 21**.
- **Anhangsverzeichnis**, Eintrag „E.1.Datengrundlage je Entitätstyp86“
  (so im gespeicherten Verzeichnis). Der eigentliche Abschnitt enthält Tabelle 21,
  die im Tabellenverzeichnis auf S. 87 steht; die Verzeichnisse vor Abgabe aktualisieren.
- Eine wörtliche Referenz auf **„Tabelle E.1“** ist im DOCX-Haupttext nicht vorhanden.
  Der Dateiname `results/05_weitere_metriken/anhang_E_1_kendall_tau_b.csv` ist
  eine historische Exportbezeichnung. Die Kendall-Auswertung wird in Kapitel 4.5
  beschrieben, ohne dort einen Verweis auf eine fehlende Tabelle E.1.

### Zuordnung der 0,08 pp bei 500 Wörtern

Quelle: `results/03_textlaenge/berichtete_adjustierte_abstaende_100_98.csv`.
Der adjustierte Kontrast ist **Zero-Shot-CoT − Auto-CoT**, n=100:
**0,076360 pp**, gerundet **0,08**, Intervall **[−1,95; 2,08]**.
Die führenden generativen Verfahren sind dagegen **Automate-CoT (63,1115 %)**
und **Zero-Shot-CoT (57,4682 %)**; ihr deskriptiver Abstand ist **5,643294 pp**.
Für diesen führenden Kontrast wird hier kein neues Intervall behauptet.

| Manuskriptteil | Tatsächliche Formulierung / Fundstelle | Ergebnis |
|---|---|---|
| Kapitel 4.3.2, S. 48/49 | „Die gepaarten Abstände zwischen den führenden generativen Verfahren … 0,08pp … bei 500“ | Falsche Paarzuordnung; korrigieren |
| Kapitel 5.3 | Rangwechsel und kleinere Spannweite bei 150; keine 0,08-pp-Angabe | Kein entsprechender Fehler gefunden |
| Kapitel 7 | Rangwechsel und Längenkontrolle; keine 0,08-pp-Angabe | Kein entsprechender Fehler gefunden |
| Deutsche Zusammenfassung / englischer Abstract | Keine 0,08-pp-Angabe | Kein entsprechender Fehler gefunden |

Davon zu unterscheiden sind **0,08 pp** als untere unadjustierte Intervallgrenze
für **Automate-CoT − Auto-CoT bei 500 Wörtern** (Kap. 4.2.2, Tab. 16/18) und
der adjustierte **75-Wort-Kontrast bei n=98** (Zero–Auto, nach Ausschluss zweier
Artikel; 0,075153 pp). Beide haben andere Bezugsgrößen.

### Original-QA und Lead-k-Bezugsgruppen

Notebook 02 enthält nach der Textkorrektur nur Aussagen über die gemessenen
Übergänge 500→300, 300→150 und 150→75. Die fehlende Original-QA-Baseline erlaubt
keine Quantifizierung eines Retentionsrückgangs Original→500. Kapitel 4.2.1 und
5.1 des Manuskripts vermischen dies noch mit dem Anteil erstmaliger
Nicht-Retained-Urteile, der in Notebook 04 separat ausgewiesen ist. Kapitel 7
verwendet bereits die vorsichtigere Formulierung „Anteil erstmaliger Nicht-nachweise“.

`lead_k_rueckwechsel_bezugsgruppen.csv` reproduziert beide Angaben:

- **21/1.407 (1,492537 %)**: erste Nicht-Retained-Stufe eindeutig vor 75
  (500, 300 oder 150); davon später mindestens einmal observed retained.
- **22/1.999 (1,100550 %)**: mindestens ein gültiges Stufenpaar unter allen sechs
  möglichen Paaren der vier Stufen; mindestens ein nicht-retained→retained-Paar.
  Fehlwerte an anderen Stufen verhindern die Aufnahme nicht automatisch.

Kapitel 4.4.1 führt die erste Gruppe mit Tabelle 7 ein, Kapitel 4.1.2/6.1 nennt
die zweite. Die Gruppen sind damit grundsätzlich unterschieden, jedoch wird
der genaue Nenner „mindestens ein gültiges Paar unter allen sechs Paaren“ und
der Grund für den unterschiedlichen Zähler nicht explizit zusammen erklärt.
Die Angaben bleiben erhalten; ein Querverweis wird vorgeschlagen.

## Offen / vom Autor zu klären

- Vorgeschlagene Änderungen aus [korrekturliste_manuskript.md](korrekturliste_manuskript.md)
  im Abgabemanuskript prüfen und selbst übernehmen; das Manuskript wurde nicht bearbeitet.
- Nach Überarbeitung Abbildungs-/Tabellen-/Anhangsverzeichnis neu aktualisieren
  und Seitenzahlen anhand des finalen PDF prüfen.
- Die Quellenwerte der nur als Bild verfügbaren Kurven sind nicht aus dem
  Manuskript rückgewinnbar; der Bildabgleich ist entsprechend begrenzt.
