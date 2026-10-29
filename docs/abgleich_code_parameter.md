# Abgleich der Codeparameter mit dem Manuskript

Geprüft am **29.10.2026** gegen **`2910_Masterarbeit.docx`**. Rein lesender
Review von `src/faktenretention/`, Einstiegsskripten und den einschlägigen
Auswertungszellen. Es wurden keine Modellanfragen gestartet. Angaben zu
wissenschaftlichen Parametern und Ergebnissen wurden im Code nicht verändert.
Die freigegebene Python-Mindestversion ist eine separate Metadatenänderung.

Die Zeilen beziehen sich auf den geprüften Repo-Stand. Bei Notebooks wird die
physische JSON-Zeile zusätzlich zur Zelle/Funktion genannt; Outputzeilen sind
kein Codebeleg. „Übereinstimmend“ bestätigt die Implementierung der Angabe,
keine Garantie identischer neuer Modelloutputs.

| Parameter | Manuskript | Code-Stelle mit Datei:Zeile | Status |
|---|---|---|---|
| Summary: Temperatur | Kap. 3.8, Tab. 10: Hauptversuch 0,0 | `src/faktenretention/pipeline.py:175`; Übergabe an Transport; `model_transport.py:37` | übereinstimmend |
| Summary: Seed | Kap. 3.8: 32 | `src/faktenretention/pipeline.py:175`; `model_transport.py:38` | übereinstimmend |
| Summary: top_p | Kap. 3.8: 0,95 | `src/faktenretention/pipeline.py:175`; `model_transport.py:42` | übereinstimmend |
| Summary: repeat_penalty | Kap. 3.8: 1,0 | `src/faktenretention/pipeline.py:175`; `model_transport.py:50` | übereinstimmend |
| Summary: num_ctx | Kap. 3.8, Tab. 9: 524.288 | `src/faktenretention/pipeline.py:175`; `model_transport.py:39` | übereinstimmend; Transport-Default 8192 ist für diesen Aufrufer überschrieben |
| Summary: Ausgaben/Versuche | Tab. 9/10: max. 4.096 Tokens; ein Versuch je Stufe | `src/faktenretention/pipeline.py:176`, `pipeline.py:184`, `pipeline.py:188` | übereinstimmend; keine Längenrevision im Hauptversuch |
| QA: Temperatur, Seed | Kap. 3.6.2/3.8, Tab. 10: 0,0 / 32 | `src/faktenretention/pipeline.py:213`; `src/faktenretention/qa.py:385` | übereinstimmend |
| QA: Wiederholungen | Kap. 3.6.2/3.8: bis drei Versuche mit Seeds 32/33/34 bei ungültigem Format, technischen Fehlern oder fehlendem Beleg | `src/faktenretention/pipeline.py:214`; `qa.py:374`, `qa.py:386`, `qa.py:394`, `qa.py:401` | übereinstimmend; drei Versuche insgesamt, nicht drei zusätzliche Retries; erste gültige Antwort beendet die Schleife |
| Wortzählung | Gemeinsame Wortregel, Lead-k erste k Wörter | `src/faktenretention/text_rules.py:6`, `text_rules.py:115`; Auswertung `data.py:75` | übereinstimmend: `\b\w+\b`; Interpunktion wird nicht als Wort gezählt |
| Wortfenster ±5 %, Rundung | Kap. 3.3: 475–525 / 285–315 / 142–158 / 71–79 | `src/faktenretention/text_rules.py:9`; `protocol.py:7` | übereinstimmend; margin=`max(1, round(target*0.05))` ergibt 25/15/8/4 |
| Lead-k-Rekursion | Kap. 3.4.1 beschreibt direktes Abschneiden des Originals | `src/faktenretention/pipeline.py:165`, `pipeline.py:171`, `pipeline.py:189`; `text_rules.py:115`; Notebook 04, Zelle 17, Präfixprüfung | **abweichende Ablaufbeschreibung, gleiche Ausgabe**: Code verarbeitet die Vorstufe; Präfixverschachtelung macht dies zum direkten Originalpräfix äquivalent |
| Auto-CoT: untere Rationalegrenze | Kap. 3.5.1: mindestens 25 Wörter | `src/faktenretention/demonstrations/select_autocot_demonstrations.py:83`; Generierung `generate_cot_candidate_rollouts.py:165` | übereinstimmend; zählt substantive thinking ohne Reflexionsmarker |
| Auto-CoT: obere Rationalegrenze | Kap. 3.5.2: höchstens 512 Wörter | `src/faktenretention/demonstrations/select_autocot_demonstrations.py:85` | übereinstimmend; Obergrenze gilt für gesamte gespeicherte Thinking-Wortzahl, nicht für die Summary und nicht für das Tokenbudget |
| Kandidaten: Versuche | Kap. 3.5.1, Tab. 10: bis zu zwölf Generierungsversuche je Stufe | `src/faktenretention/pipeline.py:285`; `demonstrations/generate_cot_candidate_rollouts.py:142`, `generate_cot_candidate_rollouts.py:362` | übereinstimmend; zusätzlich existieren technische API-Retries innerhalb eines Generierungsversuchs |
| Faktenauswahl: 24→20 | Kap. 3.6.1, Tab. 12: 24 angeforderte Kandidaten, erste 20 gültige | `src/faktenretention/pipeline.py:129`; `facts.py:254`, `facts.py:263`; Vollständigkeitsgate `pipeline.py:134` | übereinstimmend; angefordert sind 24, es wird nicht garantiert, dass ein Modell stets exakt 24 liefert |
| Duplikatschlüssel | Kap. 3.6.1 spricht von Fragen; Tab. 12 von normalisierten Fakttexten | `src/faktenretention/facts.py:173`, `facts.py:175` | **abweichend im Fließtext**; Code prüft Fakttexte, Tab. 12 stimmt |
| Längenadjustierung | Kap. 3.9.1: `(K/20)/(1+D)`, D = Abstand zum zulässigen Wortfenster / Ziel | `src/faktenretention/data.py:78`, `data.py:142`, `data.py:154` | übereinstimmend; innerhalb des Fensters D=0; technisches Missing wird beim festen Nenner nicht zum retained |
| Artikelbootstrap | Kap. 3.9, Tab. 19: 10.000 Artikelziehungen, 95-%-Perzentilintervalle; Aufgabenanforderung Seed 32 | `src/faktenretention/statistics.py:4`, `statistics.py:15`, `statistics.py:17`, `statistics.py:38`; Notebook 06, Zelle 14 | übereinstimmend; gepaarte Differenzen vor dem Ziehen; Seed 32 implementiert, Tab. 19 nennt ihn nicht ausdrücklich |
| Regression: Überlappung | Kap. 3.9.2, Tab. 19: Schnitt des 5.–95. Perzentilbereichs | `notebooks/06_laengenregression.ipynb:918` (`SUPPORT_DEFS`); `:921` (`support_for`, Zelle 6) | übereinstimmend; `central95` ist nur ein interner Name, die implementierten Grenzen sind tatsächlich 0,05/0,95 |
| Regression: Mindestzahlen | Kap. 3.9.2/Tab. 19: ≥20 Beobachtungen je Verfahren und ≥20 gemeinsame Artikel | `notebooks/06_laengenregression.ipynb:107` (`MIN_SUPPORT`), `:940`, `:942` | übereinstimmend; Code verlangt **lokal gestützte gepaarte** Artikel; Tabelle 19 ist hierzu unvollständig |
| Regression: lokales Band | Aufgabenanforderung max(5 Wörter, 10 %); Manuskript nennt mindestens fünf Ausgaben je Verfahren im „Umfeld“ | `notebooks/06_laengenregression.ipynb:933`, `:934` | Code entspricht der Aufgabenanforderung; **Bandbreite im Manuskript nicht spezifiziert**. Für beide beobachteten Wortzahlen jedes Artikelpaars wird jedes Verfahren geprüft |
| Seed-OSS: Thinking Hauptversuch | Tab. 10: Zero-Shot 1.024, übrige Hauptverfahren 0 | `src/faktenretention/pipeline.py:174`; `src/faktenretention/text_rules.py:87`, `text_rules.py:100` | übereinstimmend; aktiver Wrapper nur Zero-Shot; `thinking_budget=1024` wirkt nur bei `cot=True` |
| Seed-OSS: Thinking Kandidaten/Optimierung | Tab. 10: 512 | `src/faktenretention/text_rules.py:85`; `demonstrations/generate_cot_candidate_rollouts.py:77`; `demonstrations/select_automatecot_demonstrations.py:109` | übereinstimmend; beide Aufrufer verwenden das Default-Budget 512 |
| Seed-OSS: Reflection-Interval | In Tab. 10 und zugehörigem Fließtext kein numerischer Wert | `src/faktenretention/text_rules.py:85` (allgemeiner Default 128); `pipeline.py:174`, `generate_cot_candidate_rollouts.py:363`, `select_automatecot_demonstrations.py:109` (jeweils 512) | **nicht prüfbar als Manuskriptübereinstimmung**; tatsächliche relevanten Aufrufe setzen 512. Default 128 allein ist kein Beleg für die Experimentkonfiguration |

## Bestätigung der Rundung und Ausführungsgates

Direkt unter Python 3.13.15 geprüft: `round(7.5)==8` und `round(3.75)==4`;
`bounds_for_target(t,5.0)` liefert für 500/300/150/75 exakt die vier oben
genannten Fenster. Python rundet bei exakten Halbwerten zur geraden Zahl;
die Funktion rundet den **Randabstand** und addiert/subtrahiert diesen vom Ziel.
Separates Runden von 95/105 Prozent wäre eine andere Regel.

`03_generate.py` aktiviert echte Generierung nur mit `--test` oder
`--full-experiment`; `04_evaluate.py` verlangt für neue QA/Metrikinferenz
zusätzlich `--execute`. Die Offline-Prüfung verwendete keine dieser Modellflags.
Die Unittests verwenden kontrollierte Ersatzantworten. Eine echte neue
Modellgeneration wurde weder durch diesen Review noch durch Notebookausführung
validiert. Die SummaC-Installationsgrenze steht in der Modelldokumentation.

## Offen / vom Autor zu klären

- Die Ablaufbeschreibung von Lead-k sowie den Duplikatschlüssel im Fließtext
  präzisieren; keine Änderung der Implementierung nötig.
- Lokales Band und gepaarte Unterstützung in Tabelle 19 ergänzen.
- Reflection-Interval 512 optional in Tabelle 10 nachtragen; eine tatsächliche
  Durchsetzung von Thinking-/Reflection-Budgets durch das Modell wird vom Code
  nicht garantiert, sondern als Prompt instruiert.
- Tabellenabweichungen und die übrigen Textkorrekturen sind in
  [korrekturliste_manuskript.md](korrekturliste_manuskript.md) gebündelt.
