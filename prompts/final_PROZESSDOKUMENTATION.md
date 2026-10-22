# Prozessdokumentation: Überarbeitung des Few-Shot-CoT-Prompts

## Ziel

Für das erneute Few-Shot-CoT-Experiment sollten vier neue, stufenspezifische
Prompts erstellt werden. Jeder Prompt enthält fünf zufällig ausgewählte
Demonstrationen für genau eine der rekursiven Kompressionsstufen:

- 500 Wörter
- 300 Wörter
- 150 Wörter
- 75 Wörter

Die Demonstrationen sollten aus einem gemeinsamen Auto-CoT-/Automate-CoT-Pool
stammen und nur Kandidaten verwenden, die für beide Auswahlverfahren zulässig
sind.

## Verwendete Quelldaten

Als Ausgangspunkt wurden die bereits erzeugten und auditierten Kandidaten des
Auto-CoT-/Automate-CoT-Projekts verwendet. Die maßgeblichen Artefakte waren:

1. `automate_cot_pruning/automate_cot_pruned_pools.json`
   - Enthält pro Kompressionsstufe die 100 Kandidaten, die nach den
     Automate-CoT-Hard-Filtern und der anschließenden SummaC-Rangfolge in den
     tatsächlichen Automate-CoT-Auswahlpool aufgenommen wurden.
2. `automate_cot_pruning/automate_cot_pruning_report.json`
   - Dokumentiert die Automate-CoT-Zulassungs- und Pruningregeln.
3. `auto_cot_selection/auto_cot_selection_report.json`
   - Dokumentiert das Auto-CoT-Auswahlverfahren und dessen Qualitätsregeln.

Die SHA-256-Hashes dieser drei Quellen sind im erzeugten
`few_shot_cot_manifest.json` gespeichert. Dadurch lässt sich später prüfen,
auf welcher konkreten Version der Quelldaten die Prompts beruhen.

## Definition des gemeinsamen zulässigen Pools

Der Begriff „gemeinsamer Pool“ wurde konservativ als Schnittmenge umgesetzt:

1. Ein Kandidat musste in den echten, auf 100 Kandidaten begrenzten
   Automate-CoT-Pool der jeweiligen Stufe aufgenommen worden sein.
2. Derselbe Kandidat musste zusätzlich sämtliche Auto-CoT-Qualitätsregeln
   erfüllen.

Damit wurden weder bloß generierte noch durch Automate-CoT später verworfene
Kandidaten verwendet. Ebenso wurden Automate-CoT-Kandidaten ausgeschlossen,
die nicht den zusätzlichen Auto-CoT-Grenzwert für die Länge des Reasonings
erfüllten.

Ein Kandidat wurde nur zugelassen, wenn alle folgenden Bedingungen erfüllt
waren:

- Stufenzuordnung stimmt mit dem Zielwert überein.
- Eingabetext ist nicht leer.
- Reasoning ist nicht leer.
- Das Reasoning enthält mindestens 25 substantielle Wörter.
- Das Reasoning enthält höchstens 512 Wörter.
- Finale Zusammenfassung ist nicht leer.
- Finale Zusammenfassung liegt im vorregistrierten ±5-%-Wortfenster.
- Finale Zusammenfassung endet mit einem vollständigen Satz.
- Die Eingabelänge entspricht dem erwarteten Bereich der jeweiligen Stufe:
  - Stufe 500: 795–805 Wörter
  - Stufe 300: 475–525 Wörter
  - Stufe 150: 285–315 Wörter
  - Stufe 75: 142–158 Wörter
- Der Kandidat ist im Automate-CoT-Pool mit
  `selected_into_pool = true` markiert.
- Es liegen keine Automate-CoT-Hard-Filterfehler vor.

Nach Anwendung dieser gemeinsamen Regeln ergaben sich folgende Poolgrößen:

| Kompressionsstufe | Automate-CoT-Pool | Für beide Verfahren zulässig |
|---:|---:|---:|
| 500 | 100 | 35 |
| 300 | 100 | 48 |
| 150 | 100 | 62 |
| 75 | 100 | 53 |

Der häufigste zusätzliche Ausschlussgrund war ein Reasoning mit mehr als 512
Wörtern. Dieser Grenzwert stammt aus der Auto-CoT-Qualitätsprüfung.

## Zufallsauswahl

Die Auswahl wurde reproduzierbar mit Python `random.Random(32)` durchgeführt.
Es wurde ein einziger Zufallszahlengenerator mit Seed 32 verwendet. Die Stufen
wurden in der festgelegten Reihenfolge `500, 300, 150, 75` verarbeitet.

Vor jeder Auswahl wurde der jeweilige zulässige Pool nach `candidate_id`
sortiert. Danach wurden mit `random.sample` fünf unterschiedliche Kandidaten
gleichverteilt und ohne Zurücklegen ausgewählt. Die Reihenfolge der gezogenen
Kandidaten entspricht ihrer Demonstrationsposition im Prompt.

Ausgewählt wurden:

### 500 Wörter

1. `train:11458:500`
2. `train:17125:500`
3. `train:14913:500`
4. `train:25676:500`
5. `train:22782:500`

### 300 Wörter

1. `train:25475:300`
2. `train:10233:300`
3. `train:9587:300`
4. `train:10238:300`
5. `train:11239:300`

### 150 Wörter

1. `train:20941:150`
2. `train:24765:150`
3. `train:21448:150`
4. `train:7856:150`
5. `train:10570:150`

### 75 Wörter

1. `train:24943:75`
2. `train:6726:75`
3. `train:24463:75`
4. `train:18212:75`
5. `train:3932:75`

## Aufbau der Prompts

Für jede Stufe wurde ein eigenständiger Prompt erzeugt. Jeder Prompt besteht
aus:

1. einer Aufgabenbeschreibung mit Zielwortzahl und zulässigem Wortfenster;
2. der Anweisung, nur durch den aktuellen Input belegte Fakten zu verwenden;
3. der Anweisung, mit einem vollständigen Satz zu enden;
4. fünf Demonstrationsblöcken mit jeweils:
   - `INPUT:`
   - `STEP-BY-STEP REASONING:`
   - `FINAL_SUMMARY:`
5. genau einem Platzhalter `{{NEW_INPUT}}` für den zur Laufzeit einzusetzenden
   Eingabetext;
6. dem Beginn des Reasoning-Abschnitts für die neu zu erzeugende Antwort.

Die Wortfenster der finalen Zusammenfassungen sind:

| Ziel | Zulässiges Fenster |
|---:|---:|
| 500 | 475–525 Wörter |
| 300 | 285–315 Wörter |
| 150 | 142–158 Wörter |
| 75 | 71–79 Wörter |

## Erzeugte Dateien

Das neue Promptpaket liegt unter:

`experiments/few_shot_cot_shared_pool_20260930/`

Es enthält:

- `few_shot_cot_500.txt`
- `few_shot_cot_300.txt`
- `few_shot_cot_150.txt`
- `few_shot_cot_75.txt`
- `few_shot_cot_manifest.json`
- `README.md`
- diese Prozessdokumentation

Der reproduzierbare Builder liegt im Projektstamm unter:

`build_shared_pool_fewshot_prompts.py`

## Audit-Manifest

Das Manifest dokumentiert für jede Stufe:

- Größe des ursprünglichen Automate-CoT-Pools;
- Größe der gemeinsamen zulässigen Menge;
- ausgeschlossene Kandidaten und deren Ausschlussgründe;
- geordnete IDs der fünf zufällig ausgewählten Kandidaten;
- Dataset-Index und Artikel-ID jeder Demonstration;
- Eingabe-, Reasoning- und Ausgabelängen;
- SummaC-Rang im Automate-CoT-Pool;
- SHA-256-Hashes von Eingabe, Reasoning und Zusammenfassung;
- Dateiname und SHA-256-Hash des fertigen Prompts.

## Durchgeführte Validierungen

Nach der Erzeugung wurden folgende Prüfungen erfolgreich durchgeführt:

- Der Builder ist syntaktisch gültig (`python3 -m py_compile`).
- Es wurden genau vier Promptdateien erzeugt.
- Jede Promptdatei enthält genau fünf Demonstrationen.
- Jede Promptdatei enthält genau einen `{{NEW_INPUT}}`-Platzhalter.
- Die fünf Demonstrationen einer Stufe sind eindeutig.
- Jeder ausgewählte Kandidat befindet sich im echten Automate-CoT-Pool.
- Jeder ausgewählte Kandidat besitzt keine Automate-CoT-Hard-Filterfehler.
- Jeder ausgewählte Kandidat erfüllt zusätzlich die Auto-CoT-Regeln.
- Die Dateihashes stimmen mit dem Manifest überein.
- Kein ausgewählter Artikel überschneidet sich mit dem
  Selektions-Validierungsset oder dem finalen 100-Artikel-Testset.

## Abgrenzung zum bisherigen Few-Shot-CoT-Paket

Die früheren Few-Shot-CoT-Prompts und die damit erzeugten historischen
Ergebnisse wurden nicht überschrieben. Das neue Paket liegt in einem eigenen
Verzeichnis und kann deshalb separat geprüft und anschließend gezielt für den
erneuten 100-Artikel-Lauf verwendet werden.

Mit der Erstellung und Validierung der Prompts wurden noch keine neuen
Summary- oder QA-Modellaufrufe gestartet.
