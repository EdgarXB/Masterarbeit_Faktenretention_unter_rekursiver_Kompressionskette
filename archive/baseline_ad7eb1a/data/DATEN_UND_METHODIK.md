# Datenstand, Methodik und historische Grenzen

## Maßgebliche Quellen

Das Hauptpaket stammt aus `Masterarbeit_Abgabe/02_Auswertung` und den dort referenzierten Experimentartefakten in `03_Experiment`. Die ergänzenden vier Analysen stammen aus `Analysepaket_4_Notebooks_2026-10-17`, einschließlich der Grafikerweiterung vom 19.10.2026. `quelleninventar.json` enthält für jede verwendete Originaldatei den logischen Herkunfts- und Archivpfad, die Größe und SHA-256. Die Originaldateien wurden ausschließlich gelesen.

Die Hauptversion `aktuell_20260930` kombiniert vier ursprüngliche Verfahren mit den am 30.09. neu erzeugten Plain-/Few-Shot-Ketten, abgeschlossen am 01.10.2026. Die Ersetzung erfolgte nach dem ursprünglichen Freeze und wird als solche dokumentiert: 400 historische Ketten bleiben, je 100 Plain-/Few-Shot-Ketten ersetzen ihre früheren Fassungen. 32.000 QA-Positionen bleiben historisch, je 8.000 kommen aus den zwei ersetzten Bedingungen. Der ursprüngliche Stand `original_frozen` liegt ausschließlich als Vergleich für die vorhandene Versionssensitivität bei. Er wird niemals zum finalen Stichprobenumfang hinzuaddiert.

## Direkt lesbare Datendateien

| Datei | Zeile / Schlüssel | Inhalt |
|---|---|---|
| artikel.csv.gz | 100; article_id | Offizieller Datensatzindex, Artikeltext, Highlights, Text-Hash |
| fakten.csv.gz | 2.000; canonical_fact_id | Artikel-ID, Fakt, Frage, Referenzantwort, Ursprungsbeleg, Entity-Typ |
| zusammenfassungen.csv.gz | 600; article_id × method | Originaltext und vier rekursive Zusammenfassungen je Verfahren |
| qa_antworten.csv.gz | 48.000; Artikel × Verfahren × Stufe × Fakt | Gespeicherte Antwort, Beleg, Status, Fehler und technische Metadaten |
| modellmetriken.csv | 2.400; Artikel × Verfahren × Stufe | Gespeicherte ROUGE-, BERTScore-, SummaC- und weitere Einzelwerte |

Die ergänzenden Quelldateien liegen verlustfrei im Quellenarchiv. Das Inventar verweist bei identischem Inhalt auf bereits direkt gelieferte CSV-Dateien, Referenztabellen oder identische Referenznotebooks; diese werden nicht zusätzlich physisch kopiert. Der historische relative Pfad wird nur in der temporären Arbeitskopie wiederhergestellt. Die zwei notwendigen Bootstrap-Caches liegen im Quellenarchiv, nicht zusätzlich im Tabellenarchiv.

## Unveränderte wissenschaftliche Regeln

- Datensatz: `abisee/cnn_dailymail`, Konfiguration `3.0.0`, offizieller Testsplit. Auswahl und Ausschlüsse sind im historischen `prepare_autocot_automatecot_splits.py` und den Splitmanifesten belegt; die Auswahl ist keine nachträglich behauptete Zufallsstichprobe.
- Sechs Verfahren: Lead-k, Plain, Zero-Shot-CoT, Few-Shot-CoT, Auto-CoT, Automate-CoT.
- Rekursion: Original → 500 → 300 → 150 → 75 Wörter. Wortregel `\b\w+\b`; Fenster 475–525, 285–315, 142–158, 71–79.
- Pro Artikel 20 feste kanonische Fakten aus einem zuvor generierten Kandidatenpool; im Offlineweg keine neuen Fakten oder Fragen; der aktivierte Experimentweg erzeugt 24 Kandidaten und wählt die ersten 20 nach den unveränderten strikten Regeln.
- Finale QA-Regeln: exakt der archivierte `qa_scorer_snapshot.py`. Der Code wird von `04_evaluate.py` aus `data/code/qa_scorer_snapshot.py` importiert. Funktionen wurden per AST auf Identität geprüft.
- Grounding erfordert einen zusammenhängenden Beleg im tatsächlich bewerteten Text, nach Whitespace-Normalisierung und Casefolding. Technische Fehler werden aus dem gültigen Nenner ausgeschlossen und nicht zu Verlustlabels umetikettiert.
- Retained-Regel: Originalnormalisierung und Teilstringabgleich, Token-F1 ≥ 0,50, sichere kanonische Schreibweisen/Zahlenumformungen und die bestehende kurze-Antwort-Regel mit F1 ≥ 0,43 und unveränderten Längenbedingungen. Die Reihenfolge der Regeln bleibt unverändert. `NOT_FOUND` bedeutet lost; übrige gültige Nichttreffer sind reference_discordant.
- Primär: zunächst K/V je Artikel-Verfahrens-Stufe, anschließend gleichgewichteter Artikelmittelwert. Kein globales Pooling der Fakten als unabhängige Beobachtungen.
- Längenadjustiert: (K/20)/(1+D), mit D = max(U−W, W−O, 0)/Zielwortzahl. Technische Fehlwerte tragen bei festem Nenner 20 nicht zum Zähler bei.
- Bootstrap: 10.000 Artikelziehungen, Seed 32, punktweise 95-%-Perzentilintervalle. Methoden- und Stufenvergleiche erhalten die Artikelpaarung. Vorhandene Monte-Carlo-Sensitivitäten behalten ihre eigenen ursprünglichen Wiederholungszahlen und Seeds.
- Zwei leere Zero-Shot-Ausgaben erzeugen insgesamt 40 regelbasierte NOT_FOUND-Einträge ohne QA-Modellantwort. Eine Kette hat anschließend einen leeren Stufeneingang. Sensitivitäten halten technische Fehlwerte und Generierungsfehler getrennt.
- Längenkontrolle: gruppiertes binomiales GLM, feste Artikeleffekte, Verfahren×Stufe und stufenspezifische lineare Wortzahl; Sensitivität mit zentrierten natürlichen Splines (df=3). Support-Filter, Paarselektion, Standardisierung und Artikelbootstrap bleiben erhalten. Es gibt keine Verfahren×Wortzahl-Interaktion und keine dadurch begründbare optimale Wortzahl.
- Sekundärmetriken: vorhandene ROUGE-, BERTScore- und SummaC-Einzelwerte; lokale Eingangsreferenz bei leerem Stufeneingang bleibt gemäß Originalanalyse fehlend. Zahlenindikatoren werden rein textuell geprüft.

## Modelle und vollständiger historischer Ablauf

| Rolle | Alias | eingefrorener Digest |
|---|---|---|
| Zusammenfassung und Demonstrationsentwicklung | thesis-seed:v1 | 7a66a2f466bf48fdafa7004a7975a7f5fac6e667a6de7d01751aacb98b3f387c |
| Fakten/Fragen | thesis-gemma:v1 | 6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7 |
| QA-Antworten | thesis-qwen38-qa:v1 | d80421b9191aff0d4d8ea0911068452f11f3963e3e282c6b9d778932c4b268b3 |

Der Trainingsreserve-Nachtrag vor Validierung/Hauptlauf erweitert den ursprünglichen 200er-Kandidatenpool auf 240 Artikel; der adaptive Stopp erfolgte nach 207 Kandidatenartikeln. Die dazu gehörenden 200er-Dateien sind ausdrücklich als Herkunftsbelege dieses protokollierten Nachtrags erhalten, keine zusätzlichen Hauptdaten.

Die Rollen, Modellmetriken und genauen Software-/Modellrevisionen sind in den mitgelieferten Freeze-, Umgebungs- und Metrikmanifesten dokumentiert. Keine Modellgewichte gehören zum Anhang.

Der Originalcode im Quellenarchiv dokumentiert: Datensatzdownload und Filterung (`download_cnn_dm_articles_hf.py`, `prepare_autocot_automatecot_splits.py`), Fakten/Fragen (`extract_atomic_facts.py`), Kandidaten (`generate_cot_candidate_rollouts.py`), Auto-CoT-Clustering/-Auswahl (`cluster_autocot_traces.py`, `select_autocot_demonstrations.py`), Automate-CoT (`score_and_prune_automatecot_candidates.py`, `select_automatecot_demonstrations.py`), Few-Shot-Auswahl (`build_shared_pool_fewshot_prompts.py`), Hauptlauf und spätere Ersetzung (`run_six_method_final_100.py`, `run_plain_fewshot_rerun_final100.py`), QA (`answer_summary_questions.py`, `score_model_selection_batch.py`) sowie Metriken (`evaluate_autocot_automatecot_metrics.py`). Die lokalen Importabhängigkeiten sind mitarchiviert. Alte absolute Pfade bleiben als historische Belege erhalten; der Archivcode ist Herkunftsbeleg. Der ausführbare portable Weg verwendet die statischen Module in `data/code/`; deren Anpassungen und Hashes sind separat dokumentiert.

Im Hauptexperiment: Temperatur 0,0, Seed 32, top_p 0,95, repeat_penalty 1,0, Kontext 524.288, num_predict 4.096. Thinking-Wrapper ausschließlich bei Zero-Shot-CoT aktiv (Budgetanweisung 1.024, Reflexionsintervall 512); insbesondere bei Auto-/Automate-CoT im Hauptlauf deaktiviert. Kein Best-of, keine Längenrevision oder Summary-Retries. Die finale Plain-Bedingung lautet exakt `summarize in {target_word_length}` plus Leerzeile und Eingabetext. Die aktuellen Few-Shot-Prompts verwenden den gemeinsamen Kandidatenpool und sind nicht die ältere Few-Shot-Fassung im ursprünglichen Freeze-Manifest.

## Automate-CoT: gespeicherte Entwicklung, kein neuer Optimierungslauf

Pruning behält 100 Kandidaten je Zielstufe nach den gespeicherten strukturellen Filtern und SummaC-Eingangskonsistenz. ROUGE-L dient beim Pruning nur als exakter Gleichstandsbrecher. Für die Optimierung gibt es fünf geordnete Promptpositionen, zehn gezogene eindeutige Demonstrationskombinationen pro Update, 15 Updates je Stufe und fünf Epochen. Das Verfahren verwendet positionsspezifische Verteilungen, Policy Gradient (REINFORCE mit Leave-mean-out-Varianzreduktion) und Adam, Lernrate 0,001.

Während der Bewertungsgenerierungen: Temperatur **1,1**, aktiver Thinking-Wrapper mit **nicht technisch erzwungener 512-Token-Anweisung** und Reflexionsintervall 512. Optimierter Nutzen ist stemmed Highlight-ROUGE-L-F1 bei strukturell gültiger Ausgabe, andernfalls null; Verlust = 1−Nutzen. Das unterscheidet sich vom späteren Hauptlauf mit Temperatur 0,0 und deaktiviertem Wrapper.

Beigefügt sind alle 5.753 für die finale Optimierung benötigten Cacheausgaben mit strukturellen Urteilen, gespeicherten ROUGE-L-Einzelwerten und Nutzen/Verlust sowie die vier kleinen finalen Optimierer-Checkpoints (nur archiviert, nie deserialisiert). Aus den Caches werden 600 Update-Kombinationsverluste und vier finale Validierungsverluste unabhängig neu aggregiert; sämtliche Cachekeys, Nutzendefinitionen und Mittelverluste stimmen überein. Außerdem enthalten sind finale Demonstrationen, Kandidaten-/Pruningartefakte, Pruningaudit und -bericht, gezogene Kombinationen und ihre Verluste, Update-Mittel, finale positionsspezifische Wahrscheinlichkeiten und Abschlussberichte. `02_demonstrations.py` prüft die gespeicherten Dimensionen, die finale Greedy-Auswahl aus den Endwahrscheinlichkeiten und die Protokollkonsistenz, führt standardmäßig keine Optimierung aus. Mit `--construct --full-experiment` wird ausdrücklich die originale Konstruktionspipeline aktiviert.

## Human Validation und historische Grenzen

Die historische menschliche QA-Validierung umfasst 60 Fälle, sieben anonymisierte Reviewer und 204 vollständige Bewertungen bei ungleichmäßiger Beteiligung. Sie diente der Kalibrierung und ist keine unabhängige Fehlerratenschätzung des finalen 2.000-Fakten-Satzes. Die separate Faktenrelevanzprüfung und ihre anonymisierten Einzelwerte bleiben enthalten.

Der neuere Vergleich von drei gelieferten LLM-Urteilen ist **keine Human Validation**. Seine bereits vorhandenen 600 Urteile werden offline ausgewertet, ohne neue Modellaufrufe.

Nicht lückenlos archiviert sind die echte vollständige frühere Scorercodefassung, sämtliche ursprünglichen Clusterzuordnungen/Embeddings, alle QA-Retry-Zwischenantworten und positionsspezifische Optimierungswahrscheinlichkeiten nach jedem einzelnen Update. Diese Artefakte wurden nicht rekonstruiert oder erfunden. Archivierte Ursprungslabels, finale Antworten, Versuchszahlen, vorhandene Cluster-Audits, Kombinationsverluste und Endwahrscheinlichkeiten bleiben überprüfbar.

Die internen nativen Ollama-Chatrenderer und lückenlose historische CPU/RAM/CUDA-Messungen je Lauf sind nicht vollständig archiviert. Für BERTScore/SummaC gibt es keine ursprünglichen Einzel-Output-Hashes; die Zuordnung beruht auf dem exakt passenden Summary-Dateihash und den vorhandenen Laufprotokollen. Das Nachberechnen von Statistik validiert diese gespeicherten Werte, nicht nachträglich jeden damaligen Modellaufruf.

Der Anhang repariert keine historischen Designgrenzen: deterministische Artikelauswahl, Kalibrierung auf denselben Validierungsfällen, nachträgliche Plain-/Few-Shot-Ersetzung, nicht verifizierte Vorabfestlegung der vier Analysekontraste und fehlende Korrektur der 16 Intervalle für Mehrfachvergleiche. Referenzabweichungen belegen keine Summary-Halluzinationen. Diese Einschränkungen gelten auch bei numerisch erfolgreicher Reproduktion.

## Ressourcenschutz und Umfang der erneuten Prüfung

Nur Statistik und Textregeln wurden ausgeführt. Standardprozesse blockieren Netzwerkverbindungen, Modellbibliotheken und weitere Unterprozesse (Ausnahme: lokale Schriftartenabfrage `fc-list`). CUDA ist ausgeblendet; BLAS verwendet einen Thread. Keine Modellmetriken wurden neu inferiert. Die Regressionen werden als zwei Punktmodelle neu angepasst; die 20.000 Bootstrap-Anpassungen werden aus vorhandenen, fingerprintgebundenen Caches gelesen. Bei Cachekonflikt gibt es keinen automatischen Ersatzlauf. `05_analyze.py --recompute-regression-bootstrap` erlaubt ausdrücklich die originale Neuberechnung.

Der neue Mini-Test unterstützt die vollständigen Fakten-, Summary- und QA-Phasen für standardmäßig einen Artikel, eine Methode und eine Stufe, maximal 30 Generierungsrequests einschließlich Wiederholungen. Die Zusammenfassungsoption `--summary-only` erlaubt weitere Stufen oder bis zu fünf Artikel innerhalb desselben Budgets. Sämtliche Modellphasen wurden ausschließlich mit simulierten Antworten geprüft; echte Netzverbindungen waren in diesen Tests gesperrt. Es gab keine neue LLM-Generierung, keinen Gewichtdownload und keine erneute echte Optimierung.

Die historische Detailauswertung und die gemeinsame Auswertung neuer Daten sind getrennt: Die gemeinsame Auswertung verwendet identische Scorer-, Nenner-, Längen- und Bootstrap-Regeln. Historische Humanratings, Versionsvergleiche und die auf das ursprüngliche 100-Artikel-Panel zugeschnittenen Zusatzmodelle sind keine neuen Beobachtungen.
