# Modelle und Beschaffung

Für neue Läufe beschreibt [Neuen Lauf einrichten](reproduktion_neuer_lauf.md)
die Installation, Downloads und Einstellungen. Die Offline-Auswertung
verwendet gespeicherte Modellbewertungen und benötigt keine Modellgewichte.

## Sprachmodelle

| Finale Rolle | Ollama-Manifest-SHA-256 | Beleg in den finalen Daten |
|---|---|---|
| Seed: generative Zusammenfassungen | `7a66a2f466bf48fdafa7004a7975a7f5fac6e667a6de7d01751aacb98b3f387c` | 500 Artikel-Verfahren-Zeilen; Lead-k ist deterministisch |
| Gemma: kanonische Fakten und Fragen | `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7` | 2.000 kanonische Fakten |
| Qwen: finale QA-Antworten | `d80421b9191aff0d4d8ea0911068452f11f3963e3e282c6b9d778932c4b268b3` | 48.000 QA-Positionen, einschließlich technischer Fehlwerte |

Die Manifeste und Konfigurationsbelege liegen unter
[`data/model_metadata/`](../data/model_metadata/), die Modellzuordnung in
[`data/models.json`](../data/models.json). Die großen Gewichte sind nicht
enthalten; das Downloadskript prüft die SHA-256-Werte. Aktuelle Tags können
vom historischen Stand abweichen. Die spätere
[QA-Änderung](../data/model_metadata/provenance/qa_amendment_20260901.json)
ersetzt den ursprünglich vorgesehenen GLM-Answerer durch Qwen.

Die Bezeichnung Qwen3.8 bezeichnet das verwendete GGUF-Gewicht;
`qwen35`/`qwen3.5` bezeichnet dessen Architektur und Renderer, nicht ein
austauschbares `qwen3.5:27b`-Modell.

## Einstellungen

| Phase | Dokumentierte Einstellungen und Prüfgrenze |
|---|---|
| Finale Zusammenfassungen | Seed: Temperatur 0, Seed 32, `top_p=0.95`, `repeat_penalty=1.0`, `num_ctx=524288`, `num_predict=4096`, ein Versuch je Stufe. Für Plain/Few-Shot sind diese Werte im Rerun-Protokoll belegt; Temperatur, Seed, Versuche und Wrapper-Flag sind für alle 2.000 generativen Stufen gespeichert und geprüft. |
| Thinking im Hauptlauf | Zero-Shot nutzt den Seed-Raw-Wrapper mit instruiertem Budget 1.024 und Reflection-Intervall 512; die anderen generativen Verfahren nutzen den Wrapper ohne aktives Thinking-Budget. Die Werte sind Promptanweisungen, keine Garantie modellseitiger Einhaltung. |
| Kandidaten und Optimierung | Historisch gelten dafür eigene Einstellungen, darunter Temperatur 1,1 und ein instruiertes Thinking-Budget von 512; sie sind nicht mit dem finalen Summary-Lauf gleichzusetzen. |
| Fakten | Gemma: Temperatur 0, Basis-Seed 32, Kontext 8.192, maximal 4.096 Ausgabetoken; 24 Kandidaten angefordert, 20 gültige Fakten je Artikel übernommen. Die Pipeline verwendet bei Bedarf weitere Versuche. |
| QA | Qwen: Temperatur 0, Basis-Seed 32, Kontext 8.192, maximal 512 Ausgabetoken, `think=false`; technische Wiederholungen verwenden Seeds 32, 33 und 34. Nicht alle Zwischenantworten sind archiviert. |

Die Budgets für Thinking und Reflection sind Promptinstruktionen, keine
garantierten Modellgrenzen. Details enthalten das
[Rerun-Protokoll](../data/methoden/protocol.json) und die
[historischen Modellbelege](../data/model_metadata/provenance/).

## Gewichtsquellen

| Datei | Exakte öffentliche Quelle | Erwarteter SHA-256 |
|---|---|---|
| Seed-Gewicht | [Ollama-Registry-Blob](https://registry.ollama.ai/v2/milkey/Seed-OSS-36B-Instruct/blobs/sha256:2d9f3c32a61f2668e012708be7ec04e753e35e366f87b9e4bce89f4d073cfc16) | `2d9f3c32a61f2668e012708be7ec04e753e35e366f87b9e4bce89f4d073cfc16` |
| Historisches Gemma-Gewicht | [Ollama-Registry-Blob](https://registry.ollama.ai/v2/library/gemma4/blobs/sha256:280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313) | `280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313` |
| QA-Textgewicht | [Qwen3.8-27B-Q4_K_M.gguf, feste Revision](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/0669b98607d47046c7c2b3f801011d54a08cfccf/Qwen3.8-27B-Q4_K_M.gguf) | `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| QA-Projektor | [mmproj-Qwen3.8-27B-Q8_0.gguf, feste Revision](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/0669b98607d47046c7c2b3f801011d54a08cfccf/mmproj-Qwen3.8-27B-Q8_0.gguf) | `2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb` |

## Metrikressourcen

| Ressourcenschlüssel | Quelle | Festgelegte Revision bzw. Prüfsumme |
|---|---|---|
| `embedding` | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` |
| `bertscore` | [`distilroberta-base`](https://huggingface.co/distilroberta-base) | `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b` |
| `summac_nli` | [`tals/albert-xlarge-vitaminc-mnli`](https://huggingface.co/tals/albert-xlarge-vitaminc-mnli) | `3082ba54344bd9ddada2be1c5e9b4131721d2a5d` |
| `summac_head` | [`summac_conv_vitc_sent_perc_e.bin`](https://github.com/tingofurro/summac/blob/master/summac_conv_vitc_sent_perc_e.bin) | SHA-256 `cd880581f59bd47b8968c7f48d754cd617dfe9b256d8d76fdcbc0fa4a3b3c6fe` |
| `nltk_data` | NLTK-Downloader, Paket `punkt_tab` | englische Dateien unter `tokenizers/punkt_tab/english/` |

Die historische NLTK-Datenrevision ist nicht separat belegt. SummaC 0.0.4
verlangt `huggingface-hub<=0.17.0`, Transformers 5.15.1 dagegen `>=1.5.0`.
Die getestete Installation umgeht diesen Metadatenkonflikt mit `--no-deps`;
`pip check` meldet ihn weiterhin. Bei einem Drei-Artikel-Test wurden BERTScore
und SummaC für alle 72 Zusammenfassungen auf Linux/CUDA berechnet.
