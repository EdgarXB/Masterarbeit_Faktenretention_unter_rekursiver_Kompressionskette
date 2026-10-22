# Bezug der ursprünglichen Modellgewichte

Stand der Metadatenprüfung: 23.10.2026. Es wurden ausschließlich öffentliche Dateilisten und Hash-Metadaten gelesen, keine Gewichte heruntergeladen und keine Modellinferenz ausgeführt. Originalbelege: [Inventar](MODEL_INVENTORY.json), [Ollama-Metadaten](model_metadata/) und [Rollendigests](models.json).

## Nachgewiesene Bezugswege und Grenzen

| Rolle | Dokumentierter Bezug | Befund zur exakten Fassung |
|---|---|---|
| Summary und Demonstrationsentwicklung | [unsloth/Seed-OSS-36B-Instruct-GGUF auf ModelScope](https://modelscope.cn/models/unsloth/Seed-OSS-36B-Instruct-GGUF), Datei `Seed-OSS-36B-Instruct-Q4_K_M.gguf`, Dateirevision `9c0c6176ae95e5f59e49d304ba3f3fe51a311c1c` | Die öffentliche Dateiliste nennt denselben SHA-256 und dieselbe Größe wie das ursprüngliche Weight-Blob. Die tatsächlichen 21,8 GB wurden nicht heruntergeladen oder lokal geprüft. |
| QA | [ggml-org/Qwen3.8-27B-GGUF auf Hugging Face](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF), historische Kennung `hf.co/ggml-org/Qwen3.8-27B-GGUF:Q4_K_M` | Die aktuelle Dateiliste der Revision `a316ea527fde32c12487be7574e2db5ac2ad2a7e` enthält den ursprünglichen SHA-256 des Hauptgewichts **nicht**. Ein exakter älterer Snapshot muss separat beschafft werden; sein Bezugsweg ist nicht abschließend nachgewiesen. Der ursprüngliche Projektor ist in dieser Revision noch hashgleich als `mmproj-Qwen3.8-27B-Q8_0.gguf` vorhanden. |
| Fakten/Fragen | Das archivierte Modellmanifest belegt Gemma4, 31,3B, Q4_K_M und den Originaldigest. | Ein eindeutiger öffentlicher Bezugsweg mit exakter Revision ist nicht dokumentiert. Das ursprüngliche Blob muss aus einer vollständigen Originalkopie bzw. vom Autor beschafft werden. Ein aktuelles Gemma-Modell gilt nicht als gleichwertiger Ersatz. |

Nachprüfbare Metadatenquellen: [Seed-Dateiliste](https://modelscope.cn/api/v1/models/unsloth/Seed-OSS-36B-Instruct-GGUF/repo/files?Revision=master&Recursive=true), [QA-Dateiliste mit LFS-Hashes](https://huggingface.co/api/models/ggml-org/Qwen3.8-27B-GGUF?blobs=true). Die Links mit `master` bzw. ohne Revision sind veränderlich; für den Originalbezug sind die oben genannten Hashes/Revisionen maßgeblich. Eine fehlende Übereinstimmung im aktuellen Stand beweist nicht, dass sämtliche historischen Snapshots gelöscht wurden.

## Exakte fehlende Blobs

| Dateizweck | SHA-256 des ursprünglichen Blobs | Bytes |
|---|---|---:|
| Seed-GGUF | `2d9f3c32a61f2668e012708be7ec04e753e35e366f87b9e4bce89f4d073cfc16` | 21762149536 |
| Gemma-GGUF | `280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313` | 19868969920 |
| QA-GGUF | `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` | 18973870432 |
| QA-Projektor | `2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb` | 629247008 |

## Manuelle Bereitstellung für den optionalen Experimentweg

1. Die Originalblobs über die genannten Quellen oder ein vollständiges Originalarchiv beschaffen. Jeden heruntergeladenen Blob mit `sha256sum DATEI` gegen die Tabelle prüfen. Ein gleicher Modellname bzw. dieselbe Quantisierung reicht nicht.
2. Die kleinen, mitgelieferten Dateien unter `data/model_metadata/` enthalten sämtliche weiteren in den drei Originalmanifesten referenzierten Config-, Template-, Parameter- und Lizenzblobs. In einer **eigenen** Ollama-Modellablage müssen diese Dateien zusammen mit den vier oben genannten Blobs vollständig vorliegen. Die Originaldateinamen der Blobs sind `sha256-<Hash>`; die Manifestpfade entsprechen `manifests/thesis-seed/v1`, `manifests/thesis-gemma/v1` und `manifests/thesis-qwen38-qa/v1`.
3. Für die Ablage kann die dokumentierte Ollama-Variable `OLLAMA_MODELS` auf ein ausdrücklich gewähltes eigenes Verzeichnis gesetzt werden. Die Modelllaufzeit muss die ursprünglichen Renderer/Parser unterstützen; die archivierten Metadaten nennen unter anderem Ollama `0.20.0` als Voraussetzung. Ein späterer Laufzeitstand wurde hier nicht auf identische Inferenz getestet. Keine ungeprüfte Template-Neuerstellung oder automatische Ersatzversion verwenden.
4. Vor einem ausdrücklich aktivierten Experiment kontrolliert die Pipeline die vollständigen Ollama-Manifestdigests über `/api/tags`. Fehlende oder andere Digests führen vor der ersten Generierung zum Abbruch. Digestgleichheit allein garantiert keine bitidentische Inferenz auf anderer Hardware.

**Einschränkung:** Die vollständige Neuauflage mit allen drei exakt ursprünglichen LLMs ist derzeit nicht für Dritte lückenlos beschaffbar nachgewiesen. Der Offlineweg mit den eingefrorenen Ergebnissen ist davon unabhängig. Es wird kein Ersatzmodell verwendet und keine exakte Experimentreproduktion mit anderen Gewichten behauptet.
