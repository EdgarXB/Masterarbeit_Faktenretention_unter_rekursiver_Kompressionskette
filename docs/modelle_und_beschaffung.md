# Modelle und Beschaffung

Diese Anleitung rekonstruiert die Modellzuordnung aus den ursprünglichen
Experimentdateien und lokalen Ollama-Manifesten. Die Belege sind nach
[`data/model_metadata/provenance/`](../data/model_metadata/provenance/) kopiert;
[`sources.json`](../data/model_metadata/provenance/sources.json) nennt Originalpfade
und SHA-256. Die Experimenteingaben und eingefrorenen Manifeste bleiben unverändert.
Die Gewichte werden extern beschafft. Öffentliche Metadaten und die kleine
SummaC-Checkpointdatei wurden am **29. Oktober 2026** geprüft; große Gewichte
wurden dafür nicht erneut heruntergeladen.

## Die drei LLMs

| Rolle | Erwarteter lokaler Name | Ursprüngliche Quelle | Variante |
|---|---|---|---|
| Zusammenfassungen und CoT-Kandidaten | `thesis-seed:v1` | [`milkey/Seed-OSS-36B-Instruct:q4_K_M`](https://ollama.com/milkey/Seed-OSS-36B-Instruct:q4_K_M) | Seed-OSS 36.2B, GGUF Q4_K_M |
| Kanonische Fakten, Fragen, Referenzantworten | `thesis-gemma:v1` | [`gemma4:31b`](https://ollama.com/library/gemma4:31b) | Gemma 4 31.3B, GGUF Q4_K_M, historischer Stand |
| Antworten auf kanonische Fragen aus Summaries | `thesis-qwen38-qa:v1` | lokaler Name `qwen3.8:27b-q4_K_M`, aufgebaut aus [`ggml-org/Qwen3.8-27B-GGUF`](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) | 26.9B, GGUF Q4_K_M plus Q8_0-Projektor |

Der [Freeze vom 20. August](../data/model_metadata/provenance/model_freeze_20260820.json)
belegt Seed und Gemma. Die [Änderung vom 1. September](../data/model_metadata/provenance/qa_amendment_20260901.json)
ersetzt ausschließlich den früheren GLM-Answerer durch Qwen. Die alten GLM-Angaben
und Pilotparameter im ersten Freeze sind deshalb keine Anleitung für den finalen
QA-Lauf oder die späteren Summaryparameter; dafür gelten Pipeline und Laufprotokolle.

Die [Quellmanifeste für Seed](../data/model_metadata/provenance/source_manifest_seed.json),
[Gemma](../data/model_metadata/provenance/source_manifest_gemma.json) und
[Qwen](../data/model_metadata/provenance/source_manifest_qa.json) stimmen bytegenau
mit den jeweiligen eingefrorenen `thesis-*`-Manifesten überein. Damit sind die
Quellnamen über die vollständigen Manifest-Digests zugeordnet.

### Qwen: Modellname und Architekturbezeichnung

Das [originale Modelfile](../data/model_metadata/provenance/Modelfile.qwen38-27b-q4km)
enthält gemeinsam:

```text
FROM hf.co/ggml-org/Qwen3.8-27B-GGUF:Q4_K_M
TEMPLATE {{ .Prompt }}
RENDERER qwen3.5
PARSER qwen3.5
PARAMETER presence_penalty 1.5
PARAMETER temperature 1
PARAMETER top_k 20
PARAMETER top_p 0.95
```

`qwen35`/`qwen3.5` sind also bereits im Original die Architektur-/Rendererangaben
des unter dem Namen Qwen3.8 bezogenen Artefakts. Ein Download von `qwen3.5:27b`
würde ein anderes Gewicht liefern. Die Laufparameter, insbesondere QA-Temperatur
0 und Seed 32, setzt die Pipeline zusätzlich zu diesen Modelldefaults.

### Exakte Gewichtsquellen

Die vollständigen erwarteten Datei-Hashes stehen im unveränderten
[`MODEL_INVENTORY.json`](../data/MODEL_INVENTORY.json). Hashes von Gewichtsdateien
und Hashes kompletter Ollama-Manifeste sind unterschiedliche Prüfwerte.

| Datei | Exakte öffentliche Quelle | Erwarteter SHA-256 |
|---|---|---|
| Seed-Gewicht | [Ollama-Registry-Blob](https://registry.ollama.ai/v2/milkey/Seed-OSS-36B-Instruct/blobs/sha256:2d9f3c32a61f2668e012708be7ec04e753e35e366f87b9e4bce89f4d073cfc16) | `2d9f3c32a61f2668e012708be7ec04e753e35e366f87b9e4bce89f4d073cfc16` |
| Historisches Gemma-Gewicht | [Ollama-Registry-Blob](https://registry.ollama.ai/v2/library/gemma4/blobs/sha256:280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313) | `280af6832eca23cb322c4dcc65edfea98a21b8f8ab07dc7553bd6f7e6e7a3313` |
| QA-Textgewicht | [Qwen3.8-27B-Q4_K_M.gguf, feste Revision](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/0669b98607d47046c7c2b3f801011d54a08cfccf/Qwen3.8-27B-Q4_K_M.gguf) | `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| QA-Projektor | [mmproj-Qwen3.8-27B-Q8_0.gguf, feste Revision](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/0669b98607d47046c7c2b3f801011d54a08cfccf/mmproj-Qwen3.8-27B-Q8_0.gguf) | `2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb` |

Die historische Qwen-Revision wurde anhand der veröffentlichten LFS-SHA-256 und
Dateigrößen zugeordnet. Der aktuelle `main`-Stand hat inzwischen ein anderes
Textgewicht. Ebenso liefert der heutige Gemma-Tag einen anderen Manifest- und
Gewichtsstand. `ollama pull gemma4:31b` oder das unveränderte QA-Modelfile mit
einem heutigen `main`-Download reichen deshalb für die exakte Rekonstruktion
nicht aus. Der historische Gemma-Manifestabruf nach Digest ergab HTTP 404, der
Abruf seines Gewichtsblobs per HEAD dagegen HTTP 200. Das Manifest und seine
kleinen Metadaten liegen bereits im Repository.

[`public_checks_20261029.json`](../data/model_metadata/provenance/public_checks_20261029.json)
enthält die Prüfung der öffentlichen Quellen. Die dortige Gewichtszuordnung
beruht auf Remote-Metadaten; nach jedem tatsächlichen Download muss der Inhalt
zusätzlich gegen den angegebenen SHA-256 geprüft werden.

### Lokale Ollama-Namen mit den gespeicherten Manifesten einrichten

Die Quellen wurden ursprünglich auf lokale Namen eingefroren. Für Seed ist
`ollama pull milkey/Seed-OSS-36B-Instruct:q4_K_M` gefolgt von
`ollama cp milkey/Seed-OSS-36B-Instruct:q4_K_M thesis-seed:v1` der direkte Weg,
sofern der Manifest-Digest weiterhin übereinstimmt. Bei Gemma und Qwen lässt
sich der genaue Stand mit den gespeicherten Manifesten rekonstruieren:

1. Die vier Dateien aus der Tabelle nach `results/model-resources/ollama/blobs/`
   herunterladen. Jede Datei dort als `sha256-<vollständiger Datei-Hash>` speichern.
2. Vor dem Start eines Servers für dieses neue Modellverzeichnis das folgende
   lokale Skript im Repository ausführen. Es prüft alle vier Downloads und die
   kleinen Metadaten und installiert die drei eingefrorenen Manifeste. Es lädt
   nichts herunter und überschreibt bei abweichendem Inhalt keine Dateien.

```python
from pathlib import Path
import hashlib
import json
import shutil

metadata = Path('data/model_metadata')
store = Path('results/model-resources/ollama')
models = json.loads(Path('data/models.json').read_text())

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

copies = {}
for model in models.values():
    name, tag = model['name'].split(':')
    manifest = metadata / 'manifests' / name / tag
    assert digest(manifest) == model['digest'], manifest
    info = json.loads(manifest.read_text())
    for layer in [info['config'], *info['layers']]:
        filename = layer['digest'].replace(':', '-')
        target = store / 'blobs' / filename
        source = metadata / 'blobs' / filename
        if not source.is_file():
            source = target  # Die separat heruntergeladenen Gewichte.
        assert source.stat().st_size == layer['size'], source
        assert digest(source) == layer['digest'].split(':', 1)[1], source
        if source != target:
            copies[target] = source
    copies[store / 'manifests/registry.ollama.ai/library' / name / tag] = manifest

for target, source in copies.items():
    if target.exists():
        assert target.read_bytes() == source.read_bytes(), target
for target, source in copies.items():
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
print('Drei eingefrorene Modelle für Ollama eingerichtet:', store.resolve())
```

Die Wiederherstellung nutzt Ollamas lokales Blob-/Manifestformat, wie es in den
Originaldateien vorliegt. Ein Import mit `ollama create` unter einer anderen
Ollama-Version kann neue Konfigurations- und Manifestbytes erzeugen. Maßgeblich
bleibt der von `/api/tags` gemeldete Digest; die Pipeline lehnt Abweichungen ab.

## Ollama-Laufzeit

Der originale [Modell-Freeze](../data/model_metadata/provenance/model_freeze_20260820.json)
und die [Umgebungsaufzeichnung](../data/model_metadata/provenance/environment_20260824.json)
nennen **Ollama 0.32.6**. `requires: 0.20.0` in der Gemma-Konfiguration ist eine
Mindestanforderung. Bezug der Laufzeit: [Ollama-Releases](https://github.com/ollama/ollama/releases/tag/v0.32.6).
Die ursprünglichen Servereinstellungen stehen im
[Auszug des Forschungslogs](../data/model_metadata/provenance/runtime_excerpt.txt).
Nach Installation der passenden Laufzeit kann der separate Modellspeicher so
verwendet werden:

```bash
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_MODELS="$PWD/results/model-resources/ollama"
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
ollama serve
```

Das Experiment fordert für Seed `num_ctx=524288`; die Originalaufzeichnung
beschreibt einen Rechner mit etwa 98 GB GPU-Speicher und 121 GiB RAM. Diese
Beobachtung ist keine ermittelte Mindestanforderung für andere Rechner.

In einem zweiten Terminal lässt sich der Modellstand ohne Generierung prüfen:

```python
from pathlib import Path
import json
import urllib.request

with urllib.request.urlopen('http://127.0.0.1:11435/api/tags', timeout=10) as response:
    installed = {m['name']: m['digest'] for m in json.load(response)['models']}
for model in json.loads(Path('data/models.json').read_text()).values():
    assert installed.get(model['name']) == model['digest'], model['name']
    print('OK', model['name'], model['digest'])
```

## Embedding, BERTScore, SummaC und Tokenizerdaten

| Ressourcenschlüssel | Quelle | Festgelegte Revision bzw. Prüfsumme |
|---|---|---|
| `embedding` | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` |
| `bertscore` | [`distilroberta-base`](https://huggingface.co/distilroberta-base) | `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b` |
| `summac_nli` | [`tals/albert-xlarge-vitaminc-mnli`](https://huggingface.co/tals/albert-xlarge-vitaminc-mnli) | `3082ba54344bd9ddada2be1c5e9b4131721d2a5d` |
| `summac_head` | [`summac_conv_vitc_sent_perc_e.bin`](https://github.com/tingofurro/summac/blob/master/summac_conv_vitc_sent_perc_e.bin) | SHA-256 `cd880581f59bd47b8968c7f48d754cd617dfe9b256d8d76fdcbc0fa4a3b3c6fe` |
| `nltk_data` | NLTK-Downloader, Paket `punkt_tab` | englische Dateien unter `tokenizers/punkt_tab/english/` |

Modellnamen/Revisionen sind durch das ursprüngliche Inventar und die Originalcaches
belegt. Der 1.811 Byte große SummaC-Checkpoint stimmt lokal und als öffentlicher
Download bytegenau mit dem erwarteten Hash überein. Die URL findet sich auch im
[Code der ursprünglich installierten SummaC-Bibliothek](../data/model_metadata/provenance/summac_download_excerpt.txt).
Die historische Revision
der NLTK-Datendateien ist nicht separat versioniert dokumentiert; der Code fordert
ausdrücklich die englischen `punkt_tab`-Dateien an.

Nach Einrichtung der Experimentumgebung beschafft der folgende ausdrücklich
ausgeführte Python-Block die drei Modellrevisionen, den Checkpoint und die
Tokenizerdaten. Er schreibt eine direkt für `--resources` verwendbare JSON-Datei:

```python
from pathlib import Path
import hashlib
import json
import urllib.request
import nltk
from huggingface_hub import snapshot_download
from faktenretention.metric_resources import REVISIONS, HEAD_SHA256

root = Path('results/model-resources').resolve()
root.mkdir(parents=True, exist_ok=True)
repos = {
    'embedding': 'sentence-transformers/all-MiniLM-L6-v2',
    'bertscore': 'distilroberta-base',
    'summac_nli': 'tals/albert-xlarge-vitaminc-mnli',
}
resources = {}
for role, repo in repos.items():
    relative = Path(role) / REVISIONS[role]
    snapshot_download(
        repo_id=repo, revision=REVISIONS[role], local_dir=root / relative,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '1_Pooling/*'],
    )
    resources[role] = str(relative)

filename = 'summac_conv_vitc_sent_perc_e.bin'
url = 'https://raw.githubusercontent.com/tingofurro/summac/master/' + filename
with urllib.request.urlopen(url, timeout=60) as response:
    checkpoint = response.read()
assert hashlib.sha256(checkpoint).hexdigest() == HEAD_SHA256
(root / filename).write_bytes(checkpoint)
resources['summac_head'] = filename
nltk.download('punkt_tab', download_dir=str(root / 'nltk_data'), raise_on_error=True)
resources['nltk_data'] = 'nltk_data'
(root / 'resources.json').write_text(json.dumps(resources, indent=2) + '\n')
print(root / 'resources.json')
```

Die `*.txt`-Auswahl schließt auch `merges.txt` und `vocab.txt` ein. Die Verzeichnisse
heißen nach ihrer Revision und werden von `metric_resources.local_model` akzeptiert.
Die Dateipfade in `resources.json` werden relativ zu dieser JSON-Datei aufgelöst.
Bei späteren Experimentaufrufen angeben:
`--resources results/model-resources/resources.json`.
Die Pipeline selbst behält ihre Offline-Schalter und lädt keine Modelle implizit.

## Paketkonflikt und überprüfte Grenzen

Der historische Metrikstand enthält gleichzeitig `summac==0.0.4`,
`transformers==5.15.1` und eine neuere `huggingface-hub`-Version. Die deklarierten
Abhängigkeiten widersprechen sich: SummaC verlangt `huggingface-hub<=0.17.0`,
Transformers verlangt `huggingface-hub>=1.5.0,<2.0`. Die vorhandene Originalumgebung
ist also kein Beleg für einen mit einem Resolver sauber installierbaren Paketstand.

Die [CPU-Probe](../data/model_metadata/provenance/summac_cpu_check_20261029.json)
der ursprünglichen lokalen Metrikumgebung verwendete Python 3.13.15, Torch 2.13.0,
Transformers 5.15.1, Hugging-Face-Hub 1.28.0, NLTK 3.9.4 und SummaC 0.0.4 geprüft.
Ein CPU-Aufruf des aktuellen Refactor-Codes mit den ursprünglichen lokalen NLI-,
Checkpoint- und Tokenizerdateien lieferte für den identischen Satz
`Alice founded the company in London in 2010.` den endlichen Wert
`0.20476575195789337`. Die Bibliothek meldete beim Laden des Heads
`All keys matched successfully`. Der Code enthält dafür bereits die Kompatibilität
mit der entfernten `batch_encode_plus`-Schnittstelle.

`uv pip install --python .venv-experiment/bin/python --no-deps summac==0.0.4`
wäre eine bewusste Umgehung der SummaC-Abhängigkeitsprüfung nach Installation von
`requirements-experiment.txt`. Dieser Weg wurde hier nicht als frische vollständige
Installation getestet und ist kein als kompatibel aufgelöster Paketstand. Eine
unabhängig installierbare Gesamtumgebung bleibt gesondert zu prüfen. Die genannte
CPU-Probe validiert weder GPU-Ausführung noch alle Metrikfälle oder einen gesamten
LLM-Experimentlauf.
