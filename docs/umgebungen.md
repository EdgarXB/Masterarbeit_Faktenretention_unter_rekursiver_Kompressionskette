# Getrennte Umgebungen

Der unveränderte historische Analyse-Lock ist `requirements-analysis.lock.txt` mit
CPython 3.13.15. `data/analysis_environment.json` hält die notwendigen Versionen für
die Regressionscaches. Ein passender Versionsstand ersetzt deren vollständigen
Fingerprint aus Daten, Designmatrix, Punktkoeffizienten, Worker und Software nicht.
Die aktive historische Analyse verwendet ausschließlich vorhandene passende Caches.

## Unterstützter Experiment-Lock ohne Modellmetriken

`requirements-experiment.in` und der daraus aufgelöste
`requirements-experiment.lock.txt` gelten für Linux x86_64/Python 3.13 und den Modus
`--without-model-metrics`. Alle übernommenen historischen Paketpins bleiben erhalten.
SummaC ist ausdrücklich nicht enthalten. Dieser Lock unterstützt deshalb weder
SummaC-/BERTScore-Gesamtmetriken noch die neue Demonstrationskonstruktion mit
SummaC-Pruning. Eingefrorene Demonstrationen bleiben nutzbar.

```bash
uv venv --python 3.13.15 .venv-experiment
uv pip sync --python .venv-experiment/bin/python requirements-experiment.lock.txt
uv pip install --python .venv-experiment/bin/python --no-deps -e .
```

Der Lock wurde anhand von Paketmetadaten aufgelöst. Die Modellpakete wurden bei
dieser Umstrukturierung nicht installiert; GPU/CUDA, Ollama, Gewichte und echte
Modellantworten wurden nicht getestet. Die Phasenverbindungen wurden mit simulierten
Antworten geprüft. Die aktuelle wissenschaftliche Auswertung bleibt in der separaten
Analyseumgebung, insbesondere für historische Regressionscaches.

## Konkrete Grenze des historischen Vollstands

`requirements_experiment.txt` bleibt als beobachteter historischer Stand unverändert.
Seine gleichzeitigen Pins sind mit heutigen Resolvern nicht konsistent auflösbar:
`summac==0.0.4` verlangt `huggingface-hub<=0.17.0`, während
`transformers==5.15.1` `huggingface-hub>=1.5.0,<2.0` verlangt. Der konkrete
[Resolvernachweis](experiment-resolution-full.txt) ist gespeichert.

`requirements-experiment-full.in` dokumentiert diese zusätzliche, derzeit
unauflösbare Anforderung. Reproduzierbarer Diagnosebefehl:

```bash
uv pip compile requirements-experiment-full.in --python-version 3.13 --python-platform x86_64-unknown-linux-gnu
```

Es gibt keinen als funktionsfähig ausgegebenen Voll-Lock und keine automatische
Paketabwertung oder Installation mit `--no-deps` zum Umgehen des Konflikts. Eine
SummaC-Portierung oder eine geänderte Modellumgebung müsste separat fachlich und
technisch geprüft werden. Weitere Beschaffungsgrenzen der Originalgewichte stehen
in [`data/MODELLBEZUG.md`](../data/MODELLBEZUG.md).
