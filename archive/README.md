# Historische Quellen

`baseline_ad7eb1a/` bewahrt den vor dieser Umstrukturierung aktiven Code und die
betroffenen Dokumentations-/Inventarstände bytegetreu aus Commit
`ad7eb1a86039dbc44651e4e2cdb1dc358c3d0f14`. Das ist Herkunftsevidenz und kein
unterstützter Einstieg. Insbesondere Pilotparameter und Längenrevisionen in
`iterative_summarize.py` gehören nicht zur aktuellen Experimentsteuerung.

Die vollständigen älteren Pilot-, Debug-, Rerun- und Merge-Quellen liegen weiterhin
unverändert in `data/eingefrorene_quellen.zip`. Die Referenznotebooks bleiben in
`data/referenznotebooks.zip`. Historische relative Links und damalige Laufberichte
beschreiben den damaligen Verzeichnisbaum; aktuelle Pfade stehen in
[`data/refactor_paths.json`](../data/refactor_paths.json).

Originalprüfsummen bleiben unverändert. Prüfung einschließlich absichtlich
archivierter Codepfade: `python -m faktenretention.frozen_data --checksums`.
