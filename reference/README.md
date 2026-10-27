# Eingefrorene Referenzen

Die Originalartefakte bleiben an ihren inventarisierten Pfaden, damit Prüfsummen und
Quellenzuordnungen erhalten bleiben:

- [`results/originaltabellen.zip`](../results/originaltabellen.zip)
- [`figures/originalabbildungen.zip`](../figures/originalabbildungen.zip)
- [`data/referenznotebooks.zip`](../data/referenznotebooks.zip)
- [`data/quelleninventar.json`](../data/quelleninventar.json)

Der Runner liest die Referenz-ZIPs direkt; neue Tabellen und Grafiken entstehen
unter `LAUF/tables/` und `LAUF/figures/`. Referenzartefakte werden nie als neu
berechnete Ausgaben ausgegeben. Ihre erwarteten neuen Gegenstücke und ausdrücklich
wiederverwendete Eingaben stehen im [Notebook-Inventar](../notebooks/analyses.json).
