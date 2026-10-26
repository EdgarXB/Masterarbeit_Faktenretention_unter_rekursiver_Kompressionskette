# Eingefrorene Referenzen

Die Originalartefakte bleiben an ihren inventarisierten Pfaden, damit Prüfsummen und
Quellenzuordnungen erhalten bleiben:

- [`results/originaltabellen.zip`](../results/originaltabellen.zip)
- [`figures/originalabbildungen.zip`](../figures/originalabbildungen.zip)
- [`data/referenznotebooks.zip`](../data/referenznotebooks.zip)
- [`data/quelleninventar.json`](../data/quelleninventar.json)

Der Runner entpackt Referenzen getrennt unter `LAUF/reference/`; neue Tabellen und
Grafiken entstehen unter `LAUF/workspace/`. Referenzartefakte werden nie als neu
berechnete Ausgaben ausgegeben. Ihre erwarteten neuen Gegenstücke und ausdrücklich
wiederverwendete Eingaben stehen im [Notebook-Inventar](../notebooks/analyses.json).
