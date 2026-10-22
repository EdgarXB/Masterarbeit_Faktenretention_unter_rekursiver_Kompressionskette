# Manueller, unveränderlicher GitHub-Release

Es wurde durch diese Überarbeitung **kein Release und kein Release-Tag erstellt**. Vorgeschlagener erster Tag: `v1.0.0`. Erst nach Prüfung und Annahme des Commits/PR veröffentlichen. Die konkrete Commit-SHA im Release und in einer Zitation festhalten.

## Vor Veröffentlichung prüfen

1. Den angenommenen Commit frisch unter Linux klonen und die in der README dokumentierte Python-3.13.15-Umgebung einrichten. Der Clone darf keine ursprünglichen Entwicklungsordner benötigen. Unter WSL2 ebenfalls ausschließlich im Linux-Terminal arbeiten.
2. `git status --short` muss vor und nach den reinen Tests leer sein. Ausführen:

   ```bash
   .venv/bin/python 05_analyze.py --check-environment
   .venv/bin/python -m unittest discover -s data/tests -p 'test_*.py'
   sha256sum --check data/frozen_assets.sha256
   ```

   `01_prepare.py` prüft zusätzlich alle 6.288 logischen Originalquellen gegen deren dokumentierte SHA-256-Werte. Es schreibt einen Prüfbericht; bei dessen Aufnahme in den Release muss er vor dem finalen Commit geprüft werden. Keine LLM-Tests, Demonstrationskonstruktion oder Regression-Bootstrap-Neuberechnung ausführen.
3. Auf GitHub unter **Settings → General → Releases → Enable release immutability** die Unveränderlichkeit **vor der Veröffentlichung** einschalten. Sie gilt nur für zukünftige Releases. [GitHub-Anleitung](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/establish-provenance-and-integrity/prevent-release-changes)
4. Die exakte angenommene Commit-SHA und einen noch unbenutzten Versionsnamen wählen. Nach einem weiteren Commit erneut testen. Keinen bestehenden Release-Tag verschieben oder ersetzen.

## Tag und prüfbare Assets manuell vorbereiten

Die folgenden Befehle sind eine Anleitung und werden von keinem Skript automatisch gestartet. Im sauberen Repository auf dem geprüften Commit ausführen:

```bash
release_commit=$(git rev-parse HEAD)
git tag -a v1.0.0 "$release_commit" -m 'Geprüfter digitaler Masterarbeitsanhang'
git push origin refs/tags/v1.0.0
git archive --format=zip --prefix=Masterarbeit_Digitaler_Anhang/ v1.0.0 > ../Masterarbeit_Digitaler_Anhang-v1.0.0.zip
sha256sum ../Masterarbeit_Digitaler_Anhang-v1.0.0.zip > ../Masterarbeit_Digitaler_Anhang-v1.0.0.zip.sha256
```

Die ZIP enthält ausschließlich den getaggten Git-Bestand. Zusätzlich die tatsächlich verwendete `.venv/bin/python --version`, `requirements-analysis.lock.txt`, `data/analysis_environment.json`, die Commit-SHA und den Audit-Prüfbericht in den Release-Notizen nennen. `manifest.json` und `data/frozen_assets.sha256` liegen bereits in der ZIP. Die externe ZIP-Prüfsumme liegt bewusst außerhalb der ZIP, um eine Selbstreferenz zu vermeiden.

## Release ausschließlich manuell veröffentlichen

Auf GitHub **Releases → Draft a new release** öffnen, `v1.0.0` auswählen, die ZIP und ihre SHA-256-Datei anhängen und Notizen prüfen. Erst danach **Publish release** betätigen. GitHub sperrt beim unveränderlichen Release den Tag und die Assets; Titel und Beschreibung bleiben bearbeitbar. Auf der fertigen Seite muss **Immutable** angezeigt werden. [GitHub-Dokumentation](https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases)

Falls die Unveränderlichkeitsoption für das Repository nicht verfügbar oder nicht eingeschaltet ist, darf der Release nicht als technisch unveränderlich bezeichnet werden. Dann zunächst nur den geprüften Commit samt SHA als vorbereiteten Stand dokumentieren. Korrekturen nach Veröffentlichung benötigen eine neue Versionsnummer und eine neue Abnahme.
