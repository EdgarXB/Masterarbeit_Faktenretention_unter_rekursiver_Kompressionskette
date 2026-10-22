# Verwendete finale Prompts

`auto_cot_*.txt` und `automate_cot_*.txt` sind die unveränderten, stufenweisen finalen Prompts. Ihre Hashes stehen im originalen `manifest.json`.

Für die **aktuelle** Few-Shot-Bedingung gelten ausschließlich `final_few_shot_cot_*.txt` und `final_few_shot_cot_manifest.json`. Die ältere Few-Shot-Sektion des historischen `manifest.json` dokumentiert den ursprünglichen Freeze; ihre Texte liegen nur im Quellenarchiv und werden im Mini-Test nicht verwendet.

Der finale Plain-Text lautet `summarize in {target_word_length}`, gefolgt von einer Leerzeile und dem jeweiligen Eingabetext. Die unveränderte direkte Zero-Shot-CoT-Vorlage und der Seed-Thinking-Wrapper stehen als kurze Funktionen in `03_generate.py`. Die konkreten Modelleinstellungen, archivierten historischen Wrapper und Promptherkunft stehen im Quellenarchiv.
