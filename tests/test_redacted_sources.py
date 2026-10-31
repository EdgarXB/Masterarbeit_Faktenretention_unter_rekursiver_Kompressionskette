"""The shared analysis inputs must not contain the removed source articles."""
import csv
import gzip
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REDACTED = '[ORIGINALTEXT ENTFERNT]'


class RedactedSourcesTest(unittest.TestCase):
    def test_article_and_highlight_columns_are_redacted(self):
        for name in ('artikel.csv.gz', 'zusammenfassungen.csv.gz'):
            with self.subTest(name=name), gzip.open(ROOT / 'data' / name, 'rt', encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
                self.assertTrue(rows)
                self.assertTrue(all(row['article'] == REDACTED and row['highlights'] == REDACTED for row in rows))

    def test_initial_generation_inputs_are_redacted(self):
        with gzip.open(ROOT / 'data/generierungszustaende.jsonl.gz', 'rt', encoding='utf-8') as f:
            rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 600)
        self.assertTrue(all(row['state']['stages'][0]['input_text'] == REDACTED for row in rows))


if __name__ == '__main__':
    unittest.main()
