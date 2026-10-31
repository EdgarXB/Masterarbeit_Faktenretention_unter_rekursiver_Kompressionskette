"""Tests für Auswahl und nachträgliche Modellmetriken ohne Netz oder Modellinferenz."""
import csv
import importlib.util
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('select_articles', ROOT / 'scripts/select_articles.py')
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


class SelectionTests(unittest.TestCase):
    def test_seed_reproducible_exclusion_and_order(self):
        import hashlib
        rows = [{'article': ' '.join([str(i)] + ['word'] * 799), 'highlights': 'Reference.'} for i in range(30)]
        excluded = {hashlib.sha256(rows[0]['article'].encode()).hexdigest()}
        first, eligible = selection.select(iter(rows), 6, 32, excluded)
        again, _ = selection.select(iter(rows), 6, 32, excluded)
        other, _ = selection.select(iter(rows), 6, 33, excluded)
        indices = [row['dataset_index'] for row in first]
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        self.assertEqual(eligible, 29)
        self.assertNotIn(0, indices)
        self.assertEqual(indices, sorted(indices))

    def test_insufficient_eligible_rows_fails(self):
        with self.assertRaisesRegex(ValueError, 'Nur 0'):
            selection.select([{'article': 'short', 'highlights': 'Reference.'}], 1, 32, set())


class EvaluateTests(unittest.TestCase):
    def test_metrics_can_use_new_resources_without_ollama(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'; run.mkdir()
            (run / 'execution.json').write_text(json.dumps({'resources': {}, 'stages': [500]}))
            resources = Path(tmp) / 'resources.json'
            resources.write_text(json.dumps({'bertscore': 'model'}))
            argv = ['04_evaluate.py', '--run', str(run), '--model-metrics', '--execute',
                    '--resources', str(resources), '--summac-device', 'cuda']
            with patch.object(sys, 'argv', argv), \
                 patch('faktenretention.request_budget.activate', side_effect=AssertionError('Ollama not needed')), \
                 patch('faktenretention.pipeline.compute_metrics') as compute:
                runpy.run_path(str(ROOT / '04_evaluate.py'), run_name='__main__')
            args, kwargs = compute.call_args
            self.assertEqual(args[1]['resources']['bertscore'], str(Path(tmp) / 'model'))
            self.assertEqual(kwargs, {'semantic': True, 'summac_device': 'cuda'})
            self.assertEqual(json.loads((run / 'execution.json').read_text())['resources'], {})


if __name__ == '__main__':
    unittest.main()
