"""Prüft den Vergleich auf fachliche Änderungen und fehlende Tabellen."""
from pathlib import Path
import tempfile
import unittest

from scripts.compare_analysis_results import compare


class ResultComparisonTest(unittest.TestCase):
    def test_scientific_change_fails_but_runtime_change_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = root / 'reference', root / 'generated'
            relative = Path('06_laengenregression/C_modelldiagnostik.csv')
            for directory in (left, right):
                (directory / relative).parent.mkdir(parents=True)
            (left / relative).write_text('value,fit_seconds,covariance_rank,n_parameters\n0.5,1,23,123\n')
            (right / relative).write_text('value,fit_seconds,covariance_rank,n_parameters\n0.5,2,24,123\n')
            self.assertEqual(compare(left, right), (1, []))
            (right / relative).write_text('value,fit_seconds,covariance_rank,n_parameters\n0.6,2,24,123\n')
            self.assertEqual(len(compare(left, right)[1]), 1)
            (right / relative).write_text('value,fit_seconds,covariance_rank,n_parameters\n0.5,2,25,123\n')
            self.assertEqual(len(compare(left, right)[1]), 1)
            (right / relative).write_text('value,fit_seconds,covariance_rank,n_parameters\n0.5,2,23.5,123\n')
            self.assertEqual(len(compare(left, right)[1]), 1)

    def test_missing_output_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = root / 'reference', root / 'generated'
            left.mkdir(); right.mkdir()
            (left / 'result.csv').write_text('value\n1\n')
            self.assertIn('fehlt', compare(left, right)[1][0])


if __name__ == '__main__':
    unittest.main()
