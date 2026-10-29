"""Numerische Rundung akzeptieren, veränderte Cache-Grundlagen ablehnen."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from faktenretention.bootstrap_cache import load_bootstrap_cache


class BootstrapCacheTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'bootstrap_linear.npz'
        self.inputs = dict(X=np.array([[1., 0.], [1., 1.]]),
                          Y=np.array([[3., 2.], [2., 3.]]), beta=np.array([0.1, -0.3]),
                          bootstrap_counts=np.array([[2, 0], [1, 1]], dtype=np.int16),
                          worker_source='original worker\n', software={'python': '3.13.15'})
        a = self.inputs
        self.fingerprint = hashlib.sha256(
            a['X'].tobytes() + a['Y'].tobytes() + a['beta'].tobytes()
            + a['bootstrap_counts'].tobytes() + a['worker_source'].encode()
            + json.dumps(a['software'], sort_keys=True).encode()).hexdigest()
        self.coefficients = np.array([[0.2, -0.4], [0.1, -0.3]])
        self.records = [[True, 3, ''], [True, 3, '']]
        np.savez_compressed(self.path, coefficients=self.coefficients,
                            records=json.dumps(self.records), fingerprint=self.fingerprint,
                            bootstrap_counts=a['bootstrap_counts'])
        self.manifest = dict(schema_version=1,
                             cache_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                             fingerprint=self.fingerprint, software=a['software'],
                             reference_beta=a['beta'].tolist())
        self.save_manifest()

    def save_manifest(self):
        self.path.with_suffix('.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_rounding_does_not_invalidate_original_bootstrap_results(self):
        for beta in [self.inputs['beta'], np.nextafter(self.inputs['beta'], np.inf),
                     self.inputs['beta'] + 1e-12]:
            with self.subTest(beta=beta):
                coefficients, records, fingerprint = load_bootstrap_cache(
                    self.path, **{**self.inputs, 'beta': beta})
                np.testing.assert_array_equal(coefficients, self.coefficients)
                self.assertEqual(records, self.records)
                self.assertEqual(fingerprint, self.fingerprint)

    def test_changed_model_inputs_worker_or_versions_are_rejected(self):
        changes = dict(X=self.inputs['X'] + 1e-12, Y=self.inputs['Y'] + 1,
                       bootstrap_counts=self.inputs['bootstrap_counts'][::-1],
                       worker_source=self.inputs['worker_source'] + '# changed',
                       software={'python': '3.14.0'}, beta=self.inputs['beta'] + 1e-4)
        for field, value in changes.items():
            with self.subTest(field=field):
                with self.assertRaises(RuntimeError):
                    load_bootstrap_cache(self.path, **{**self.inputs, field: value})

    def test_nonfinite_or_wrong_shape_fit_is_rejected(self):
        for beta in [np.array([np.nan, 0.]), np.array([np.inf, 0.]), np.zeros(3)]:
            with self.subTest(beta=beta):
                with self.assertRaisesRegex(RuntimeError, 'Modellfit'):
                    load_bootstrap_cache(self.path, **{**self.inputs, 'beta': beta})

    def test_predictor_tolerance_cannot_be_bypassed_by_small_coefficient_change(self):
        # Ein kleiner Koeffizientenunterschied kann bei großem X fachlich relevant sein.
        X = self.inputs['X'].copy()
        X[1, 1] = 1e8
        with self.assertRaisesRegex(RuntimeError, 'Modellfit'):
            load_bootstrap_cache(self.path, **{**self.inputs, 'X': X,
                                              'beta': self.inputs['beta'] + 1e-9})

    def test_modified_cache_is_rejected(self):
        self.path.write_bytes(self.path.read_bytes() + b'changed')
        with self.assertRaisesRegex(RuntimeError, 'Cache-Datei verändert'):
            load_bootstrap_cache(self.path, **self.inputs)

    def test_reference_parameters_remain_bound_to_original_fingerprint(self):
        self.manifest['reference_beta'][0] += 1e-12
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, 'Designmatrix'):
            load_bootstrap_cache(self.path, **self.inputs)

    def test_missing_manifest_has_actionable_error(self):
        self.path.with_suffix('.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'Referenzmanifest fehlt.*RECOMPUTE_BOOTSTRAP'):
            load_bootstrap_cache(self.path, **self.inputs)


if __name__ == '__main__':
    unittest.main()
