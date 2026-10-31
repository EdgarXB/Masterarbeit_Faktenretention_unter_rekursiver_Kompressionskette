"""Historische Modellmanifeste und finale Modellrollen offline abgleichen."""
import csv
import gzip
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
PROVENANCE = DATA / 'model_metadata/provenance'


class ModelProvenanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = json.loads((DATA / 'models.json').read_text())
        cls.freeze = json.loads((PROVENANCE / 'model_freeze_20260820.json').read_text())
        cls.amendment = json.loads((PROVENANCE / 'qa_amendment_20260901.json').read_text())
        cls.protocol = json.loads((DATA / 'methoden/protocol.json').read_text())

    def test_model_manifests_digests_and_retained_metadata(self):
        roles = {
            'summary': (self.freeze['models'][0], 'thesis-seed', 'source_manifest_seed.json'),
            'facts': (self.freeze['models'][1], 'thesis-gemma', 'source_manifest_gemma.json'),
            'qa': (self.amendment['model'], 'thesis-qwen38-qa', 'source_manifest_qa.json'),
        }
        for role, (record, name, source) in roles.items():
            with self.subTest(role=role):
                raw = (DATA / 'model_metadata/manifests' / name / 'v1').read_bytes()
                self.assertEqual(raw, (PROVENANCE / source).read_bytes())
                self.assertEqual(hashlib.sha256(raw).hexdigest(), self.models[role]['digest'])
                self.assertEqual(self.models[role]['digest'], record['manifest_digest'])
                self.assertEqual(self.models[role]['name'], record['frozen_alias'])
                manifest = json.loads(raw)
                layers = [manifest['config'], *manifest['layers']]
                model = next(x for x in manifest['layers'] if x['mediaType'] == 'application/vnd.ollama.image.model')
                weight = record.get('weight_blob_sha256') or record['text_weight_blob_sha256']
                self.assertEqual(model['digest'], 'sha256:' + weight)
                if role == 'qa':
                    projector = next(x for x in manifest['layers'] if x['mediaType'] == 'application/vnd.ollama.image.projector')
                    self.assertEqual(projector['digest'], 'sha256:' + record['vision_projector_blob_sha256'])
                for item in layers:
                    path = DATA / 'model_metadata/blobs' / item['digest'].replace(':', '-')
                    if item['mediaType'] not in ('application/vnd.ollama.image.model', 'application/vnd.ollama.image.projector'):
                        self.assertTrue(path.is_file(), str(path))
                        self.assertEqual(path.stat().st_size, item['size'])
                        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item['digest'].split(':', 1)[1])

    def test_qa_amendment_and_recorded_rerun_settings(self):
        self.assertEqual(self.amendment['supersedes_role_from_freeze'], self.freeze['freeze_id'])
        self.assertEqual(self.freeze['models'][2]['frozen_alias'], 'thesis-glm:v1')
        self.assertEqual(self.amendment['model']['frozen_alias'], self.models['qa']['name'])
        self.assertEqual(self.protocol['model_digest'], self.models['summary']['digest'])
        self.assertEqual((self.protocol['temperature'], self.protocol['seed'], self.protocol['top_p'],
                          self.protocol['repeat_penalty'], self.protocol['num_ctx'], self.protocol['num_predict']),
                         (0.0, 32, 0.95, 1.0, 524288, 4096))
        qa = self.amendment['inference']
        self.assertEqual((qa['temperature'], qa['seed'], qa['num_ctx'], qa['num_predict'], qa['think']),
                         (0.0, 32, 8192, 512, False))
        self.assertEqual(qa['technical_recovery_seeds'], [32, 33, 34])

    def test_final_data_use_frozen_model_roles(self):
        for name, column, expected, count in [
            ('fakten.csv.gz', 'canonical_source_model_name', self.models['facts']['name'], 2000),
            ('qa_antworten.csv.gz', 'answer_model_name', self.models['qa']['name'], 48000),
        ]:
            with self.subTest(file=name), gzip.open(DATA / name, 'rt', encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), count)
                self.assertTrue(all(row[column] == expected for row in rows))
        with gzip.open(DATA / 'zusammenfassungen.csv.gz', 'rt', encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 600)
        self.assertTrue(all(row['model_name'] == ('deterministic' if row['method'] == 'lead_k'
                                                  else self.models['summary']['name']) for row in rows))

    def test_recorded_summary_wrapper_and_sampling(self):
        from faktenretention.text_rules import seed_raw

        cot = seed_raw('sample', True, reflection_interval=512, thinking_budget=1024)
        direct = seed_raw('sample', False, reflection_interval=512, thinking_budget=1024)
        self.assertIn('thinking budget, which is 1024', cot)
        self.assertIn('every 512 tokens', cot)
        self.assertIn('thinking budget is 0', direct)
        self.assertFalse(self.protocol['native_thinking'])
        n = 0
        with gzip.open(DATA / 'generierungszustaende.jsonl.gz', 'rt', encoding='utf-8') as stream:
            for line in stream:
                rec = json.loads(line)
                if rec['method'] == 'lead_k':
                    continue
                for stage in rec['state']['stages']:
                    selected = stage['selected']
                    self.assertEqual((selected['temperature'], selected['sampling_seed'], selected['attempt_index']),
                                     (0.0, 32, 1))
                    self.assertEqual(selected['raw_wrapper_cot'], rec['method'] == 'zero_shot_cot')
                    n += 1
        self.assertEqual(n, 2000)


if __name__ == '__main__':
    unittest.main()
