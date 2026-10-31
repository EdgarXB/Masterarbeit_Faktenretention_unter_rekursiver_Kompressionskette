"""Check that shipped provenance does not depend on a researcher's home directory."""
import hashlib
import json
import re
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERSONAL_PATH = re.compile(r'(?:/(?:home|Users)/[^/\s"\\]+/|[A-Za-z]:\\\\Users\\\\)', re.I)


class PortableProvenanceTest(unittest.TestCase):
    def test_metadata_hashes_match_portable_files(self):
        root = ROOT / 'data/model_metadata/provenance'
        sources = json.loads((root / 'sources.json').read_text())
        for name, entry in sources.items():
            with self.subTest(name=name):
                content = (root / name).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), entry['sha256'])
                self.assertNotRegex(json.dumps(entry), PERSONAL_PATH)
        self.assertNotRegex((root / 'environment_20260824.json').read_text(), PERSONAL_PATH)

    def test_archive_manifests_are_portable(self):
        for name in ('demonstrationen.zip', 'modellpilot.zip'):
            with zipfile.ZipFile(ROOT / 'data' / name) as archive:
                for member in archive.namelist():
                    if member.endswith(('.json', '.jsonl', '.csv', '.txt')):
                        with self.subTest(archive=name, member=member):
                            self.assertNotRegex(archive.read(member).decode('utf-8'), PERSONAL_PATH)
                if name == 'demonstrationen.zip':
                    for member in ('candidate_rollouts/manifest.json', 'validation_rollouts/manifest.json'):
                        manifest = json.loads(archive.read(member))
                        self.assertIn(manifest['input'], archive.namelist())


if __name__ == '__main__':
    unittest.main()
