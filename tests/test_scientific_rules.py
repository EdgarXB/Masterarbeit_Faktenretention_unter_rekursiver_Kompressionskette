"""Guard the extracted scientific rules against their byte-preserved baseline."""
from pathlib import Path
import ast
import unittest
import hashlib
import json
import zipfile
from faktenretention.frozen_data import baseline_bytes

ROOT=Path(__file__).resolve().parents[1]

def functions(path):
    return {n.name:n for n in ast.parse(path if isinstance(path,str) else path.read_text()).body if isinstance(n,ast.FunctionDef)}

class ScientificRulesTest(unittest.TestCase):
    def test_compact_baseline_is_complete_and_byte_preserved(self):
        entries={name:digest for digest,name in
                 (line.split('  ',1) for line in (ROOT/'archive/baseline_ad7eb1a.sha256').read_text().splitlines())}
        with zipfile.ZipFile(ROOT/'archive/baseline_ad7eb1a.zip') as archive:
            self.assertEqual(set(archive.namelist()),set(entries))
            for name,digest in entries.items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(),digest,name)
            provenance=json.loads((ROOT/'data/code_provenance.json').read_text())
            original={row['file']:row for row in json.loads(archive.read('data/code_provenance.json'))}
            for row in provenance['paths']:
                self.assertTrue((ROOT/row['active']).is_file())
                self.assertIn(row['baseline_member'],entries)
                if 'original_sha256' in row:
                    self.assertEqual(row['original_sha256'],original[row['baseline_member']]['original_sha256'])

    def test_fingerprinted_sources_and_protocol_constants_are_preserved(self):
        for old,new in [('qa_scorer_snapshot','scoring'),('summary_rules','text_rules'),('analysis_bootstrap_worker','analysis_bootstrap_worker')]:
            self.assertEqual(baseline_bytes(f'data/code/{old}.py'),
                             (ROOT/f'src/faktenretention/{new}.py').read_bytes(),new)
        from faktenretention.protocol import METHODS, STAGES, CONTRASTS
        from faktenretention.statistics import SEED, B
        self.assertEqual(METHODS,['lead_k','plain','zero_shot_cot','few_shot_cot','auto_cot','automate_cot'])
        self.assertEqual(STAGES,[500,300,150,75])
        self.assertEqual(CONTRASTS,[('zero_shot_cot','plain'),('few_shot_cot','plain'),('auto_cot','few_shot_cot'),('automate_cot','auto_cot')])
        self.assertEqual((SEED,B),(32,10000))

    def test_scorer_text_and_statistical_functions_keep_the_original_ast(self):
        for old,new in [('qa_scorer_snapshot','scoring'),('summary_rules','text_rules'),('analysis_statistics','statistics')]:
            original=functions(baseline_bytes(f'data/code/{old}.py').decode())
            current=functions(ROOT/f'src/faktenretention/{new}.py')
            for name,node in original.items():
                self.assertEqual(ast.dump(node),ast.dump(current[name]),f'{new}.{name}')

    def test_extracted_prompt_transport_and_cache_functions_are_unchanged(self):
        original=functions(baseline_bytes('data/code/iterative_summarize.py').decode())
        current=functions(ROOT/'src/faktenretention/model_transport.py')
        for name in ['build_prompt','apply_model_prompt_controls','ollama_generate']:
            for node in ast.walk(original[name]):
                if isinstance(node,ast.ImportFrom) and node.module=='request_budget':
                    node.module='faktenretention.request_budget'
            self.assertEqual(ast.dump(original[name]),ast.dump(current[name]),name)
        original=functions(baseline_bytes('02_demonstrations.py').decode())
        current=functions(ROOT/'src/faktenretention/demonstrations/rules.py')
        for name in ['sha256_text','cache_key','stage_input_valid']:
            self.assertEqual(ast.dump(original[name]),ast.dump(current[name]),name)

if __name__=='__main__':unittest.main()
