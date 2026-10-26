"""Rechenarme Auditprüfungen. Alle Netzverbindungen sind im Test gesperrt."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
from faktenretention.input_checks import validate_article_keys, validate_candidate_keys
from faktenretention.runtime_checks import check_analysis_environment, require_supported_platform
import faktenretention.pipeline as pipeline
class AuditTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='anhang_audit_')
        self.root=Path(self.temp.name)
        self.network=patch('urllib.request.urlopen',side_effect=AssertionError('Netzwerk im Audit-Test gesperrt.'))
        self.network.start()
        self.article='Alice '+('word '*799)
    def tearDown(self):
        self.network.stop()
        self.temp.cleanup()
    def test_tested_environment_and_cache_mismatch(self):
        self.assertTrue(check_analysis_environment()['cache_compatible'])
        versions=check_analysis_environment()['versions']
        for key in ['python','numpy','statsmodels']:
            wrong={**versions,key:'abweichend'}
            with self.assertRaisesRegex(SystemExit,key):check_analysis_environment(versions=wrong)
        with self.assertRaisesRegex(SystemExit,'WSL2'):require_supported_platform('Windows')
    def test_duplicate_indices_rejected_even_outside_limit(self):
        for second in ['7','07']:
            source=self.root/'input.csv'
            pipeline.write_csv(source,[dict(dataset_index='7',article=self.article,highlights='A'),
                                       dict(dataset_index=second,article='Bob '+('word '*799),highlights='B')])
            with self.assertRaisesRegex(ValueError,'Doppelter dataset_index'):
                pipeline.prepare_articles(source,self.root/'run',limit=1)
            self.assertFalse((self.root/'run/data/articles.csv').exists())
    def test_implicit_and_explicit_indices_cannot_collide(self):
        source=self.root/'input.csv'
        pipeline.write_csv(source,[dict(dataset_index='1',article=self.article,highlights='A'),
                                   dict(dataset_index='',article='Bob '+('word '*799),highlights='B')])
        with self.assertRaisesRegex(ValueError,'Doppelter dataset_index'):
            pipeline.prepare_articles(source,self.root/'run')
    def test_altered_prepared_indices_fail_before_activation(self):
        source=self.root/'input.csv';run=self.root/'run'
        pipeline.write_csv(source,[dict(article=self.article,highlights='A')])
        path=pipeline.prepare_articles(source,run)
        rows=pipeline.read_csv(path);rows.append({**rows[0],'article_id':'2'})
        pipeline.write_csv(path,rows)
        with patch.object(pipeline,'activate',side_effect=AssertionError('Aktivierung darf nicht erreicht werden.')):
            with self.assertRaisesRegex(ValueError,'Doppelter dataset_index'):
                pipeline.start_run(run,mode='mini')
        self.assertFalse((run/'execution.json').exists())
    def test_original_candidate_cli_checks_all_rows_before_generation(self):
        import faktenretention.demonstrations.generate_cot_candidate_rollouts as original
        source=self.root/'candidates.csv'
        pipeline.write_csv(source,[dict(article_id=str(i),split='train',dataset_index='7',
            article=self.article,highlights='A') for i in [1,2]])
        with patch.object(original,'generate_stage',side_effect=AssertionError('Generierung darf nicht erreicht werden.')):
            with self.assertRaisesRegex(ValueError,'Doppelter dataset_index'):
                pipeline.run_original(original.main,['--input',source,'--output-dir',self.root/'candidates','--limit','1'])
    def test_preparation_resume_requires_same_source_options_and_bytes(self):
        source=self.root/'input.csv';run=self.root/'run'
        pipeline.write_csv(source,[dict(article=self.article,highlights='A')])
        path=pipeline.prepare_articles(source,run,limit=5)
        original=path.read_bytes();stamp=path.stat().st_mtime_ns
        self.assertEqual(path,pipeline.prepare_articles(source,run,limit=5,resume=True))
        self.assertEqual(path.stat().st_mtime_ns,stamp)
        with self.assertRaisesRegex(ValueError,'Konfiguration'):
            pipeline.prepare_articles(source,run,limit=6,resume=True)
        # Gleiche CSV-Zellen mit anderen Bytes reichen für Wiederaufnahme nicht aus.
        path.write_bytes(original.replace(b'\r\n',b'\n'))
        with self.assertRaisesRegex(ValueError,'verändert'):
            pipeline.prepare_articles(source,run,limit=5,resume=True)
        path.write_bytes(original)
        source.write_text(source.read_text().replace('Alice','Carol'))
        with self.assertRaisesRegex(ValueError,'verändert'):
            pipeline.prepare_articles(source,run,limit=5,resume=True)
    def test_candidate_collisions_and_source_mismatch(self):
        row=dict(candidate_id='train:4:500',split='train',dataset_index=4,target_words=500,article_sha256='a')
        with self.assertRaisesRegex(ValueError,'candidate_id'):validate_candidate_keys([row,row])
        other={**row,'candidate_id':'other'}
        with self.assertRaisesRegex(ValueError,'Stufenschlüssel'):validate_candidate_keys([row,other])
        other={**row,'candidate_id':'train:4:300','target_words':300,'article_sha256':'b'}
        with self.assertRaisesRegex(ValueError,'unterschiedliche'):validate_candidate_keys([row,other])
        validate_article_keys([dict(article_id='1',split='train',dataset_index='4'),
                               dict(article_id='2',split='validation',dataset_index='4')])


if __name__=='__main__':unittest.main(verbosity=2)
