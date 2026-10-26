"""Declared outputs, failure reports and separate single/full run directories."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from faktenretention import notebook_runner as runner
from faktenretention.validation import compare_table, require_comparisons

class RunnerTest(unittest.TestCase):
    def test_missing_expected_output_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            spec={'expected':[{'path':'absent.csv','reference':'reference.csv'}],'figures':[]}
            with self.assertRaisesRegex(FileNotFoundError,'absent.csv'):
                runner.verify_outputs(spec,path,path)

    def test_empty_comparisons_fail(self):
        with self.assertRaisesRegex(ValueError,'Leere Vergleichsliste'):require_comparisons([])

    def test_missing_additional_figure_export_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            spec={'expected':[],'figures':[],'additional_exports':['figure.svg']}
            with self.assertRaisesRegex(FileNotFoundError,'figure.svg'):
                runner.verify_outputs(spec,path,path)

    def test_table_keys_tolerance_and_diagnostic_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'table.csv'
            old=b'method,value,n_warning_fits\na,0.5,2\nb,0.2,0\n'
            path.write_text('method,value,n_warning_fits\nb,0.2,0\na,0.5,2\n')
            self.assertTrue(compare_table(path,old,'test')['passed'])
            path.write_text('method,value,n_warning_fits\nb,0.2,0\na,0.50000001,2\n')
            self.assertFalse(compare_table(path,old,'test')['passed'])
            path.write_text('method,value,n_warning_fits\nb,0.2,0\na,0.5,3\n')
            self.assertFalse(compare_table(path,old,'test')['passed'])
            path.write_text('article_id,value\n2,0.2\n1,0.5\n')
            self.assertTrue(compare_table(path,b'article_id,value\n1,0.5\n2,0.2\n','numeric_keys')['passed'])

    def test_selection_includes_dependencies(self):
        specs=runner.load_analyses()
        ids=[s['id'] for s in runner.selected_analyses(specs,'11_statistische_laengenkontrolle')]
        self.assertEqual(ids,['03_qa_retention','04_laenge_und_adjustierte_retention','11_statistische_laengenkontrolle'])
        self.assertEqual(ids,[s['id'] for s in runner.selected_analyses(specs,'03_QA_Faktenretention_Laengenkontrolle')])

    def test_single_full_failure_and_skipped_reports_are_separate(self):
        specs=[{'id':name,'dependencies':[],'reused':[],'expected':[{'path':name+'.csv','reference':name+'.csv'}],'figures':[]} for name in ['first','second']]
        def prepare(output,specs):
            output.mkdir();work=output/'workspace';work.mkdir()
            folder=output/'reference/tables';folder.mkdir(parents=True)
            for spec in specs:(folder/(spec['id']+'.csv')).write_text('value\n1\n')
            return work
        def execute(spec,work,output,**kwargs):(work/(spec['id']+'.csv')).write_text('value\n1\n')
        with tempfile.TemporaryDirectory() as temp,patch.object(runner,'prepare_workspace',side_effect=prepare):
            root=Path(temp)
            single=runner.run_analyses(specs,specs[:1],root/'single',execute=execute)
            before=(root/'single/validation.json').read_bytes()
            full=runner.run_analyses(specs,specs,root/'full',execute=execute)
            failed=runner.run_analyses(specs,specs,root/'failed',execute=lambda *a,**kw:None)
            self.assertTrue(single['passed']);self.assertTrue(full['passed']);self.assertFalse(failed['passed'])
            self.assertEqual(single['analyses'][1]['status'],'skipped')
            self.assertEqual(failed['analyses'][0]['status'],'failed')
            self.assertEqual(before,(root/'single/validation.json').read_bytes())

    def test_partial_execution_skips_dependent_analysis(self):
        specs=[{'id':'first','dependencies':[],'reused':[],
                'expected':[{'path':'first.csv','reference':'first.csv'}],'figures':[]},
               {'id':'dependent','dependencies':['first'],'reused':[],
                'expected':[{'path':'dependent.csv','reference':'dependent.csv'}],'figures':[]}]
        def prepare(output,specs):
            output.mkdir();work=output/'workspace';work.mkdir();return work
        def interrupted(spec,work,output,**kwargs):
            (work/'first.csv').write_text('value\n1\n')
            raise RuntimeError('interrupted after one output')
        with tempfile.TemporaryDirectory() as temp,patch.object(runner,'prepare_workspace',side_effect=prepare):
            report=runner.run_analyses(specs,specs,Path(temp)/'run',execute=interrupted)
        self.assertFalse(report['passed'])
        self.assertEqual(report['analyses'][0]['status'],'partially_executed')
        self.assertEqual(report['analyses'][1]['status'],'skipped')
        self.assertEqual(report['analyses'][1]['reason'],'dependency_failed')

if __name__=='__main__':unittest.main()
