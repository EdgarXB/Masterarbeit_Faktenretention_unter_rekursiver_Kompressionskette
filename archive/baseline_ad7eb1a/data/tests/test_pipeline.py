"""Nur simulierte Antworten. Netzverbindungen sind in diesem Test gesperrt."""
from pathlib import Path
import json,sys,tempfile,unittest,urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import request_budget as budget
import research_pipeline as pipeline
from analyze_run import analyze_run


def deny_network(*args,**kwargs):raise AssertionError('Echte Netzverbindung im Mock-Test!')


def fake_response(payload):
    model=payload['model']
    if model==pipeline.MODELS['facts']['name']:
        facts=[dict(fact_id=str(i),fact=f'Alice reported event {i}.',question=f'Who reported event {i}?',reference_answer='Alice',entity_type='PERSON',source_evidence='Alice reported event '+str(i)+'.') for i in range(24)]
        return dict(response=json.dumps(facts),done=True)
    if model==pipeline.MODELS['qa']['name']:
        return dict(response=json.dumps(dict(answer='Alice',support='Alice',confidence='HIGH')),done=True)
    # Different lengths are deliberate: the actual length diagnostics must detect them.
    return dict(response='Alice reported event 0. '+('word '*495)+'.',done=True)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='anhang_mock_')
        self.path=Path(self.temp.name)
        self.old=urllib.request.urlopen;urllib.request.urlopen=deny_network
        budget.SIMULATED_RESPONDER=fake_response
        article=' '.join('Alice reported event '+str(i)+'.' for i in range(24))+' '+('word '*704)
        assert pipeline.count_words(article)==800
        self.input=self.path/'input.csv'
        pipeline.write_csv(self.input,[dict(article=article,highlights='Alice reported event 0.')])
    def tearDown(self):
        urllib.request.urlopen=self.old;budget.SIMULATED_RESPONDER=None;budget.ACTIVE_RUN=None;self.temp.cleanup()
    def prepare(self,name='run'):
        run=self.path/name;pipeline.prepare_articles(self.input,run);return run
    def test_default_mini_complete_pipeline(self):
        run=self.prepare()
        config=pipeline.start_run(run,mode='mini')
        pipeline.generate_facts(run,config);pipeline.generate_summaries(run,config);pipeline.generate_qa(run,config);pipeline.compute_metrics(run,config);analyze_run(run)
        ledger=json.loads((run/'request_ledger.json').read_text())
        self.assertEqual(ledger,dict(actual_generation_requests=0,simulated_requests=22))
        self.assertEqual(len(pipeline.read_csv(run/'data/canonical_facts.csv')),20)
        self.assertEqual(len(pipeline.read_csv(run/'data/answers.csv')),20)
        labels=pipeline.read_csv(run/'analysis/qa_labels.csv.gz');self.assertTrue(all(r['label']=='retained' for r in labels))
        # Resume uses identical config and cached stages/QA, with no further request.
        pipeline.start_run(run,mode='mini');pipeline.generate_facts(run,config);pipeline.generate_summaries(run,config);pipeline.generate_qa(run,config)
        self.assertEqual(json.loads((run/'request_ledger.json').read_text()),ledger)
    def test_budget_including_retries(self):
        run=self.prepare();config=pipeline.start_run(run,mode='mini')
        def failing(payload):raise OSError('Simulated transport failure')
        budget.SIMULATED_RESPONDER=failing
        for i in range(30):
            with self.assertRaises(OSError):budget.guarded_generate(dict(model=pipeline.MODELS['qa']['name']),config['host'],1)
        with self.assertRaises(SystemExit):budget.guarded_generate(dict(model=pipeline.MODELS['facts']['name']),config['host'],1)
        self.assertEqual(json.loads((run/'request_ledger.json').read_text())['simulated_requests'],30)
    def test_caps_before_generation(self):
        for kwargs in [dict(article_limit=6),dict(max_requests=31),dict(method='plain',stages=[500,300])]:
            with self.assertRaises(ValueError):pipeline.start_run(self.prepare(str(len(list(self.path.iterdir())))),mode='mini',**kwargs)
        self.assertIsNone(budget.ACTIVE_RUN)
    def test_missing_runtime_and_wrong_digest_make_zero_requests(self):
        from unittest.mock import patch
        import io
        budget.SIMULATED_RESPONDER=None
        for name,response in [('unavailable',OSError('Simulated missing runtime')),('wrong_digest',None)]:
            run=self.prepare(name)
            if response is not None:
                with patch.object(urllib.request,'urlopen',side_effect=response):
                    with self.assertRaises(RuntimeError):pipeline.start_run(run,mode='mini')
            else:
                with patch.object(urllib.request,'urlopen',return_value=io.BytesIO(b'{"models":[]}')):
                    with self.assertRaises(RuntimeError):pipeline.start_run(run,mode='mini')
            self.assertEqual(json.loads((run/'request_ledger.json').read_text())['actual_generation_requests'],0)
    def test_all_methods_recursive_full_mock(self):
        run=self.prepare();config=pipeline.start_run(run,mode='full',max_requests=510)
        pipeline.generate_facts(run,config);pipeline.generate_summaries(run,config);pipeline.generate_qa(run,config);pipeline.compute_metrics(run,config);analyze_run(run)
        self.assertEqual(len(pipeline.read_csv(run/'data/summaries.csv')),6)
        self.assertEqual(len(pipeline.read_csv(run/'data/answers.csv')),480)
        ledger=json.loads((run/'request_ledger.json').read_text())
        self.assertEqual(ledger,dict(actual_generation_requests=0,simulated_requests=501))
        for path in (run/'states').glob('*/*.json'):
            stages=json.loads(path.read_text())['stages'];self.assertEqual([r['target_words'] for r in stages],pipeline.STAGES)
            for previous,current in zip(stages,stages[1:]):self.assertEqual(current['input_text'],previous['selected']['response'])

class ConstructionControlTest(unittest.TestCase):
    def test_explicit_phase_wiring_without_models(self):
        from unittest.mock import patch
        import types
        calls=[]
        interrupted=[False]
        with tempfile.TemporaryDirectory(prefix='anhang_demo_mock_') as temporary:
            root=Path(temporary);out=root/'bank'
            def articles(path,n,prefix):
                pipeline.write_csv(path,[dict(article=(prefix+str(i)+' ')+('word '*799),highlights='A reference.') for i in range(n)])
            articles(root/'train.csv',130,'Train');articles(root/'validation.csv',30,'Validation')
            def phase(name):
                def main():
                    args=sys.argv[1:];calls.append((name,args))
                    if name=='generate_cot_candidate_rollouts' and not interrupted[0]:
                        interrupted[0]=True
                        raise RuntimeError('simulated interruption after article preparation')
                    destination=Path(args[args.index('--output-dir')+1]);destination.mkdir(parents=True,exist_ok=True)
                    if name in ['select_autocot_demonstrations','select_automatecot_demonstrations']:
                        method='auto_cot' if name=='select_autocot_demonstrations' else 'automate_cot'
                        selected={str(t):[dict(candidate_id=str(i),input_text='Input.',thinking='Reason.',response='Summary.') for i in range(5)] for t in pipeline.STAGES}
                        (destination/(method+'_demonstrations.json')).write_text(json.dumps(selected))
                    if name=='build_shared_pool_fewshot_prompts':
                        prompts={}
                        for t in pipeline.STAGES:
                            f=destination/f'few_shot_cot_{t}.txt';f.write_text('{{NEW_INPUT}}')
                            prompts[str(t)]=dict(path=f.name,sha256=pipeline.sha(f))
                        (destination/'few_shot_cot_manifest.json').write_text(json.dumps(dict(prompts=prompts)))
                    return 0
                return main
            names=['generate_cot_candidate_rollouts','select_autocot_demonstrations','score_and_prune_automatecot_candidates','select_automatecot_demonstrations','build_shared_pool_fewshot_prompts']
            fake_modules={name:types.SimpleNamespace(main=phase(name)) for name in names}
            fake_modules['select_automatecot_demonstrations'].build_fewshot_prompt=lambda text,target,demos: text
            fake_resources=types.SimpleNamespace(configure=lambda r:None,local_model=lambda role:root, summac_resources=lambda:(root,root))
            fake_modules['metric_resources']=fake_resources
            with patch.dict(sys.modules,fake_modules),patch.object(urllib.request,'urlopen',deny_network):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                    pipeline.construct_demonstrations(root/'train.csv',root/'validation.csv',out,resources={},max_requests=100000,verify_models=False)
                paths=[out/'candidate_input/data/articles.csv',out/'validation_input/data/articles.csv']
                unchanged=[(path.stat().st_mtime_ns,pipeline.sha(path)) for path in paths]
                calls.clear()
                budget.save_json(out/'request_ledger.json',dict(actual_generation_requests=0,simulated_requests=3))
                bank=pipeline.construct_demonstrations(root/'train.csv',root/'validation.csv',out,resources={},max_requests=100000,verify_models=False)
                self.assertEqual(unchanged,[(path.stat().st_mtime_ns,pipeline.sha(path)) for path in paths])
                self.assertEqual(json.loads((out/'request_ledger.json').read_text())['simulated_requests'],3)
                with self.assertRaisesRegex(ValueError,'andere Daten, Modelle oder Einstellungen'):
                    pipeline.construct_demonstrations(root/'train.csv',root/'validation.csv',out,resources={},max_requests=100000,device='cuda',verify_models=False)
                old_calls=len(calls)
                (root/'train.csv').write_text((root/'train.csv').read_text().replace('A reference.','Changed reference.'))
                with self.assertRaisesRegex(ValueError,'verändert'):
                    pipeline.construct_demonstrations(root/'train.csv',root/'validation.csv',out,resources={},max_requests=100000,verify_models=False)
                self.assertEqual(len(calls),old_calls)
            self.assertTrue((bank/'bank_manifest.json').is_file())
            self.assertEqual([n for n,a in calls],[names[0],names[0],*names[1:]])
            args=dict(calls[4:5])['select_automatecot_demonstrations']
            for flag,value in [('--sample-size','5'),('--pge-avg-samples','10'),('--epochs','5'),('--batch-size','10'),('--lr','0.001'),('--num-ctx','524288')]:
                self.assertEqual(args[args.index(flag)+1],value)
            self.assertEqual(json.loads((out/'request_ledger.json').read_text())['actual_generation_requests'],0)

if __name__=='__main__':unittest.main(verbosity=2)
