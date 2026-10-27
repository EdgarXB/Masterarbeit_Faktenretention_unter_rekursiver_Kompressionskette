# Zweck: Finale Demonstrationen und historische Automate-CoT-Protokolle prüfen.
# Eingaben: ausgewählte Demonstrationsdaten und finale Prompts.
# Ausgaben: Prüfergebnis auf stdout.
# LLM-Aufrufe: keine; keine Kandidatenerzeugung oder Optimierung.
# Beispiel: python 02_demonstrations.py
from pathlib import Path
import hashlib
import math
import random
import re
from faktenretention.demonstrations.rules import cache_key, stage_input_valid
import json
import zipfile

from faktenretention.io import PROJECT_ROOT as ROOT


def audit_cached_optimization(archive, base, report):
    pools = json.loads(archive.read(base+'automate_cot_pruning/automate_cot_pruned_pools.json'))
    validation = [json.loads(line) for line in archive.read(base+'validation_rollouts/stage_candidates.jsonl').decode().splitlines()]
    keys = set()
    comparisons = 0
    for target, stage in report['stages'].items():
        target = int(target)
        val = [r for r in validation if int(r['target_words']) == target and stage_input_valid(r,target)]
        pool = {r['candidate_id']:r for r in pools[str(target)]}
        batches = [(u['sampled_candidate_ids'], val[(u['batch']-1)*10:u['batch']*10], u['sample_losses']) for u in stage['history']]
        batches.append(([stage['ordered_candidate_ids']],val,[stage['validation_mean_loss']]))
        for combinations, batch, expected_losses in batches:
            for ids, expected in zip(combinations,expected_losses):
                losses = []
                for item in batch:
                    key = cache_key(target,[pool[i] for i in ids],item,524288)
                    row = json.loads(archive.read(base+'automate_cot_selection/generation_cache/'+key+'.json'))
                    keys.add(key)
                    assert row['cache_key'] == key
                    assert row['ordered_demo_ids'] == ids
                    assert row['validation_candidate_id'] == item['candidate_id']
                    validity = row['validity']
                    assert row['hard_valid'] == bool(validity['nonempty'] and validity['word_window_success'] and validity['ends_with_complete_sentence'])
                    utility = row['highlight_rougeL_f1'] if row['hard_valid'] else 0.0
                    assert math.isclose(row['utility'],utility,abs_tol=1e-12)
                    assert math.isclose(row['loss'],1-utility,abs_tol=1e-12)
                    losses.append(row['loss'])
                assert math.isclose(sum(losses)/len(losses),expected,abs_tol=1e-12)
                comparisons += 1
    return dict(unique_cached_generations=len(keys), reproduced_combination_losses=comparisons,
                cache_keys_verified=True, optimization_rerun=False)


def audit():
    with zipfile.ZipFile(ROOT / 'data/demonstrationen.zip') as archive:
        base = ''
        report = json.loads(archive.read(base + 'automate_cot_selection/automate_cot_selection_report.json'))
        pruning = json.loads(archive.read(base + 'automate_cot_pruning/automate_cot_pruning_report.json'))
        pool_sources = json.loads(archive.read(base + 'automate_cot_pruning/automate_cot_pruned_pools.json'))
        checks = []
        assert report['policy']['sample_size'] == 5
        assert report['policy']['pge_avg_samples'] == 10
        assert report['policy']['epochs'] == 5
        for target in [500, 300, 150, 75]:
            stage = report['stages'][str(target)]
            prune = pruning['stages'][str(target)]
            assert prune['retained_pool_size'] == 100
            assert prune['minimum_retained_summac'] >= prune['maximum_rejected_eligible_summac']
            assert stage['pool_size'] == 100 and stage['training_updates'] == 15
            assert len(stage['history']) == 15 and len(stage['ordered_candidate_ids']) == 5
            assert len(set(stage['ordered_candidate_ids'])) == 5
            for update in stage['history']:
                assert len(update['sampled_candidate_ids']) == 10
                assert all(len(ids) == len(set(ids)) == 5 for ids in update['sampled_candidate_ids'])
                assert len(update['sample_losses']) == 10
                assert math.isclose(sum(update['sample_losses'])/10, update['mean_loss'], abs_tol=1e-12)
            assert math.isclose(stage['validation_mean_loss'] + stage['validation_mean_utility'], 1.0, abs_tol=1e-12)
            pool = json.loads(archive.read(base+'automate_cot_pruning/automate_cot_pruned_pools.json'))[str(target)]
            chosen = []
            assert len(stage['probabilities']) == 5
            for probabilities in stage['probabilities']:
                assert len(probabilities) == 100 and math.isclose(sum(probabilities),1.0,abs_tol=1e-12)
                index = max((i for i in range(100) if i not in chosen),key=lambda i: probabilities[i])
                chosen.append(index)
            assert [pool[i]['candidate_id'] for i in chosen] == stage['ordered_candidate_ids']
            checks.append(dict(stage=target, candidates=100, positions=5, combinations_per_update=10,
                               updates=15, epochs=5, final_utility=stage['validation_mean_utility']))
        optimization = audit_cached_optimization(archive, base, report)
    # Die aktuellen Few-Shot-Prompts ersetzen die historischen Prompts von Anfang September.
    manifest = json.loads((ROOT / 'prompts/final_few_shot_cot_manifest.json').read_text())
    from faktenretention.demonstrations.build_shared_pool_fewshot_prompts import common_eligibility_reasons, build_prompt
    rng = random.Random(32)
    # Finale Few-Shot-Auswahl aus demselben konservativen gemeinsamen Pool rekonstruieren.
    for target, entry in manifest['prompts'].items():
        prompt = ROOT / 'prompts' / ('final_' + Path(entry['path']).name)
        assert hashlib.sha256(prompt.read_bytes()).hexdigest() == entry['sha256']
        assert prompt.read_text().count('{{NEW_INPUT}}') == 1
        eligible = sorted([row for row in pool_sources[str(target)]
                           if not common_eligibility_reasons(row, int(target))], key=lambda row: str(row['candidate_id']))
        selected = rng.sample(eligible, 5)
        stage = manifest['stages'][str(target)]
        assert len(eligible) == stage['common_eligible_pool_size']
        assert [row['candidate_id'] for row in selected] == stage['ordered_selected_candidate_ids']
        assert build_prompt(int(target), selected) == prompt.read_text()
    result = dict(optimization_cache_audit=optimization, stages=checks, final_fewshot_hashes_verified=True, final_fewshot_selection_reproduced=True, llm_generation_calls=0,
                  optimization_rerun=False)
    print(json.dumps(result,indent=2))
    print('Finale Few-Shot-Prompts und vier historische Optimierungsprotokolle geprüft.')


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Standard: eingefrorene Demonstrationen prüfen. Konstruktion nur ausdrücklich.')
    parser.add_argument('--construct',action='store_true')
    parser.add_argument('--full-experiment',action='store_true')
    parser.add_argument('--train',type=Path)
    parser.add_argument('--validation',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resources',type=Path)
    parser.add_argument('--max-requests',type=int)
    parser.add_argument('--host',default='127.0.0.1:11435')
    parser.add_argument('--device',default='cpu')
    a=parser.parse_args()
    if not a.construct:
        if a.full_experiment:parser.error('--full-experiment benötigt --construct')
        audit();return
    if not a.full_experiment or any(x is None for x in [a.train,a.validation,a.output,a.resources,a.max_requests]):
        parser.error('Konstruktion benötigt --full-experiment --train --validation --output --resources --max-requests')
    from faktenretention.pipeline import construct_demonstrations,load_resources
    construct_demonstrations(a.train,a.validation,a.output,resources=load_resources(a.resources),max_requests=a.max_requests,host=a.host,device=a.device)

if __name__=='__main__':main()
