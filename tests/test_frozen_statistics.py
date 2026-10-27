"""Ergebnisregression aus eingefrorenen Daten: keine Modelle, keine neuen Experimente."""
from pathlib import Path
import hashlib
import io
import sys
import tempfile
import unittest
from faktenretention.frozen_data import baseline_bytes
import zipfile

ROOT=Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd
from faktenretention.analysis_data import analyze_run


class FrozenStatisticsTest(unittest.TestCase):
    def test_frozen_hashes(self):
        for line in (ROOT/'data/frozen_assets.sha256').read_text().splitlines():
            digest,name=line.split('  ',1)
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes() if (ROOT/name).is_file() else baseline_bytes(name)).hexdigest(),digest,name)

    def test_qa_labels_means_intervals_and_paired_contrasts(self):
        from faktenretention.offline_guard import install
        install()
        with tempfile.TemporaryDirectory(prefix='anhang_statistiktest_') as temporary:
            out=analyze_run(output=Path(temporary))
            with zipfile.ZipFile(ROOT/'results/originaltabellen.zip') as reference:
                def original(name):
                    return pd.read_csv(io.BytesIO(reference.read(name)),keep_default_na=False,
                                       compression='gzip' if name.endswith('.gz') else None)
                labels=original('qa__qa_labels_reproduziert.csv.gz')
                new=pd.read_csv(out/'qa_labels.csv.gz',keep_default_na=False,low_memory=False)
                key=['article_id','method','stage_target_words','canonical_fact_id']
                joined=labels.merge(new,on=key,validate='one_to_one',suffixes=('_old','_new'))
                self.assertEqual(len(joined),48000)
                for column in ['label','analysis_eligible','technical_missing','grounded']:
                    self.assertTrue(joined[column+'_old'].equals(joined[column+'_new']),column)
                article=pd.read_csv(out/'retention_artikelebene.csv')
                old=original('qa__retention_artikelebene.csv')
                j=old.merge(article,on=key[:3],validate='one_to_one',suffixes=('_old','_new'))
                self.assertEqual(len(j),2400)
                for column in ['n_total','n_valid','n_missing','n_retained','n_lost','n_reference_discordant',
                               'retention_observed','retention_lower','retention_upper','fully_observed','generation_chain_failure']:
                    np.testing.assert_allclose(j[column+'_old'].astype(float),j[column+'_new'].astype(float),rtol=0,atol=1e-12)
                means=pd.read_csv(out/'mittelwerte_95ci.csv')
                old=original('qa__retention_mittelwerte_95ci.csv')
                j=old[old.policy.eq('observed')].merge(means[means.metric.eq('retention_observed')],
                    on=['method','stage_target_words'],validate='one_to_one',suffixes=('_old','_new'))
                self.assertEqual(len(j),24)
                for column in ['mean','ci_low','ci_high','n_articles']:
                    np.testing.assert_allclose(j[column+'_old'],j[column+'_new'],rtol=0,atol=1e-12)
                old=original('laenge__retention_und_drei_rangfolgen_24_gruppen.csv')
                j=old.merge(means[means.metric.eq('retention_adjusted')],on=['method','stage_target_words'],validate='one_to_one')
                for before,after in [('R_adj_mean','mean'),('R_adj_ci_low','ci_low'),('R_adj_ci_high','ci_high')]:
                    np.testing.assert_allclose(j[before],j[after],rtol=0,atol=1e-12)
                old=original('qa__vier_gepaarte_kontraste_pp_95ci.csv')
                new=pd.read_csv(out/'retention_observed_kontraste.csv')
                j=old.merge(new,on=['first','second','stage_target_words'],validate='one_to_one',suffixes=('_old','_new'))
                self.assertEqual(len(j),16)
                for column in ['mean','ci_low','ci_high','n_articles','n_first_higher','n_tied','n_first_lower']:
                    np.testing.assert_allclose(j[column+'_old'],j[column+'_new'],rtol=0,atol=1e-12)


if __name__=='__main__':unittest.main(verbosity=2)
