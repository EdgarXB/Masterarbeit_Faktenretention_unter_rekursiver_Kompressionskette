"""Rohdaten, Nenner und feste Ergebnisreferenzen; keine Modellaufrufe."""
from pathlib import Path
import hashlib,unittest
import numpy as np
import pandas as pd
from faktenretention.data import prepare,load_inputs,panel,verify_sources,CELL,KEY
from faktenretention.statistics import means_for,contrasts_for
ROOT=Path(__file__).resolve().parents[1]

class FrozenStatisticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare()
        cls.qa,cls.article=panel(load_inputs())
    def test_frozen_sources_and_position_labels(self):
        self.assertGreater(verify_sources(),0)
        qa=self.qa
        self.assertEqual((len(qa),int(qa.analysis_eligible.sum()),int(qa.technical_missing.sum())),(48000,47804,196))
        self.assertEqual(int(qa.empty_output.sum()),40)
        self.assertFalse(qa.loc[qa.empty_output,'technical_missing'].any())
        # Positionsgenaue Labels der bisherigen wissenschaftlichen Referenz.
        digest=hashlib.sha256(qa.sort_values(KEY)[KEY+['label']].to_csv(index=False).encode()).hexdigest()
        self.assertEqual(digest,'cf5992de26bf282fd04ed7da87d7b698333ef74ab09e97b017444b15f048b4aa')
    def test_article_means_and_paired_bootstrap_intervals(self):
        article=self.article
        self.assertEqual(len(article),2400)
        self.assertTrue(article.n_total.eq(20).all())
        self.assertTrue(article.R_zero.le(article.R_obs+1e-15).all())
        self.assertTrue(article.R_adj.le(article.R_zero+1e-15).all())
        old=pd.read_csv(ROOT/'tests/referenzwerte.csv',keep_default_na=False)
        for kind,new in [('mean',means_for(article).assign(first=lambda x:x.method,second='')),('contrast',contrasts_for(article))]:
            expected=old[old.kind.eq(kind)]
            joined=expected.merge(new,on=['first','second','stage_target_words'],validate='one_to_one',suffixes=('_old','_new'))
            self.assertEqual(len(joined),24 if kind=='mean' else 16)
            for c in ['mean','ci_low','ci_high']:
                np.testing.assert_allclose(joined[c+'_old'],joined[c+'_new'],rtol=0,atol=1e-12)

if __name__=='__main__':unittest.main()
