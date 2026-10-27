"""Gezielte Regeln und unveränderte wissenschaftliche Kernfunktionen."""
from pathlib import Path
import hashlib, unittest
from faktenretention import scoring
from faktenretention.text_rules import count_words, bounds_for_target, lead_k
from faktenretention.protocol import METHODS,STAGES,CONTRASTS
from faktenretention.statistics import SEED,B
ROOT=Path(__file__).resolve().parents[1]

class ScientificRulesTest(unittest.TestCase):
    def test_original_scientific_sources(self):
        # Gegen den wissenschaftlichen Ausgangsstand vor der Kürzung geprüft.
        expected={'scoring.py': '190720ab8a6fcb161a01deb99f51d34fabf5a07445a2734e63010dc8b904d2af', 'text_rules.py': '67bb777c48d60981ff02f63d98d06ca025786b906a89180280eed46516c59285', 'statistics.py': '7630748f085fc99d3e6f0da276cc0d1b2c944cf163309b9cfd00ae6d52e421f7', 'analysis_bootstrap_worker.py': 'f544319c4d80d2549693f2be35adb00b9e634c2664cc1de6a694335d790b916f'}
        for name,digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT/'src/faktenretention'/name).read_bytes()).hexdigest(),digest,name)
    def test_protocol_and_boundaries(self):
        self.assertEqual(STAGES,[500,300,150,75])
        self.assertEqual(len(METHODS),6)
        self.assertEqual(len(CONTRASTS),4)
        self.assertEqual((SEED,B),(32,10000))
        self.assertEqual(bounds_for_target(75,5.0),(71,79))
        self.assertEqual(bounds_for_target(150,5.0),(142,158))
        text='Alice, Bob! Carol-Dan.'
        self.assertEqual(count_words(text),4)
        short=lead_k(text,3)
        self.assertEqual(count_words(short),3)
        self.assertTrue(text.startswith(short))
    def test_qa_grounding_and_reference_match(self):
        self.assertTrue(scoring.support_is_grounded('Alice won 50 percent.','50 percent','Alice won 50 percent.'))
        self.assertFalse(scoring.support_is_grounded('Alice won 50 percent.','Bob','Bob won.'))
        self.assertFalse(scoring.retained('Alice','NOT_FOUND'))
        self.assertTrue(scoring.retained('50 percent','50%'))

if __name__=='__main__':unittest.main()
