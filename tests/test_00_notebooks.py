"""Fresh kernels, pure imports and the absence of script-launching notebook cells."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import nbformat
from faktenretention import notebook_runner as runner

ROOT=Path(__file__).resolve().parents[1]

class NotebookArchitectureTest(unittest.TestCase):
    def test_analysis_imports_without_experiment_or_linux_lock(self):
        code='''import builtins,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name == 'fcntl':
        raise ImportError('Simulated platform without fcntl')
    if name in ['faktenretention.request_budget','faktenretention.pipeline','torch','transformers']:
        raise AssertionError('Forbidden analysis dependency: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
import faktenretention.io, faktenretention.scoring, faktenretention.statistics
import faktenretention.analysis_data, faktenretention.validation, faktenretention.notebook_runner
assert 'faktenretention.request_budget' not in sys.modules
assert 'fcntl' not in sys.modules
'''
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_notebook_inventory_and_no_old_script_execution(self):
        specs=runner.load_analyses()
        self.assertEqual(len(specs),12)
        self.assertEqual(sum(len(s['figures']) for s in specs),65)
        for spec in specs:
            nb=nbformat.read(ROOT/spec['notebook'],4);nbformat.validate(nb)
            self.assertTrue(any(c.cell_type=='markdown' for c in nb.cells))
            embedded=sum('image/png' in o.get('data',{}) for c in nb.cells for o in c.get('outputs',[]))
            self.assertGreaterEqual(embedded,len(spec['figures']),spec['id'])
            for cell in nb.cells:
                if cell.cell_type=='code':
                    self.assertNotIn('%run',cell.source)
                    self.assertNotIn('analysis__',cell.source)
                    self.assertNotIn('sys.path',cell.source)
                    self.assertNotIn('Path.cwd()',cell.source)

    def test_each_execution_starts_a_fresh_kernel(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);work=root/'work';work.mkdir()
            nb=nbformat.v4.new_notebook(cells=[
                nbformat.v4.new_code_cell("assert 'previous_execution' not in globals()\nprevious_execution = 42\nprint(previous_execution)"),
                nbformat.v4.new_code_cell("from faktenretention.offline_guard import install\ninstall()\nimport matplotlib.pyplot as plt\nfrom faktenretention.notebook_context import display_figure\nfigure, ax = plt.subplots()\nax.plot([1,2],[3,4])\ndisplay_figure(figure)\nplt.close(figure)")])
            nb.cells[0].execution_count=77
            nbformat.write(nb,root/'fresh.ipynb')
            for number in [1,2]:
                out=root/str(number);out.mkdir()
                with patch.object(runner,'PROJECT_ROOT',root):
                    destination=runner.execute_notebook({'notebook':'fresh.ipynb'},work,out,timeout=30)
                executed=nbformat.read(destination,4)
                self.assertEqual(executed.cells[0].execution_count,1)
                self.assertIn('42',executed.cells[0].outputs[0].text)
                self.assertTrue(any('image/png' in o.get('data',{}) for o in executed.cells[1].outputs))

if __name__=='__main__':unittest.main()
