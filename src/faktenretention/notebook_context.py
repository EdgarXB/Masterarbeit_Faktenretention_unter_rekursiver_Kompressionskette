"""Explicit paths and offline plotting for fresh notebook kernels."""
from dataclasses import dataclass
from pathlib import Path
import json
import os
from faktenretention.paths import PROJECT_ROOT

@dataclass(frozen=True)
class AnalysisContext:
    project: Path
    work: Path
    analysis_id: str

def display_figure(figure):
    """Embed a bounded PNG even when the original Agg backend is retained."""
    from io import BytesIO
    from IPython.display import Image, display
    buffer = BytesIO()
    figure.savefig(buffer,format='png',dpi=110,bbox_inches='tight')
    display(Image(data=buffer.getvalue()))

def open_context(analysis_id):
    configured = os.environ.get('FAKTENRETENTION_WORK')
    if not configured:
        raise RuntimeError('FAKTENRETENTION_WORK auf den vorbereiteten workspace setzen; '
                           'oder python 05_analyze.py --only '+analysis_id+' verwenden.')
    work = Path(configured).expanduser()
    if not work.is_absolute() or not (work/'workspace.json').is_file():
        raise ValueError('Absoluter, vorbereiteter workspace erforderlich.')
    if json.loads((work/'workspace.json').read_text())['project'] != str(PROJECT_ROOT):
        raise ValueError('Arbeitskopie gehört zu einem anderen Projekt.')
    from faktenretention.offline_guard import install
    install()
    import matplotlib.pyplot as plt
    def show(*args, **kwargs):
        for number in plt.get_fignums():
            figure = plt.figure(number)
            display_figure(figure)
            plt.close(figure)
    plt.show = show
    return AnalysisContext(PROJECT_ROOT, work, analysis_id)
