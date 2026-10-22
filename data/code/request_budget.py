"""Ein gemeinsames, dauerhaftes Anfragebudget für Fakten, Summaries und QA.

Keine Verbindung und keine Generierung beim Import. Nur activate() schaltet
explizit ausgewählte Experimentphasen frei. Reservierung erfolgt vor jedem POST.
"""
from pathlib import Path
import fcntl
import json
import os
import urllib.request

ACTIVE_RUN = None
SIMULATED_RESPONDER = None  # Ausschließlich vom mitgelieferten Schnittstellentest gesetzt.


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def activate(run, config, *, verify_models=True):
    global ACTIVE_RUN
    run = Path(run).resolve()
    if config['mode'] not in ['mini', 'full']:
        raise ValueError('Generierung benötigt ausdrücklich mini oder full.')
    if config['mode'] == 'mini':
        if len(config['article_ids']) > 5 or len(config['methods']) != 1 or not 1 <= config['max_requests'] <= 30:
            raise ValueError('Mini-Test: höchstens fünf Artikel, eine Methode, 30 Anfragen.')
    elif not isinstance(config['max_requests'], int) or config['max_requests'] < 1:
        raise ValueError('Vollausführung benötigt ein ausdrückliches Anfragebudget.')
    run.mkdir(parents=True, exist_ok=True)
    config_path = run / 'execution.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != config:
            raise ValueError('Vorhandener Lauf hat andere Daten, Modelle oder Einstellungen. Neues Ausgabeverzeichnis wählen.')
    else:
        save_json(config_path, config)
    ledger_path = run / 'request_ledger.json'
    if not ledger_path.exists():
        save_json(ledger_path, dict(actual_generation_requests=0, simulated_requests=0))
    ACTIVE_RUN = run
    if verify_models and SIMULATED_RESPONDER is None and config['required_roles']:
        try:
            with urllib.request.urlopen(f"http://{config['host']}/api/tags", timeout=5) as response:
                installed = {m['name']:m.get('digest') for m in json.load(response).get('models', [])}
        except OSError as error:
            ACTIVE_RUN = None
            raise RuntimeError('Lokale Modelllaufzeit nicht erreichbar. Kein Download.') from error
        for role in config['required_roles']:
            model = config['models'][role]
            if installed.get(model['name']) != model['digest']:
                ACTIVE_RUN = None
                raise RuntimeError('Exaktes Modell fehlt oder Digest stimmt nicht: '+model['name']+'. Kein Download.')


def guarded_generate(payload, host, timeout):
    if ACTIVE_RUN is None:
        raise SystemExit('Generierung gesperrt: kein ausdrücklich aktivierter Experimentlauf.')
    run = ACTIVE_RUN
    config = json.loads((run/'execution.json').read_text())
    if host != config['host'] or payload['model'] not in {x['name'] for x in config['models'].values()}:
        raise SystemExit('Modell oder Server weicht vom aktivierten Lauf ab.')
    # Die Sperre bleibt auch während des Requests aktiv: keine parallelen Aufrufe.
    with (run/'requests.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Ein anderer Modellaufruf läuft bereits.')
        ledger_path = run/'request_ledger.json'
        ledger = json.loads(ledger_path.read_text())
        total = ledger['actual_generation_requests'] + ledger['simulated_requests']
        if total >= config['max_requests']:
            # SystemExit wird vom historischen except Exception nicht verschluckt.
            raise SystemExit('Anfragebudget erreicht; Lauf kontrolliert beendet. Keine weitere Anfrage.')
        field = 'simulated_requests' if SIMULATED_RESPONDER else 'actual_generation_requests'
        ledger[field] += 1
        save_json(ledger_path, ledger)
        if SIMULATED_RESPONDER:
            return SIMULATED_RESPONDER(payload)
        request = urllib.request.Request(f'http://{host}/api/generate', data=json.dumps(payload).encode(),
                                         headers={'Content-Type':'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
