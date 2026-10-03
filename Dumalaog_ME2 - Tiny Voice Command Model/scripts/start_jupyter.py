from pathlib import Path
import json, os, subprocess, sys, time, webbrowser
from urllib.request import Request, urlopen
from urllib.parse import urlencode
root=Path(__file__).resolve().parents[1]
workspace=root.parents[1]
runtime=workspace/'.jupyter_runtime_vcm'
runtime.mkdir(exist_ok=True)
def open_existing():
    for path in runtime.glob('jpserver-*.json'):
        try:
            server=json.loads(path.read_text())
            if server.get('port')!=8890 or Path(server.get('root_dir','')).resolve()!=root.resolve(): continue
            req=Request(server['url']+'api/status',headers={'Authorization':'token '+server['token']})
            with urlopen(req,timeout=2) as response:
                if response.status!=200: continue
            webbrowser.open(server['url']+'lab/tree/notebooks/ME2_Tiny_VCM_Training.ipynb?'+urlencode({'token':server['token']}))
            print('Opened the ME2 training notebook in JupyterLab at http://127.0.0.1:8890 (authentication token omitted).')
            return True
        except (OSError,ValueError,KeyError): pass
    return False
if open_existing(): raise SystemExit(0)
env=os.environ.copy(); env['JUPYTER_RUNTIME_DIR']=str(runtime)
proc=subprocess.Popen([sys.executable,'-m','jupyterlab','--no-browser','--ip=127.0.0.1','--port=8890','--ServerApp.port_retries=0','--ServerApp.log_level=CRITICAL','--ServerApp.root_dir='+str(root)],cwd=root,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
for attempt in range(60):
    if open_existing(): raise SystemExit(0)
    if proc.poll() is not None: raise RuntimeError('Jupyter exited during startup: '+str(proc.returncode))
    time.sleep(1)
raise RuntimeError('Jupyter startup timed out')
