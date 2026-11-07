"""Loopback-only keyboard panel; control files target one dedicated SITL container."""
import json
import secrets
import socket
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent/'artifacts/keyboard'
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.Lock()
INPUT = dict(enabled=False,keys=[],action='none',action_id=time.time_ns(),updated_at=0.)
KEYS = {'KeyW','KeyS','KeyA','KeyD','KeyQ','KeyE','ArrowUp','ArrowDown'}


def worker():
    try:
        value = json.loads((DATA/'status.json').read_text())
        if time.time()-value.get('updated_at',0) > 3:
            return dict(phase='UNAVAILABLE',ready=False,reason='Simulator controller is not responding')
        return value
    except (OSError,ValueError):
        return dict(phase='STARTING',ready=False,reason='Starting simulator controller')


def save():
    DATA.mkdir(parents=True,exist_ok=True)
    tmp=DATA/'input.tmp'
    tmp.write_text(json.dumps(INPUT))
    tmp.replace(DATA/'input.json')


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,directory=str(ROOT),**kwargs)
    def allowed(self):
        host = self.headers.get('Host','')
        return host in ('127.0.0.1:8087','localhost:8087') and self.headers.get('Origin', 'http://'+host) == 'http://'+host
    def reply(self,code,value):
        data=json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type','application/json')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        if not self.allowed():
            return self.reply(403,{'error':'Local panel only'})
        path=urlparse(self.path).path
        if path=='/api/rc/status':
            return self.reply(200,dict(token=TOKEN,worker=worker(),simulator_url='http://127.0.0.1:8086/vnc.html?autoconnect=1&resize=scale'))
        if path in ('/','/keyboard.html'):
            self.path='/keyboard.html'
            return super().do_GET()
        self.send_error(404)
    def do_POST(self):
        if not self.allowed() or self.path!='/api/rc/input':
            return self.reply(403,{'error':'Local panel only'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0 < size <= 4096:
                raise ValueError('Invalid request size')
            row=json.loads(self.rfile.read(size))
            if not isinstance(row,dict) or row.get('token')!=TOKEN:
                return self.reply(403,{'error':'Reload the control panel'})
            if type(row.get('enabled')) is not bool or not isinstance(row.get('keys'),list) or any(k not in KEYS for k in row['keys']):
                raise ValueError('Invalid keyboard input')
            action=row.get('action','none')
            if action not in ('none','takeoff','land'):
                raise ValueError('Invalid action')
            with LOCK:
                if action=='takeoff' and (not row['enabled'] or not worker().get('ready')):
                    return self.reply(409,{'error':'Wait for simulator ready, then enable keyboard RC'})
                if action!='none':
                    INPUT.update(action=action,action_id=INPUT['action_id']+1)
                INPUT.update(enabled=row['enabled'],keys=row['keys'] if row['enabled'] else [],updated_at=time.time())
                save()
                try:
                    with socket.create_connection(('127.0.0.1',8088),timeout=.3) as control:
                        control.sendall((json.dumps(INPUT)+'\n').encode())
                        if control.recv(32).strip()!=b'OK':
                            raise OSError('Control packet rejected')
                except OSError:
                    return self.reply(503,{'error':'Simulator control connection unavailable'})
            self.reply(200,{'ok':True})
        except (ValueError,TypeError):
            self.reply(400,{'error':'Invalid control request'})
    def log_message(self,*_):
        pass

if __name__=='__main__':
    save()
    print('Keyboard RC: http://127.0.0.1:8087',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8087),Handler).serve_forever()
