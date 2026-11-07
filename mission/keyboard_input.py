"""Bounded local TCP mailbox for simulator keyboard inputs."""
import json
import math
import socketserver
import threading
import time

KEYS=frozenset(('KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','ArrowUp','ArrowDown'))

class InputMailbox:
    def __init__(self,port=8088):
        self.lock=threading.Lock()
        self.last=None
        self.received=0.
        owner=self
        class Receiver(socketserver.StreamRequestHandler):
            def handle(self):
                self.request.settimeout(.3)
                try:
                    data=self.rfile.readline(4097)
                    row=json.loads(data) if len(data)<=4096 else None
                    accepted=owner.accept(row,time.monotonic())
                    self.wfile.write(b'OK\n' if accepted else b'REJECTED\n')
                except (OSError,ValueError,TypeError):
                    return
        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address=True
            daemon_threads=True
        self.server=Server(('0.0.0.0',port),Receiver)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()

    def accept(self,row,now):
        if (not isinstance(row,dict) or type(row.get('enabled')) is not bool
                or not isinstance(row.get('keys'),list) or len(row['keys'])>8
                or any(not isinstance(k,str) or k not in KEYS for k in row['keys'])
                or row.get('action') not in ('none','takeoff','land')
                or type(row.get('action_id')) is not int
                or not isinstance(row.get('updated_at'),(int,float))
                or not math.isfinite(row['updated_at'])):
            return False
        with self.lock:
            if self.last and row['updated_at']<=self.last['updated_at']:
                return False
            self.last=dict(row)
            self.received=now
        return True

    def poll(self,now=None):
        with self.lock:
            # Read the receiver clock under the same lock as the packet.
            # A loop timestamp captured earlier can predate a new arrival.
            current=time.monotonic() if now is None else now
            return self.last if self.last is not None and 0<=current-self.received<=.6 else None

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
