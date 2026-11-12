"""Keep image processing off the flight-control and MAVLink timesync loop."""
from threading import Event, Lock, Thread


class CameraWorker:
    def __init__(self, vision):
        self.vision = vision
        self.stop = Event()
        self.lock = Lock()
        self.latest = None
        self.error = None
        self.thread = Thread(target=self._run, daemon=True, name='camera-localization')
        self.thread.start()

    @property
    def sim_usec_now(self):
        return getattr(self.vision, 'sim_usec_now', None)

    def _run(self):
        try:
            while not self.stop.is_set():
                record = self.vision.poll()
                if record is None:
                    self.stop.wait(.002)
                    continue
                with self.lock:
                    self.latest = record
        except Exception as error:
            with self.lock:
                self.error = error

    def poll(self):
        with self.lock:
            if self.error is not None:
                raise self.error
            latest, self.latest = self.latest, None
            return latest

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
        close = getattr(self.vision, 'close', None)
        if close:
            close()
