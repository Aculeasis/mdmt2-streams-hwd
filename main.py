import json
import threading
import time

from websocket import create_connection

from lib.audio_utils import StreamDetector
from lib.detectors import Detector
from lib.proxy import proxies
from owner import Owner
from utils import RecognitionCrashMessage

NAME = 'streams-hwd'
TERMINAL_VER_MIN = (0, 16, 1)
CFG_RELOAD = {'vosk-rest': ('server',)}


class Main:
    def __init__(self, cfg, log, owner: Owner):
        self.cfg, self.log, self.own = cfg, log, owner

    def start(self):
        self.own.insert_detectors(DetectorVOSK)

    def stop(self):
        self.own.extract_detectors(DetectorVOSK)

    def reload(self):
        self.own.terminal_call('reload')


class StreamVOSK(StreamDetector, threading.Thread):
    def __init__(self, rate, full_cfg, hot_word_files, **kwargs):
        StreamDetector.__init__(self, **kwargs, resample_rate=rate if rate < 16000 else 16000, rate=rate)
        threading.Thread.__init__(self)
        self._models = hot_word_files
        self._ws = None
        self._end_time, self._final_time, self._start_time = 0, 0, 0
        self._model_info = (None, None)

        url = full_cfg.gt('vosk-rest', 'server')
        try:
            self._ws = create_connection(
                url,
                timeout=60,
                enable_multithread=True,
                **proxies('stt_vosk-rest', ws_format=True),
            )
            self._ws.send(json.dumps({'config': {'sample_rate': self._resample_rate}}))
        except Exception as e:
            raise RecognitionCrashMessage('connection to {}:{}.'.format(url, e))
        self.start()

    def new_chunk(self, buffer: bytes, is_speech=False):
        try:
            self._ws.send_binary(buffer)
        except Exception as e:
            raise RuntimeError(e)

    def end(self):
        if self._ws:
            self._end_time = time.time()
            # noinspection PyBroadException
            try:
                self._ws.send('{"eof" : 1}')
            except Exception:
                pass
            try:
                self.join(timeout=10)
            except RuntimeError:
                pass
            finally:
                self.die()

    def reset(self):
        self.die()

    def die(self):
        if self._ws:
            # noinspection PyBroadException
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    @property
    def recognition_time(self):
        return self._final_time - self._end_time if self._final_time > self._end_time else 0

    @property
    def record_time(self):
        return self._final_time - self._start_time if self._final_time > self._start_time else 0

    @property
    def model_info(self):
        msg = ': "{}"'.format(self._model_info[1]) if self._model_info[1] else ''
        return (*self._model_info, msg)

    def run(self):
        candidate_p, candidate = None, None
        while self._ws:
            # noinspection PyBroadException
            try:
                recv = json.loads(self._ws.recv())
            except ValueError:
                continue
            except Exception:
                break
            if 'partial' in recv:
                if self._current_state == -2:
                    result = self._models.text_processing(recv['partial'])
                    if result:
                        candidate_p = result
                        *self._model_info, _ = candidate_p
                        self._has_detect()
            elif 'text' in recv:
                result = self._models.text_processing(recv['text'], candidate_p)
                if result:
                    candidate = result
                    self._has_detect()
                if self._current_state != -2:
                    break
        self._final_time = time.time()
        self.processing = False
        candidate = candidate if candidate and candidate[2] else candidate_p
        if candidate:
            *self._model_info, self.text = candidate
        self.is_ok = bool(self.text)

    def _has_detect(self):
        if self._current_state == -2:
            self._start_time = time.time()
            self._current_state = 1


class DetectorVOSK(Detector):
    NAME = 'stream-vosk'
    DETECTOR = StreamVOSK
    MODELS_SUPPORT = tuple('1',)
    MUST_PRELOAD = False
    FAKE_MODELS = True
