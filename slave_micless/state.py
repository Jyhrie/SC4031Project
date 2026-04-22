import numpy as np

class State:
    def __init__(self):
        self.audio_buffer = None
        self.streaming = False
        self.stream_buf = []
        self.stream_count = 0

        self.ws = None
        self.loop = None