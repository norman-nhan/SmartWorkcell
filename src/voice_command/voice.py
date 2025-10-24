from vosk import Model, KaldiRecognizer
import pyaudio, json 
import os 

def voice_command():
    model = Model("model")
    rec = KaldiRecognizer(model, 16000)

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    stream.start_stream()
    while True:
        data = stream.read(4096)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result()["text"])
            return result