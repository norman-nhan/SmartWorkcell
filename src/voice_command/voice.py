from vosk import Model, KaldiRecognizer
import sounddevice as sd 
import queue 
import json 
import langid

model_en=Model("vosk-model-small-en-us-0.22")
model_jp=Model("vosk-model-small-ja-0.22")

q=queue.Queue() 

def callback(indata, frames, time, status):
    q.put(bytes(indata))

def listen():
    rec=KaldiRecognizer(model_en, 16000)

    with sd.RawInputStream(samplerate=16000, channels=1, device=0, dtype='int16', frames_per_buffer=8192, callback=callback):
        print("Listening...")
        while True:
            data = q.get()
            if rec.AcceptWaveform(data): 
                result = json.loads(rec.Result())
                text=result.get("text", "")
                if text:
                    yield text

def translate(text, src_lang, dest_lang): 
    if src_lang == "en":
        model = model_en
    elif src_lang == "jp":
        model = model_jp
    else:
        raise ValueError(f"Unsupported language: {src_lang}")

    rec = KaldiRecognizer(model, 16000)
    rec.AcceptWaveform(text.encode('utf-8'))
    result = json.loads(rec.Result())
    return result.get("text", "")