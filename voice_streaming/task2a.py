# streaming_test.py
from pesq import pesq
import pandas as pd
import torch
from TTS.api import TTS
import librosa
from speechmos import aecmos, dnsmos, plcmos
import time
# print(TTS().list_models())

device = "cuda" if torch.cuda.is_available() else "cpu"

# text = "Hello! Welcome to our demo"
text = "This example notebook has been designed as a low level example for binary real time streaming using only the prediction of the model  processing the binary data"

# Load model
# tts = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=True, gpu=False).to(device)
SAMPLE_RATE = 16000
model_base = "tts_models/en/ljspeech/"
models = ["glow-tts", "tacotron2-DCA"]
# Split into chunks
chunks = [text[i:i+40] for i in range(0, len(text), 40)]

results = []
for model in models:
    tts = TTS(model_name=model_base+model, progress_bar=True, gpu=False).to(device)
    print(f"\nComplete audio")
    start = time.time()
    tts.tts_to_file(text=text, file_path=f"complete_chunk_{model}.wav")
    latency = round(time.time() - start, 2)
    print(f"Latency: {latency}s")
    ref, _ = librosa.load("complete_chunk_"+model+".wav", sr=16000)
    pesq_score = pesq(16000, ref, ref, 'wb')
    print(pesq_score)
    results.append({
        "model_name": model,
        "chunk_no": "full_text",
        "latency_ms": latency,
        "pesq": pesq_score,
        "relative_pesq": 0
    })
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}: {chunk}")
        start = time.time()
        tts.tts_to_file(text=chunk, file_path=f"chunk_{i + 1}_{model}.wav")
        latency = round(time.time() - start, 2)
        deg, _ = librosa.load("chunk_" + str(i+1) + "_" + model + ".wav", sr=16000)
        print(f"Latency: {latency}s")
        pesq_score = pesq(16000, ref, deg, 'wb')
        print(pesq_score)
        results.append({
            "model_name": model,
            "chunk_no": i,
            "latency_ms": latency,
            "pesq": pesq_score,
            "relative_pesq": pesq_score - pesq(16000, deg, deg, 'wb')
        })
pd.DataFrame(results).to_csv("task2a_metrics.csv")
