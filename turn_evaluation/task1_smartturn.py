import os
import librosa
import numpy as np
from transformers import logging
from pipecat.audio.turn.smart_turn.local_coreml_smart_turn import LocalCoreMLSmartTurnAnalyzer
from pipecat.audio.turn.smart_turn.local_smart_turn import LocalSmartTurnAnalyzer
from PipecatCoreMLSmartTurnVADWrapper import PipecatCoreMLSmartTurnVADWrapper
import pandas as pd
import time
import asyncio
import matplotlib.pyplot as plt

logging.set_verbosity(logging.INFO)

def evaluate(annotations, detections, tolerance_ms=200, manual_annotation = False):
    print("annotations = ", annotations)
    print("detections = ", detections)
    tp, fp, fn, latency = 0, 0, 0, []
    # Convert annotations to list of turn-end times
    if not manual_annotation:
        true_turns = [ann[1] for ann in annotations]  # end_ms
    else:
        true_turns = annotations
    detect = [x['end']/1000 for x in detections]
    for turn_true in true_turns:
        closest_detection = min(detect, key=lambda x: abs(x - turn_true))
        if abs(closest_detection - turn_true) <= tolerance_ms:
            tp += 1
            latency.append(abs(closest_detection - turn_true))
        else:
            fn += 1
    fp = abs(len(detections) - tp)  # False positives
    print(tp, fp, fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    avg_latency = np.mean(latency) if latency else 0
    return precision, recall, avg_latency

# Initialize model
test_path = "../data/archive/voxconverse_dev_wav/audio/wmori.wav"
audio, sr = librosa.load(test_path, sr=16000)
df = pd.read_csv("task1_test_groundtruth.csv")
annotation = df['1'].tolist()

# # Original code:
smart_turn_model_path = os.getenv("LOCAL_SMART_TURN_MODEL_PATH")
if not smart_turn_model_path:
    print("Error: LOCAL_SMART_TURN_MODEL_PATH environment variable not set.")
    print("Please clone https://github.com/pipecat-ai/smart-turn and set the variable.")
    print("Example: export LOCAL_SMART_TURN_MODEL_PATH=\"/Users/youruser/Documents/smart-turn\"")
model = LocalCoreMLSmartTurnAnalyzer(smart_turn_model_path=smart_turn_model_path)
model.append_audio(audio, True)
k = asyncio.run(model.analyze_end_of_turn())
print(k[1])

model2 = LocalSmartTurnAnalyzer(smart_turn_model_path=smart_turn_model_path)
model2.append_audio(audio, True)
k2 = asyncio.run(model2.analyze_end_of_turn())
for x in k2:
    print(x)

# # Initialize the VAD wrapper
vad_wrapper = PipecatCoreMLSmartTurnVADWrapper(
    smart_turn_model_path=smart_turn_model_path,
    sampling_rate=sr,
    min_speech_duration_ms=100,
    min_silence_duration_ms=50,
    speech_pad_ms=10,
    vad_stop_secs=0.5,
    completion_probability_threshold=0.7
)

# print(f"Processing {duration_secs}s audio with chunk size of 512 samples...")
start = time.time()
speech_timestamps = asyncio.run(vad_wrapper.get_speech_timestamps(audio, chunk_size_samples=512))
latency = time.time()-start
print("latency = ", latency)
print("\n--- Detected Speech Segments ---")
if not speech_timestamps:
    print("No speech detected.")
for ts in speech_timestamps:
    start_sec = ts["start"] / 1000
    end_sec = ts["end"] / 1000
    print(f"Start: {ts['start']} ms ({start_sec:.2f}s), End: {ts['end']} ms ({end_sec:.2f}s)")

with open("task1_smartturn_test_timestamps.csv", "w+") as f:
    # f.write()
    f.write("Start, End\n")
    for x in speech_timestamps:
        f.write(str(x['start'])+", "+str(x["end"])+"\n")

df = pd.read_csv("task1_test_groundtruth.csv")
annotation = df['1'].tolist()
precision_test, recall_test, latency_test = evaluate(annotation, speech_timestamps, manual_annotation=True)
print(precision_test, recall_test, latency_test)


df_pr = pd.read_csv("task1_vad_metrics.csv")
_, _, precision_silero, recall_silero, latency_silero = df_pr.iloc[0]

metrics = {
    'Method': ['Silero VAD', 'SmartTurn'],
    'Precision': [round(precision_silero, 2), round(precision_test,2)],
    'Recall': [round(recall_silero,2), round(recall_test, 2)],
    'Latency (ms)': [round(latency_silero, 2), round(latency_test, 2)]
}
print(metrics)
df = pd.DataFrame(metrics)
df.to_csv("task1_compare_metrics.csv")

# Plot comparison
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].bar(metrics['Method'], metrics['Precision'], color='blue')
ax[0].set_title('Precision')
ax[1].bar(metrics['Method'], metrics['Recall'], color='green')
ax[1].set_title('Recall')
ax[2].bar(metrics['Method'], metrics['Latency (ms)'], color='red')
ax[2].set_title('Latency')
plt.show()
fig.savefig("Plot_Compare")
