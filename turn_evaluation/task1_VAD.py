import librosa
import numpy as np
import pandas as pd
from transformers import logging
from scipy.io import wavfile
from pydiarization.diarization_wrapper import rttm_to_string
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import matplotlib.pyplot as plt
logging.set_verbosity_info()

# Load audio (replace with your file)
path = "../data/archive/voxconverse_dev_wav/audio/ahnss.wav"
ground_truth_path = "../data/archive/labels/dev/ahnss.rttm"
audio, sr = librosa.load(path, sr=16000, mono=True)

def readRTTM(file):
    data = {}
    rttm = rttm_to_string(file)
    rttm = rttm.split('\n')[:-1]
    for line in rttm:
        data[float(line.split(' ')[3])] = [float(line.split(' ')[4]),
                                           line.split(' ')[7]]

    ground_truth = []
    for start in sorted(data):
        ground_truth.append(((start, start+data[start][0], data[start][1])))
    return ground_truth

def evaluate(annotations, detections, tolerance_ms=20, manual_annotation = False):
    tp, fp, fn, latency = 0, 0, 0, []
    # Convert annotations to list of turn-end times
    if not manual_annotation:
        true_turns = [ann[1] for ann in annotations]  # end_ms
    else:
        true_turns = annotations
    detect = [x['end'] for x in detections]
    for turn_true in true_turns:
        closest_detection = min(detect, key=lambda x: abs(x - turn_true))
        if abs(closest_detection - turn_true) <= tolerance_ms:
            tp += 1
            latency.append(abs(closest_detection - turn_true))
        else:
            fn += 1
    print("detections = ", len(detections), "fn =", fn)
    fp = abs(len(detections) - tp)  # False positives
    print(tp, fp, fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    avg_latency = np.mean(latency) if latency else 0
    return precision, recall, avg_latency

ground_truth = readRTTM(ground_truth_path)
# Ground truth annotations (manual annotations)
# Format: [(start_ms, end_ms, speaker_id), ...]
print(ground_truth)


# BaselineModel: Standard VAD (Silero)
model = load_silero_vad()
wav = read_audio(path)
speech_timestamps = get_speech_timestamps(
  audio,
  model,
  return_seconds=True,  # Return speech timestamps in seconds (default is samples)
)
print(speech_timestamps)
precision_silero, recall_silero, latency_silero = evaluate(ground_truth, speech_timestamps)
print(precision_silero, recall_silero, latency_silero)


test_path = "../data/archive/voxconverse_dev_wav/audio/wmori.wav"
df = pd.read_csv("task1_test_groundtruth.csv")
annotation = df['1'].tolist()
print(type(annotation), annotation)
test_audio, sr = librosa.load(test_path, sr=16000, mono=True)
test_speech_timestamps = get_speech_timestamps(
  test_audio,
  model,
  return_seconds=True,  # Return speech timestamps in seconds (default is samples)
)
print(test_speech_timestamps)
precision_test, recall_test, latency_test = evaluate(annotation, test_speech_timestamps, manual_annotation=True)
print(precision_test, recall_test, latency_test)
metrics = {
    'Method': ['Silero VAD', 'Test Data'],
    'Precision': [precision_silero, precision_test],
    'Recall': [recall_silero, recall_test],
    'Latency (ms)': [latency_silero, latency_test]
}
print(metrics)
# Plot comparison
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].bar(metrics['Method'], metrics['Precision'], color='blue')
ax[0].set_title('Precision')
ax[1].bar(metrics['Method'], metrics['Recall'], color='green')
ax[1].set_title('Recall')
ax[2].bar(metrics['Method'], metrics['Latency (ms)'], color='red')
ax[2].set_title('Latency')
plt.show()
fig.savefig("Plot for Precision, recall & latency using Silero VAD")
