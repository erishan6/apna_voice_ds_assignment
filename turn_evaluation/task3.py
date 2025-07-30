import os
import librosa
from transformers import logging
from pipecat.audio.turn.smart_turn.local_coreml_smart_turn import LocalCoreMLSmartTurnAnalyzer
from pipecat.audio.turn.smart_turn.local_smart_turn import LocalSmartTurnAnalyzer
from PipecatCoreMLSmartTurnVADWrapper import PipecatCoreMLSmartTurnVADWrapper
import pandas as pd

# Load annotated dataset (simplified)
# Format: (audio_chunk, label) where label=1 for turn end

import asyncio

logging.set_verbosity(logging.INFO)

# Initialize model
test_path = "../data/archive/voxconverse_dev_wav/audio/szsyz.wav"
audio, sr = librosa.load(test_path, sr=16000)

# Original code:
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

# Initialize the VAD wrapper
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

chuck_size_list = [256, 512, 1024]
speech_timestamps_list = []
for chuck_size in chuck_size_list:
    speech_timestamps = asyncio.run(vad_wrapper.get_speech_timestamps(audio, chunk_size_samples=512))
    print("\n--- Detected Speech Segments ---")
    if not speech_timestamps:
        print("No speech detected.")
    for ts in speech_timestamps:
        start_sec = ts["start"] / 1000
        end_sec = ts["end"] / 1000
        print(f"Start: {ts['start']} ms ({start_sec:.2f}s), End: {ts['end']} ms ({end_sec:.2f}s)")
    speech_timestamps_list.append(speech_timestamps)

with open("task3_chunksize_test_timestamps.csv", "w+") as f:
    f.write("Start, End\n")
    i = 256
    for test_speech_timestamps in speech_timestamps_list:
        f.write("chunk ="+str(i)+"\n")
        f.write("Start, End\n")
        i = 2*i
        for x in test_speech_timestamps:
            f.write(str(x['start'])+", "+str(x["end"])+"\n")

