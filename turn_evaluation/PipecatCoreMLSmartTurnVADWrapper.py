from typing import List, Dict, Any
import numpy as np
from loguru import logger
from pipecat.audio.turn.smart_turn.local_coreml_smart_turn import LocalCoreMLSmartTurnAnalyzer
import os
import librosa
import asyncio

# logger.disable("pipecat")
#
class PipecatCoreMLSmartTurnVADWrapper:
    def __init__(self, smart_turn_model_path: str, sampling_rate: int = 16000,
                 min_speech_duration_ms: int = 250, min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30, vad_stop_secs: float = 0.8,
                 completion_probability_threshold: float = 0.5):
        if sampling_rate != 16000:
            raise ValueError("Only 16kHz sampling rate is supported for LocalCoreMLSmartTurnAnalyzer.")
        self.analyzer = LocalCoreMLSmartTurnAnalyzer(smart_turn_model_path=smart_turn_model_path)
        self.sampling_rate = sampling_rate

        self.min_speech_duration_samples = int(min_speech_duration_ms * sampling_rate / 1000)
        self.speech_pad_samples = int(speech_pad_ms * sampling_rate / 1000)
        self.vad_stop_secs = vad_stop_secs
        self.completion_probability_threshold = completion_probability_threshold

        self._audio_buffer = np.array([], dtype=np.float32)
        self._current_speech_start_sample = None
        self._last_speech_activity_timestamp = 0.0
        self._total_processed_samples = 0
        self._last_turn_prediction_is_complete = False
        self._smart_turn_inference_chunk_size = 800

    def _simple_vad_check(self, audio_chunk: np.ndarray, threshold: float = 0.005) -> bool:
        return np.mean(np.abs(audio_chunk)) > threshold

    async def process_audio_chunk(self, audio_chunk: np.ndarray) -> List[Dict[str, int]]:
        detected_timestamps = []
        audio_chunk = audio_chunk.astype(np.float32)
        self._audio_buffer = np.concatenate((self._audio_buffer, audio_chunk))
        current_global_time_secs = (self._total_processed_samples + len(audio_chunk)) / self.sampling_rate

        is_current_chunk_active = self._simple_vad_check(audio_chunk)
        if is_current_chunk_active:
            self._last_speech_activity_timestamp = current_global_time_secs
            if self._current_speech_start_sample is None:
                self._current_speech_start_sample = max(0, self._total_processed_samples - self.speech_pad_samples)
                logger.debug(f"Speech START detected at sample {self._current_speech_start_sample}")
                self._last_turn_prediction_is_complete = False
        else:
            if self._current_speech_start_sample is not None:
                elapsed_silence_secs = current_global_time_secs - self._last_speech_activity_timestamp
                if elapsed_silence_secs >= self.vad_stop_secs:
                    end_sample = self._total_processed_samples + len(audio_chunk) + self.speech_pad_samples
                    if (end_sample - self._current_speech_start_sample) >= self.min_speech_duration_samples:
                        detected_timestamps.append({
                            "start": int(self._current_speech_start_sample / self.sampling_rate * 1000),
                            "end": int(end_sample / self.sampling_rate * 1000)
                        })
                        logger.debug(f"Speech ended detected")
                    self._current_speech_start_sample = None
                    self._audio_buffer = np.array([], dtype=np.float32)
                    self._last_turn_prediction_is_complete = False
                    self._last_speech_activity_timestamp = 0.0
                    self._total_processed_samples += len(audio_chunk)
                    return detected_timestamps

        if self._current_speech_start_sample is not None and len(self._audio_buffer) >= self._smart_turn_inference_chunk_size:
            segment_for_analysis = self._audio_buffer[-self._smart_turn_inference_chunk_size:]
            prediction_result = await self.analyzer._predict_endpoint(segment_for_analysis)
            self._last_turn_prediction_is_complete = (
                prediction_result["prediction"] == 1 and
                prediction_result["probability"] > self.completion_probability_threshold
            )
            if self._last_turn_prediction_is_complete:
                end_sample = self._total_processed_samples + len(audio_chunk) + self.speech_pad_samples
                if (end_sample - self._current_speech_start_sample) >= self.min_speech_duration_samples:
                    detected_timestamps.append({
                        "start": int(self._current_speech_start_sample / self.sampling_rate * 1000),
                        "end": int(end_sample / self.sampling_rate * 1000)
                    })
                self._current_speech_start_sample = None
                self._audio_buffer = np.array([], dtype=np.float32)
                self._last_turn_prediction_is_complete = False
                self._last_speech_activity_timestamp = 0.0

        self._total_processed_samples += len(audio_chunk)
        return detected_timestamps

    async def flush(self) -> List[Dict[str, int]]:
        detected_timestamps = []
        if self._current_speech_start_sample is not None:
            end_sample = self._total_processed_samples + self.speech_pad_samples
            if (end_sample - self._current_speech_start_sample) >= self.min_speech_duration_samples:
                detected_timestamps.append({
                    "start": int(self._current_speech_start_sample / self.sampling_rate * 1000),
                    "end": int(end_sample / self.sampling_rate * 1000)
                })
            self._current_speech_start_sample = None
            self._audio_buffer = np.array([], dtype=np.float32)
            self._last_turn_prediction_is_complete = False
            self._last_speech_activity_timestamp = 0.0
        return detected_timestamps

    async def get_speech_timestamps(self, audio_array: np.ndarray, chunk_size_samples: int = 256) -> List[Dict[str, int]]:
        all_timestamps = []
        self._audio_buffer = np.array([], dtype=np.float32)
        self._current_speech_start_sample = None
        self._last_speech_activity_timestamp = 0.0
        self._total_processed_samples = 0
        self._last_turn_prediction_is_complete = False
        print(len(audio_array))
        for i in range(0, len(audio_array), chunk_size_samples):
            chunk = audio_array[i: i + chunk_size_samples]
            if len(chunk) == 0:
                continue
            detected_timestamps = await self.process_audio_chunk(chunk)
            all_timestamps.extend(detected_timestamps)

        final_timestamps = await self.flush()
        all_timestamps.extend(final_timestamps)
        return all_timestamps


import os
import asyncio
import numpy as np

async def main():
    # Load path to Smart Turn model from environment variable
    smart_turn_model_path = os.getenv("LOCAL_SMART_TURN_MODEL_PATH")
    if not smart_turn_model_path:
        print("Error: LOCAL_SMART_TURN_MODEL_PATH environment variable not set.")
        print("Please clone https://github.com/pipecat-ai/smart-turn and set the variable.")
        print("Example: export LOCAL_SMART_TURN_MODEL_PATH=\"/Users/youruser/Documents/smart-turn\"")
        return

    path = "../data/archive/voxconverse_dev_wav/audio/ahnss.wav"
    ground_truth_path = "../data/archive/labels/dev/ahnss.rttm"
    audio, sr = librosa.load(path, sr=16000, mono=True)

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
    speech_timestamps = await vad_wrapper.get_speech_timestamps(audio, chunk_size_samples=512)

    print("\n--- Detected Speech Segments ---")
    if not speech_timestamps:
        print("No speech detected.")
    for ts in speech_timestamps:
        start_sec = ts["start"] / 1000
        end_sec = ts["end"] / 1000
        print(f"Start: {ts['start']} ms ({start_sec:.2f}s), End: {ts['end']} ms ({end_sec:.2f}s)")

if __name__ == "__main__":
    asyncio.run(main())
