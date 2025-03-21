"""_summary_
Speech to text using Silero VAD, Whisper and Porcupine (Picovoice) for keyword detection.
"""
import os
import struct
import math
import time
import datetime
import threading
import warnings
import wave
import numpy as np
import rospy
import rospkg

import pyaudio
from playsound import playsound
import whisper
import openai
import pvporcupine
from silero_vad import load_silero_vad, VADIterator

from std_msgs.msg import String

# Suppress warnings from whisper (if any)
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

# Get the absolute path of the current file directory
# TODO: need to change to relative path...
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
USER_ID = "user001"

PICO_ACCESS_KEY = "YOUR_PICO"

def get_incremented_recording_path(directory, base_name="curr_recording", ext="wav"):
    """
    Returns a unique file path by incrementing the filename if needed.
    Example: curr_recording.wav, curr_recording_1.wav, curr_recording_2.wav, ...
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, f"{base_name}.{ext}")
    counter = 1
    while os.path.exists(path):
        path = os.path.join(directory, f"{base_name}_{counter}.{ext}")
        counter += 1
    return path

class SpeechTranscriber():
    def __init__(self):
        # Initialize ROS node and publisher.
        rospy.init_node('speech_to_text_node')
        self.pub = rospy.Publisher("user_preference", String, queue_size=10)

        # Audio parameters.
        self.CHANNELS = 1
        self.SAMPLE_FORMAT = pyaudio.paInt16

        # Initialize Porcupine wake-word detector.
        if PICO_ACCESS_KEY == "YOUR_PICO":
            raise ValueError("Please set your Picovoice access key.")
        
        self.porcupine = pvporcupine.create(
            access_key=PICO_ACCESS_KEY,
            keyword_paths=[os.path.join(BASE_PATH, '../../Hey-Frank_en_linux_v3_0_0.ppn')],
            sensitivities=[0.5]
        )
        self._sampling_rate = self.porcupine.sample_rate    # 16000 Hz
        self._vad_chunk_size = 512  # to match the sampling rate

        # Initialize PyAudio instance.
        self.pa_instance = pyaudio.PyAudio()
        self.mic_index = self.select_microphone(self.pa_instance)
        self.audio_stream = self.initialize_audio_stream(self.mic_index, self.porcupine.frame_length)

        # Load Silero VAD model and create a VADIterator.
        self.vad_model = load_silero_vad(onnx=True)
        self.vad_iterator = VADIterator(
            model=self.vad_model,
            sampling_rate=self._sampling_rate,
            threshold=0.5,  # Adjust silence threshold as needed
            min_silence_duration_ms=300,
        )

    def select_microphone(self, pa_instance):
        """
        Select an appropriate microphone based on device name.
        """
        mic_index = -1
        for i in range(pa_instance.get_device_count()):
            device_name = pa_instance.get_device_info_by_index(i)['name']
            rospy.loginfo("Found device: %s", device_name)
            if 'USB PHY' in device_name or 'Microphone' in device_name:
                mic_index = i
                break
        if mic_index == -1:
            raise RuntimeError("No microphone found.")
        rospy.loginfo("Selected mic: %s", pa_instance.get_device_info_by_index(mic_index)['name'])
        return mic_index

    def initialize_audio_stream(self, mic_index, frames_per_buffer):
        """
        Initializes and returns a PyAudio stream.
        """
        return self.pa_instance.open(
            rate=self._sampling_rate,
            channels=self.CHANNELS,
            format=self.SAMPLE_FORMAT,
            input=True,
            frames_per_buffer=frames_per_buffer,
            input_device_index=mic_index
        )

    def record_audio_with_vad(self, max_speech_secs=15, silence_timeout=1.5):
        """
        Records audio from the current stream using Silero VAD.
        Returns a list of audio frames (bytes).
        """
        frames = []
        recording = True
        silence_start = None
        start_time = time.time()

        while recording and not rospy.is_shutdown():
            # Use a chunk size of 512, as expected by Silero VAD for 16000 Hz.
            data = self.audio_stream.read(self._vad_chunk_size, exception_on_overflow=False)
            frames.append(data)

            # Run VAD on the raw chunk (convert bytes to numpy float32)
            # Here we assume 16-bit little-endian.
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # Process the chunk with Silero VAD.
            vad_result = self.vad_iterator(audio_chunk)

            # Compute the total elapsed recording time in seconds (assuming 2 bytes per sample, 1 channel)
            elapsed = len(b''.join(frames)) / (self._sampling_rate * 2)

            if silence_start is not None:
                # print("silence_start:", silence_start)
                if time.time() - silence_start >= silence_timeout:
                    recording = False
            
            # Check VAD events
            if vad_result and "end" in vad_result:
                # If we detect an "end" event, start or continue counting silence.
                if silence_start is None:
                    silence_start = time.time()
            elif vad_result and "start" in vad_result:
                silence_start = None
            
            # Stop if max speech duration is reached
            elapsed = len(b''.join(frames)) / (self._sampling_rate * 2)  # 2 bytes per sample, 1 channel
            if elapsed >= max_speech_secs:
                recording = False
                rospy.logwarn(f"Max speech duration reached. Stopping recording. Elapsed: {elapsed:.2f}s")

        return frames
    
    def save_and_transcribe(self, frames):
        """
        Saves recorded frames to a WAV file, plays a sound, transcribes with Whisper,
        and returns the transcription.
        """
        recording_path = get_incremented_recording_path(os.path.join(BASE_PATH), USER_ID)
        rospy.loginfo("Finished recording. Saving to %s", recording_path)

        # Play a sound in a separate thread.
        sound_path = os.path.join(BASE_PATH, "sounds", "negative_short.wav")
        sound_thread = threading.Thread(target=playsound, args=(sound_path,))
        sound_thread.start()

        # Ensure the directory exists.
        recording_dir = os.path.dirname(recording_path)
        os.makedirs(recording_dir, exist_ok=True)

        # Save WAV file.
        with wave.open(recording_path, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.pa_instance.get_sample_size(self.SAMPLE_FORMAT))
            wf.setframerate(self._sampling_rate)
            wf.writeframes(b''.join(frames))

        sound_thread.join()

        # Transcribe using Whisper.
        rospy.loginfo("Transcribing audio...")
        model = whisper.load_model("turbo")
        transcription = model.transcribe(recording_path, fp16=False)["text"]
        rospy.loginfo("Transcription: %s", transcription)
        return transcription

    def run(self):
        """
        Main loop: listen for the wake word, record speech, transcribe it, and publish the result.
        """
        self.audio_stream.start_stream()
        rospy.loginfo("Speech-to-text node READY")

        while not rospy.is_shutdown():
            # Read a frame for wake word detection.
            data = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, data)
            keyword_index = self.porcupine.process(pcm)

            if keyword_index == 0:
                # Play beep sound in a separate thread.
                beep_path = os.path.join(BASE_PATH, "sounds", "positive_short.wav")
                beep_thread = threading.Thread(target=playsound, args=(beep_path,))
                beep_thread.start()

                rospy.loginfo('Wake word detected. Please start speaking.')

                # Record using Silero VAD.
                frames = self.record_audio_with_vad()

                beep_thread.join()

                # Stop stream to process the recording.
                self.audio_stream.stop_stream()
                transcription = self.save_and_transcribe(frames)
                self.pub.publish(transcription)
                rospy.loginfo("Published transcription: %s", transcription)


                # Reinitialize the audio stream for wake word detection.
                self.audio_stream.start_stream()
                rospy.loginfo("Speech-to-text node READY")

        self.audio_stream.close()
        self.pa_instance.terminate()

if __name__ == "__main__":
    try:
        stt_node = SpeechTranscriber()
        stt_node.run()
    except rospy.ROSInterruptException:
        stt_node.audio_stream.close()
        stt_node.pa_instance.terminate()
        pass
