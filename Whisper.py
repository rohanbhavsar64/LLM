import os
import time
import warnings
import torch
from faster_whisper import WhisperModel
import streamlit as st

class WhisperTranscriber:
    def __init__(self, model_size="large-v2"):
        warnings.filterwarnings("ignore")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        # Measure the time taken to load the model
        start_time = time.time()
        print("Loading Faster Whisper model...")
        # Use mixed precision (float16) by setting compute_type to 'float16'
        self.model = WhisperModel(model_size, device=self.device, compute_type="float16")
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds.")

    def transcribe_and_translate(self, audio_file, language_code):
        start_time = time.time()
        print("Transcribing and translating audio...")

        # Perform transcription and translation using the provided language code
        segments, info = self.model.transcribe(audio_file, beam_size=2, task="translate", language=language_code)
        transcription = "".join([segment.text for segment in segments])
        
        # Measure the time taken
        process_time = time.time() - start_time
        print(f"Process completed in {process_time:.2f} seconds.")
        
        return {"transcription": transcription, "language": language_code}

if __name__ == "__main__":
    detected_language = input("Select the language: ")  # Language code is provided as input
    transcriber = WhisperTranscriber(model_size="large-v2")

    while True:
        live_audio = st.file_uploader("Enter audio file (or type 'exit' to quit ")
        if live_audio.lower() == "exit":
            print("Exiting program.")
            break
        elif not os.path.exists(live_audio):
            print("File not found. Please enter a valid file path.")
        else:  
            total_start_time = time.time()
            result = transcriber.transcribe_and_translate(live_audio, detected_language)
            total_time = time.time() - total_start_time

            print("\n--- Speech-to-Text Results ---")
            st.markdown(f"Transcription: {result['transcription']}")
            print(f"Provided Language: {result['language']}")
            print(f"Total time for processing: {total_time:.2f} seconds.")
