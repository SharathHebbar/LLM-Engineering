import streamlit as st

import whisper
import numpy as np
import time
from pydub import AudioSegment

# Load Whisper model
model = whisper.load_model("base")

st.header("🔊 Audio Streaming using Whisper")

    

# chunk_duration_ms = st.sidebar()

if audio_file := st.text_input(label="Enter Audio Filename", placeholder="Enter the input path....."):
    chunk_duration_ms = st.number_input(label="Enter the chunk duration in ms for streaming.", step=1000, min_value=0, max_value=60000)
    if chunk_duration_ms > 0 :
        st.write(audio_file, chunk_duration_ms)


        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_channels(1).set_frame_rate(16000)

        # chunk_duration_ms = 30000  # 5 seconds
        num_chunks = len(audio) // chunk_duration_ms

        # print(f"🔊 Total audio length: {len(audio)/1000:.2f} seconds")
        # print("🎧 Streaming...")
        st.write(f"🔊 Total audio length: {len(audio)/1000:.2f} seconds")

        for i in range(num_chunks):
            chunk = audio[i * chunk_duration_ms: (i + 1) * chunk_duration_ms]
            
            # Convert to numpy float32 array
            samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0

            # Transcribe using Whisper
            result = model.transcribe(samples, fp16=False, language="en")
            # print(f"🕒 Chunk {i+1}: {result['text'].strip()}")
            st.text(result['text'].strip())

            time.sleep(1)  # simulate real-time delay
