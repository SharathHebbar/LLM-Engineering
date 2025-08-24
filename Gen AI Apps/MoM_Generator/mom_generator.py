import streamlit as st

import whisper
import numpy as np
import time
from pydub import AudioSegment

# Load Whisper model
model = whisper.load_model("base")

from openai import OpenAI
model_name = "medgemma-4b-it"
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

st.header("MoM Generator using Whisper and MedGemma")

system_prompt = """
Role: You are an expert assistant specialized in generating professional Minutes of the Meeting (MoM) from meeting transcripts.

Objective: Your task is to extract and organize critical insights, decisions, action points, and relevant discussions from the transcript into a clear, structured MoM format suitable for stakeholders.

Guidelines:

1. Identify and highlight the key discussion points, decisions made, and any follow-ups mentioned.
2. Clearly list the following components in the MoM:
   - Meeting Title and Date (if available)
   - Attendees (if mentioned)
   - Summary of Discussions
   - Key Objectives and Outcomes
   - Action Items with Assignees and Deadlines
   - Open Questions or Pending Items
   - Any follow-up meeting or review dates

Formatting Requirements:

- Use bullet points or numbered lists for clarity.
- Keep the tone professional, concise, and informative.
- Group related points together by topic or agenda item.
- Ensure action items are clear, with responsible persons and due dates if mentioned.

Constraints:

- Do not add any information not present in the transcript.
- If any information is missing (e.g., deadlines or assignees), flag it appropriately (e.g., “TBD”).

Your output should be directly usable as formal MoM documentation.
"""


# chunk_duration_ms = st.sidebar()

if audio_file := st.text_input(label="Enter Audio Filename", placeholder="Enter the input path....."):
    chunk_duration_ms = st.number_input(label="Enter the chunk duration in ms for streaming.", step=1000, min_value=0, max_value=60000)
    if chunk_duration_ms > 0 :
        st.write(audio_file, chunk_duration_ms)
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_channels(1).set_frame_rate(16000)
        num_chunks = len(audio) // chunk_duration_ms
        st.write(f"🔊 Total audio length: {len(audio)/1000:.2f} seconds")
        
        transcript = ""
        progress_text = "Transcription in progress..."
        my_bar = st.progress(0, text=progress_text)
        
        with st.spinner(text="Transcribing.....", show_time=True):
        
            for i in range(num_chunks):
                my_bar.progress(i + 1, text = progress_text)
                chunk = audio[i * chunk_duration_ms: (i + 1) * chunk_duration_ms]
                samples = np.array(chunk.get_array_of_samples()).astype(np.float32) / 32768.0
                result = model.transcribe(samples, fp16=False, language="en")
                # st.text(result['text'].strip())
                time.sleep(1)
                transcript += result['text'].strip()
        
        with st.spinner(text="Generating MoM.....", show_time=True):
            if not transcript == "":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": transcript}
                    ],
                )

                st.markdown(response.choices[0].message.content)
