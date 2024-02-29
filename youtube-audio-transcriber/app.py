import streamlit as st
from dotenv import load_dotenv
from gtts import gTTS 

load_dotenv() 
import os
import pyttsx3 
import google.generativeai as genai
import torch
from transformers import pipeline

from youtube_transcript_api import YouTubeTranscriptApi

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

prompt="""Act as an AI specializing in summarizing YouTube videos, your task is to distill the key points from the provided transcript text as paragraphs. Your summary should be concise, not exceeding 250 words, and should be structured in easy-to-understand paragraphs. 

The summary should cover all the important take aways from the video, including any main arguments, findings, or conclusions. The goal is to provide a comprehensive yet succinct overview of the video content, enabling users to understand the essence of the video without having to watch it in its entirety.

Please generate a summary for the following transcript text:  """

tamilprompt="Translate this to tamil language: "
def extract_transcript_details(youtube_video_url):
    try:
        video_id=youtube_video_url.split("=")[1]
        
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        raise e
def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)

    # Speech synthesis
    summary_text = response.text
    language = "en"  # Change language code if needed (e.g., "es" for Spanish)
    tts = gTTS(text=summary_text, lang=language,tld='co.in')
    tts.save("summary.mp3")  # Save audio to a file
    st.audio("summary.mp3")  # This plays the audio in Streamlit

    return response.text # Return text summary as well
# def gen_tamil(summary,tamilprompt):
#     model = genai.GenerativeModel("gemini-pro")
#     response = model.generate_content(tamilprompt + summary)
#     return response

st.title("YouTube Transcript to Detailed Notes Converter")
youtube_link = st.text_input("Enter YouTube Video Link:")


if st.button("Get Detailed Notes"):
    transcript_text=extract_transcript_details(youtube_link)


    if transcript_text:
        summary=generate_gemini_content(transcript_text,prompt)
        audio = "summary.mp3"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-tamil-medium", chunk_length_s=30, device=device)
        transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")

        print('Transcription: ', transcribe(audio)["text"])
        # tamil_sum = gen_tamil(summary,tamilprompt)
        st.markdown("## Detailed Notes:")
        # st.write(tamil_sum)
        st.write(summary)


