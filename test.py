from moviepy.editor import *
import speech_recognition as sr
import multiprocessing
import numpy as np
from pocketsphinx import pocketsphinx


# Function to convert video to audio
def video_to_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)

# Function to extract text from audio using CMU Sphinx
def audio_to_text(audio_path, text_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_sphinx(audio_data)
        with open(text_path, 'w') as file:
            file.write(text)

# Function for parallel processing of audio transcription
def parallel_transcription(audio_paths, text_paths):
    processes = []
    for audio_path, text_path in zip(audio_paths, text_paths):
        process = multiprocessing.Process(target=audio_to_text, args=(audio_path, text_path))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

# Main function
def main(video_path, output_dir):
    audio_path = f"{output_dir}/audio.wav"
    text_path = f"{output_dir}/text.txt"

    # Convert video to audio
    video_to_audio(video_path, audio_path)
    
    # Split audio into smaller chunks
    chunk_size = 30  # seconds
    audio_paths = []
    text_paths = []
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    for i, start_time in enumerate(np.arange(0, duration, chunk_size)):
        end_time = min(start_time + chunk_size, duration)
        chunk_audio_path = f"{output_dir}/audio_chunk_{i}.wav"
        chunk_text_path = f"{output_dir}/text_chunk_{i}.txt"
        chunk = audio_clip.subclip(start_time, end_time)
        chunk.write_audiofile(chunk_audio_path)
        audio_paths.append(chunk_audio_path)
        text_paths.append(chunk_text_path)

    # Perform parallel transcription
    parallel_transcription(audio_paths, text_paths)

    # Concatenate text from chunks
    with open(text_path, 'w') as output_file:
        for chunk_text_path in text_paths:
            with open(chunk_text_path, 'r') as input_file:
                output_file.write(input_file.read())

if __name__ == "__main__":
    video_path = "D:\Facial Emotion detection\Videos\presidential_debate.mp4"
    output_dir = "output"
    main(video_path, output_dir)
