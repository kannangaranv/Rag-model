from openai import OpenAI
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import os
import tempfile
import shutil
import time 
from pathlib import Path

def extract_audio(video_path, audio_path):
    try:
        clip = VideoFileClip(video_path)
        audio = clip.audio
        if audio:
            audio.write_audiofile(audio_path, codec='libmp3lame')
            print(f"Audio extracted to {audio_path}")
        else:
            print("No audio track found in the video.")
        clip.reader.close()
        clip.audio.reader.close_proc()
    except Exception as e:
        print(f"Error extracting audio: {e}")
        raise

def get_file_size(file_path):
    return os.path.getsize(file_path)

def split_audio(audio_path, chunk_dir, chunk_size=20 * 1024 * 1024):
    t0 = time.perf_counter()
    os.makedirs(chunk_dir, exist_ok=True)
    chunks = []

    audio = AudioSegment.from_file(audio_path)
    total_size = get_file_size(audio_path)
    total_duration_ms = len(audio)  
    chunk_duration_ms = int(chunk_size / total_size * total_duration_ms)
    if chunk_duration_ms >= total_duration_ms:
        chunk_duration_ms = total_duration_ms 

    for i in range(0, total_duration_ms, chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        idx = i // chunk_duration_ms
        chunk_path = os.path.join(chunk_dir, f"chunk_{idx:03d}.mp3")
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)

    t1 = time.perf_counter()
    return chunks, (t1 - t0)

def transcribe_audio_chunk(audio_path, client: OpenAI):
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1"
        )
    return response.text

def transcribe_audio(audio_path):
    timings = {
        "split_seconds": 0.0,
        "chunks_count": 0,
        "per_chunk_seconds": [],
        "transcription_total_seconds": 0.0
    }

    chunk_dir = tempfile.mkdtemp(prefix="chunks_")
    chunks = []
    try:
        chunks, split_secs = split_audio(audio_path, chunk_dir)
        timings["split_seconds"] = split_secs
        timings["chunks_count"] = len(chunks)

        client = OpenAI()
        t_trans_total_start = time.perf_counter()

        transcript_parts = []
        for i, chunk in enumerate(chunks):
            t0 = time.perf_counter()
            print(f"Transcribing chunk {i+1}/{len(chunks)}: {os.path.basename(chunk)}")
            text = transcribe_audio_chunk(chunk, client)
            t1 = time.perf_counter()
            dt = t1 - t0
            timings["per_chunk_seconds"].append(dt)
            transcript_parts.append(text)

        t_trans_total_end = time.perf_counter()
        timings["transcription_total_seconds"] = t_trans_total_end - t_trans_total_start

        return " ".join(transcript_parts).strip(), timings

    finally:
        try:
            if os.path.exists(chunk_dir):
                shutil.rmtree(chunk_dir)
        except Exception as e:
            print(f"Could not delete chunk dir: {e}")

def format_seconds(sec: float) -> str:
    if sec < 60:
        return f"{sec:.2f}s"
    m, s = divmod(sec, 60)
    return f"{int(m)}m {s:.1f}s"

def get_transcription_from_video(video_path: str | Path):
    video_path = os.fspath(video_path)
    t_total_start = time.perf_counter()

    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio_file:
        audio_path = temp_audio_file.name

    extract_secs = 0.0
    try:
        t0 = time.perf_counter()
        extract_audio(video_path, audio_path)
        t1 = time.perf_counter()
        extract_secs = t1 - t0

        file_size_mb = get_file_size(audio_path) / (1024 * 1024)
        print(f"Temporary audio size: {file_size_mb:.2f} MB")

        transcription, timings = transcribe_audio(audio_path)

        t_total_end = time.perf_counter()
        end_to_end = t_total_end - t_total_start

        print("\n=== Timing Summary ===")
        print(f"Extract audio:            {format_seconds(extract_secs)}")
        print(f"Split audio:              {format_seconds(timings['split_seconds'])} "
              f"(chunks: {timings['chunks_count']})")
        if timings["per_chunk_seconds"]:
            print("Per-chunk transcription:")
            for i, secs in enumerate(timings["per_chunk_seconds"], start=1):
                print(f"  - Chunk {i:02d}:        {format_seconds(secs)}")
        print(f"Transcription total:      {format_seconds(timings['transcription_total_seconds'])}")
        print(f"End-to-end total:         {format_seconds(end_to_end)}")

        return transcription

    finally:
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            print(f"Could not delete temp audio file: {e}")