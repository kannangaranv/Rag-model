from openai import OpenAI
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import os
import tempfile
import shutil
import time

def extract_audio(video_path, audio_path):
    t0 = time.perf_counter()
    try:
        clip = VideoFileClip(video_path)
        audio = clip.audio
        if audio:
            audio.write_audiofile(audio_path, codec='libmp3lame')
            print(f"[OK] Audio extracted to {audio_path}")
        else:
            print("[WARN] No audio track found in the video.")
        clip.reader.close()
        clip.audio.reader.close_proc()
    except Exception as e:
        print(f"[ERR] Error extracting audio: {e}")
        raise
    finally:
        t1 = time.perf_counter()
        print(f"[TIME] Extract audio: {(t1 - t0):.2f}s")

def get_file_size(file_path):
    return os.path.getsize(file_path)

def split_audio(audio_path, chunk_dir, chunk_size=20 * 1024 * 1024):
    print("Splitting audio into chunks...")
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
    print(f"[OK] Generated {len(chunks)} chunks in {(t1 - t0):.2f}s "
          f"(est. chunk duration ~{chunk_duration_ms/1000:.1f}s)")
    return chunks, (t1 - t0)

def transcribe_audio_chunk(audio_path, client: OpenAI):
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1"
        )
    return response.text

def transcribe_audio(audio_path):
    print("Transcribing audio...")
    timings = {
        "split_seconds": 0.0,
        "chunks_count": 0,
        "per_chunk_seconds": [],
        "transcription_total_seconds": 0.0
    }

    chunk_dir = tempfile.mkdtemp(prefix="chunks_")
    chunks = []
    try:
        # split
        chunks, split_secs = split_audio(audio_path, chunk_dir)
        timings["split_seconds"] = split_secs
        timings["chunks_count"] = len(chunks)

        client = OpenAI()
        t_trans_total_start = time.perf_counter()

        transcript_parts = []
        for i, chunk in enumerate(chunks):
            t0 = time.perf_counter()
            print(f"[INFO] Transcribing chunk {i+1}/{len(chunks)}: {os.path.basename(chunk)}")
            text = transcribe_audio_chunk(chunk, client)
            t1 = time.perf_counter()
            dt = t1 - t0
            timings["per_chunk_seconds"].append(dt)
            print(f"[TIME] Chunk {i+1}: {dt:.2f}s")
            transcript_parts.append(text)

        t_trans_total_end = time.perf_counter()
        timings["transcription_total_seconds"] = t_trans_total_end - t_trans_total_start

        return " ".join(transcript_parts).strip(), timings

    finally:
        # Cleanup chunk dir
        try:
            if os.path.exists(chunk_dir):
                shutil.rmtree(chunk_dir)
        except Exception as e:
            print(f"[WARN] Could not delete chunk dir: {e}")

def format_seconds(sec: float) -> str:
    if sec < 60:
        return f"{sec:.2f}s"
    m, s = divmod(sec, 60)
    return f"{int(m)}m {s:.1f}s"

def get_transcription_from_video(video_path):
    """
    Run end-to-end: extract audio -> split -> transcribe -> save transcript.
    Prints a timing summary. Returns transcript text.
    """
    t_total_start = time.perf_counter()

    # temp audio file
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio_file:
        audio_path = temp_audio_file.name

    extract_secs = 0.0
    try:
        # reuse transcript if exists
        if os.path.exists('transcription.txt'):
            with open('transcription.txt', 'r', encoding='utf-8') as f:
                transcription = f.read()
            print("[INFO] Transcription loaded from file.")
            # no timings in this branch
            t_total_end = time.perf_counter()
            print("\n=== Timing Summary ===")
            print("Loaded existing transcription.txt (no processing).")
            print(f"End-to-end total: {format_seconds(t_total_end - t_total_start)}")
            return transcription

        # extract
        t0 = time.perf_counter()
        extract_audio(video_path, audio_path)
        t1 = time.perf_counter()
        extract_secs = t1 - t0

        # file size info
        file_size_mb = get_file_size(audio_path) / (1024 * 1024)
        print(f"[INFO] Temporary audio size: {file_size_mb:.2f} MB")

        # transcribe (includes split + per chunk)
        transcription, timings = transcribe_audio(audio_path)

        # save
        with open('transcription.txt', 'w', encoding='utf-8') as f:
            f.write(transcription)

        # summary
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
        # Cleanup temp audio
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            print(f"[WARN] Could not delete temp audio file: {e}")

# Run it and time it
get_transcription_from_video("C:\\Users\\nuwank\\Documents\\BoardPAC\\Development\\Rag-model\\dataset\\test1.mp4")
