import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from faster_whisper import WhisperModel
from scipy.io import wavfile

videos = sorted(Path("videos").glob("*.MOV"))
out_dir = Path("analysis/transcripts")
out_dir.mkdir(parents=True, exist_ok=True)
model = WhisperModel("small.en", device="cpu", compute_type="int8")

for video in videos:
    target = out_dir / f"{video.stem}.json"
    if target.exists():
        continue
    wav = out_dir / f"{video.stem}.wav"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-map", "0:a:0", "-ac", "1", "-ar", "16000", str(wav)], check=True)
    rate, samples = wavfile.read(wav)
    segments, _ = model.transcribe(samples.astype("float32") / 32768, word_timestamps=True, vad_filter=False)
    words = [{"start": w.start, "end": w.end, "word": w.word} for s in segments for w in s.words]
    target.write_text(json.dumps(words, indent=0))
    wav.unlink()
    print(video.name, len(words), flush=True)
