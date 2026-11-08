import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "analysis" / "transcripts"
WINDOW_SECONDS = 3.0
MIN_WINDOW_SECONDS = 1.0
PAUSE_SECONDS = 0.35
LOOKAHEAD_WORDS = 7
ANSWER_TAIL_SECONDS = 5.0
FILLERS = {"uh", "um", "uhm", "hmm", "like", "maybe", "probably", "around", "about", "guess", "think", "sure", "honestly", "actually", "so", "well"}

QUESTIONS = [
    ("name", "baseline", r"name\b"),
    ("grade", "baseline", r"(grade|year)\b"),
    ("major", "baseline", r"major\b"),
    ("state", "baseline", r"\w*state\b"),
    ("university", "baseline", r"(uni|university|school|college)\b|go to\b"),
    ("food", "baseline", r"(eat|ate|food)\b|last thing\b"),
    ("campus", "baseline", r"\w*campus\b"),
    ("class", "baseline", r"class\b"),
    ("gpa", "risky", r"(gpa|g\.p\.a\.?)( |$)"),
    ("screentime", "risky", r"screen"),
    ("sleep", "risky", r"sle(ep|pt)\b"),
    ("ai", "risky", r"(ai|a\.i\.?|chatgpt)( |$)"),
    ("dated", "risky", r"dated?\b"),
    ("bank", "risky", r"(bank|money)\b"),
]

ROTATIONS = {"IMG_1927": 270}


def token(word):
    return re.sub(r"[^a-z0-9.&]", "", word.lower()).strip(".")


def question_end(words, i):
    for j in range(i, min(i + LOOKAHEAD_WORDS, len(words))):
        if words[j]["word"].strip().endswith("?"):
            return j, True
        if j + 1 < len(words) and words[j + 1]["start"] - words[j]["end"] >= PAUSE_SECONDS:
            return j, False
    return i, False


def question_onset(words, i):
    k = i
    while k > 0 and i - k < 6 and words[k]["start"] - words[k - 1]["end"] < PAUSE_SECONDS and not words[k - 1]["word"].strip().endswith(("?", ".", "!")):
        k -= 1
    return words[k]["start"]


def find_questions(words):
    tokens = [token(w["word"]) for w in words]
    used = set()
    found = []
    for key, kind, pattern in QUESTIONS:
        candidates = []
        for i in range(len(words)):
            if i in used or not re.match(pattern, " ".join(tokens[i:i + 3])):
                continue
            end, punctuated = question_end(words, i)
            if any(j in used for j in range(i, end + 1)):
                continue
            candidates.append((not punctuated, i, end))
        if not candidates:
            print(f"  missing question '{key}'", file=sys.stderr)
            continue
        _, i, end = min(candidates)
        used.update(range(i, end + 1))
        found.append({
            "key": key,
            "kind": kind,
            "onset": question_onset(words, i),
            "end": words[end]["end"],
            "text": " ".join(w["word"].strip() for w in words[max(i - 4, 0):end + 1]),
        })
    return sorted(found, key=lambda q: q["end"])


def speech_metrics(words, start, stop):
    answer = [w for w in words if start <= w["start"] < stop]
    tokens = [re.sub(r"[^a-z]", "", w["word"].lower()) for w in answer]
    return {
        "latency": round((answer[0]["start"] if answer else stop) - start, 3),
        "answer_words": len(answer),
        "fillers": sum(t in FILLERS for t in tokens),
        "answer_span": round(answer[-1]["end"] - answer[0]["start"], 3) if answer else 0.0,
    }


def windows_for(questions, words):
    windows = []
    for i, q in enumerate(questions):
        next_onset = questions[i + 1]["onset"] if i + 1 < len(questions) else q["end"] + ANSWER_TAIL_SECONDS
        end = max(min(q["end"] + WINDOW_SECONDS, next_onset), q["end"] + MIN_WINDOW_SECONDS)
        windows.append({
            "question": q["key"],
            "kind": q["kind"],
            "start": round(q["end"], 2),
            "end": round(end, 2),
            "speech": speech_metrics(words, q["end"], next_onset),
        })
    return windows


def main():
    participants = []
    for transcript in sorted(TRANSCRIPTS.glob("*.json")):
        stem = transcript.stem
        print(stem, file=sys.stderr)
        words = json.loads(transcript.read_text())
        windows = windows_for(find_questions(words), words)
        video = next((ROOT / "videos").glob(f"{stem}.*"))
        participants.append({
            "id": stem.replace("Copy of ", ""),
            "video": video.relative_to(ROOT).as_posix(),
            "rotate": ROTATIONS.get(stem, 0),
            "windows": windows,
        })
    (ROOT / "participants.json").write_text(json.dumps({"participants": participants}, indent=2))
    print(f"wrote {len(participants)} participants", file=sys.stderr)


if __name__ == "__main__":
    main()
