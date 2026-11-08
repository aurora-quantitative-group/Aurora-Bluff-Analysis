import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline

FEAR_AUS = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU07_r", "AU20_r", "AU26_r"]
MIN_CONFIDENCE = 0.80
MIN_FRAMES = 10
AU20_THRESHOLD = 0.10
TRANSIENT_MAX_MS = 500
CLIP_FPS = 30
CLIP_MAX_WIDTH = 1920
ROTATION_FILTERS = {0: [], 90: ["transpose=1"], 180: ["hflip", "vflip"], 270: ["transpose=2"]}

FEATURE_WEIGHTS = {
    "delta_eyebrow_coherence": 2.0,
    "delta_AU20_r_mean": 1.8,
    "delta_AU05_r_mean": 1.8,
    "delta_au20_avg_duration": 1.5,
    "delta_au20_transient_count": 1.5,
}
NIU_FEATURES = ["AU05_r_mean", "AU20_r_mean", "eyebrow_coherence", "au20_avg_duration", "au20_transient_count"]


def find_binary(env_var, *names):
    if os.environ.get(env_var):
        return os.environ[env_var]
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def imageio_ffmpeg_binary():
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


FFMPEG = find_binary("FFMPEG_BIN", "ffmpeg", "ffmpeg.exe") or imageio_ffmpeg_binary()
OPENFACE = find_binary("OPENFACE_BIN", "FeatureExtraction", "FeatureExtraction.exe")


def run(cmd):
    result = subprocess.run([str(part) for part in cmd], capture_output=True, text=True)
    if result.returncode != 0:
        output = (result.stderr or result.stdout)[-2000:]
        raise RuntimeError(f"{Path(cmd[0]).name} exited with {result.returncode}:\n{output}")


def process_video_segment(video_path, start_time, end_time, output_dir, rotate=0, keep_files=False):
    if not FFMPEG or not OPENFACE:
        raise RuntimeError("ffmpeg and OpenFace FeatureExtraction are required (PATH, FFMPEG_BIN or OPENFACE_BIN)")
    if end_time <= start_time:
        raise ValueError(f"{video_path}: window end {end_time} is not after start {start_time}")

    work_dir = Path(output_dir) / f"{Path(video_path).stem}_{start_time:.2f}-{end_time:.2f}_{uuid.uuid4().hex[:6]}"
    work_dir.mkdir(parents=True)
    clip = work_dir / "clip.mp4"
    filters = ROTATION_FILTERS[rotate % 360] + [f"fps={CLIP_FPS}", f"scale='min({CLIP_MAX_WIDTH},iw)':-2"]

    try:
        run([
            FFMPEG, "-y", "-loglevel", "error",
            "-ss", f"{start_time:.3f}", "-i", video_path, "-t", f"{end_time - start_time:.3f}",
            "-an", "-vf", ",".join(filters),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            clip,
        ])
        run([OPENFACE, "-f", clip, "-out_dir", work_dir, "-aus", "-gaze", "-pose", "-2Dfp"])
        df = pd.read_csv(work_dir / "clip.csv", skipinitialspace=True)
    finally:
        if not keep_files:
            shutil.rmtree(work_dir, ignore_errors=True)

    df.columns = df.columns.str.strip()
    return df


def reliable_frames(df):
    frames = df[(df["success"] == 1) & (df["confidence"] >= MIN_CONFIDENCE)]
    if len(frames) < MIN_FRAMES:
        raise ValueError(f"only {len(frames)} frames passed the OpenFace confidence filter")
    return frames


def au20_event_durations(frames):
    active = frames["AU20_r"].to_numpy() > AU20_THRESHOLD
    frame_numbers = frames["frame"].to_numpy()
    windows = frames["window"].to_numpy() if "window" in frames else np.zeros(len(frames))

    boundaries = np.r_[
        True,
        (active[1:] != active[:-1]) | (np.diff(frame_numbers) != 1) | (np.diff(windows) != 0),
    ]
    starts = np.flatnonzero(boundaries)
    lengths = np.diff(np.r_[starts, len(active)])
    return lengths[active[starts]]


def eyebrow_coherence(frames):
    windows = frames["window"] if "window" in frames else pd.Series(0, index=frames.index)
    left = frames["y_38"] - frames["y_20"]
    right = frames["y_44"] - frames["y_24"]
    left = left - left.groupby(windows).transform("mean")
    right = right - right.groupby(windows).transform("mean")

    if len(left) < 3 or left.std() == 0 or right.std() == 0:
        return np.nan
    r, _ = pearsonr(left, right)
    return r


def extract_fear_metrics(df, fps=30):
    frames = reliable_frames(df)

    metrics = {}
    for au in FEAR_AUS:
        metrics[f"{au}_mean"] = frames[au].mean()
        metrics[f"{au}_std"] = frames[au].std(ddof=0)

    durations = au20_event_durations(frames)
    metrics["au20_event_count"] = len(durations)
    metrics["au20_avg_duration"] = durations.mean() / fps * 1000 if len(durations) else 0.0
    metrics["au20_transient_count"] = int((durations <= round(TRANSIENT_MAX_MS / 1000 * fps)).sum())
    metrics["eyebrow_coherence"] = eyebrow_coherence(frames)
    return metrics


def extract_behavior_metrics(df, fps=30):
    frames = reliable_frames(df)
    seconds = len(frames) / fps
    head = frames[["pose_Rx", "pose_Ry", "pose_Rz"]]
    blink_onsets = (frames["AU45_c"].diff() > 0).sum() + int(frames["AU45_c"].iloc[0] > 0)

    return {
        "gaze_x_std": frames["gaze_angle_x"].std(ddof=0),
        "gaze_y_std": frames["gaze_angle_y"].std(ddof=0),
        "gaze_y_mean": frames["gaze_angle_y"].mean(),
        "head_pitch_std": frames["pose_Rx"].std(ddof=0),
        "head_yaw_std": frames["pose_Ry"].std(ddof=0),
        "head_roll_std": frames["pose_Rz"].std(ddof=0),
        "head_speed": head.diff().abs().sum(axis=1).mean() * fps,
        "blink_rate": blink_onsets / seconds,
    }


def estimate_fps(df):
    if "timestamp" not in df:
        return CLIP_FPS
    steps = df.groupby("window")["timestamp"].diff() if "window" in df else df["timestamp"].diff()
    steps = steps[steps > 0]
    return float(round(1 / steps.median())) if len(steps) else CLIP_FPS


def window_metrics(df):
    fps = estimate_fps(df)
    return extract_fear_metrics(df, fps) | extract_behavior_metrics(df, fps)


def load_window(source, window, output_dir):
    rotate = source.get("rotate", 0)
    cache = Path(output_dir) / f"{Path(source['video']).stem}_{window['start']:.2f}-{window['end']:.2f}_r{rotate}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    df = process_video_segment(source["video"], window["start"], window["end"], output_dir, rotate)
    df.to_csv(cache, index=False)
    return df


def load_segment(segment, output_dir, loader=load_window):
    clips = [loader(segment, window, output_dir).assign(window=i) for i, window in enumerate(segment["windows"])]
    return pd.concat(clips, ignore_index=True)


def build_participant_delta_vector(baseline_mov_config, risky_mov_config, output_dir="openface_cache", loader=load_window):
    baseline = window_metrics(load_segment(baseline_mov_config, output_dir, loader))
    risky = window_metrics(load_segment(risky_mov_config, output_dir, loader))

    vector = {f"delta_{name}": risky[name] - value for name, value in baseline.items()}
    vector.update({f"baseline_{name}": value for name, value in baseline.items()})
    return pd.Series(vector, dtype=float)


def build_window_table(participants, output_dir, loader, workers=6):
    jobs = [(p, w) for p in participants for w in p["windows"]]

    def measure(job):
        participant, window = job
        try:
            return window_metrics(loader(participant, window, output_dir)) | window.get("speech", {})
        except (RuntimeError, ValueError) as error:
            print(f"Skipping {participant['id']} / {window['question']}: {error}", file=sys.stderr)
            return None

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(measure, jobs))

    rows = [
        {
            "participant": p["id"],
            "question": w["question"],
            "kind": w["kind"],
            "label": int(w.get("label", w["kind"] == "risky")),
            **metrics,
        }
        for (p, w), metrics in zip(jobs, results)
        if metrics is not None
    ]
    return pd.DataFrame(rows)


def baseline_normalize(table):
    metric_columns = [c for c in table.columns if c not in ("participant", "question", "kind", "label")]
    rows = []
    for _, group in table.groupby("participant", sort=False):
        baseline = group.loc[group["kind"] == "baseline", metric_columns]
        for index, row in group.iterrows():
            reference = baseline.drop(index=index, errors="ignore").mean()
            delta = row[metric_columns].astype(float) - reference
            rows.append(pd.concat([delta.add_prefix("delta_"), reference.add_prefix("baseline_")]).rename(index))
    return pd.DataFrame(rows).loc[table.index]


def train_loocv_classifier(X, y, groups=None, model_path="deception_rf_model.pkl"):
    counts = y.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError(f"cross-validation needs at least two samples per class, got {counts.to_dict()}")

    weights = pd.Series({column: FEATURE_WEIGHTS.get(column, 1.0) for column in X.columns})
    X_weighted = X * weights

    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, class_weight="balanced"),
    )
    cv = LeaveOneGroupOut() if groups is not None else LeaveOneOut()
    probabilities = cross_val_predict(model, X_weighted, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    scores = {
        "roc_auc": roc_auc_score(y, probabilities),
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "kappa": cohen_kappa_score(y, predictions),
    }

    model.fit(X_weighted, y)
    importances = pd.Series(model[-1].feature_importances_, index=X.columns).sort_values(ascending=False)

    scheme = f"leave-one-participant-out, {groups.nunique()} folds" if groups is not None else "leave-one-out"
    print(f"\nCross-validation results ({scheme}, {len(y)} samples, chance = {counts.max() / len(y):.3f})")
    for name, value in scores.items():
        print(f"  {name:<10}{value:.3f}")
    print("\nConfusion matrix (rows = truth, cols = predicted; 0 = baseline/truthful, 1 = risky/deceptive)")
    print(confusion_matrix(y, predictions, labels=[0, 1]))
    print("\nGini feature importances (top 20)")
    for rank, (feature, value) in enumerate(importances.head(20).items(), start=1):
        print(f"  {rank:>2}. {feature:<32}{value:.4f}")

    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": list(X.columns), "weights": weights.to_dict()}, f)
    print(f"\nSaved model to {model_path}")

    return {"scores": scores, "probabilities": probabilities, "importances": importances, "model": model}


def niu_effect_report(participants, output_dir, loader):
    deltas = []
    for p in participants:
        segments = {
            kind: {**p, "windows": [w for w in p["windows"] if w["kind"] == kind]}
            for kind in ("baseline", "risky")
        }
        try:
            deltas.append(build_participant_delta_vector(segments["baseline"], segments["risky"], output_dir, loader))
        except (RuntimeError, ValueError) as error:
            print(f"Skipping {p['id']} in effect report: {error}", file=sys.stderr)
    deltas = pd.DataFrame(deltas)

    print(f"\nRisky vs baseline shifts across {len(deltas)} participants (Niu et al. features)")
    print(f"  {'feature':<24}{'mean delta':>12}{'d_z':>8}{'p':>8}")
    for feature in NIU_FEATURES:
        values = deltas[f"delta_{feature}"].dropna()
        d_z = values.mean() / values.std() if values.std() > 0 else np.nan
        p = wilcoxon(values).pvalue if (values != 0).sum() > 1 else np.nan
        print(f"  {feature:<24}{values.mean():>12.4f}{d_z:>8.2f}{p:>8.3f}")


def load_participants(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    for p in config["participants"]:
        p["video"] = config_path.parent / p["video"]
    return config["participants"]


def synthetic_window(n_frames, rng, stressed, expressiveness):
    df = pd.DataFrame({"frame": np.arange(1, n_frames + 1), "timestamp": np.arange(n_frames) / CLIP_FPS})
    df["success"] = (rng.random(n_frames) > 0.02).astype(int)
    df["confidence"] = rng.uniform(0.7, 1.0, n_frames)

    for au in FEAR_AUS:
        df[au] = expressiveness * rng.gamma(2.0, 0.12, n_frames)
    if stressed:
        df["AU05_r"] *= 0.55

    au20 = np.abs(rng.normal(0, 0.03, n_frames))
    burst_frames, burst_amplitude = ((3, 12), (0.2, 0.6)) if stressed else ((14, 30), (0.5, 1.2))
    for _ in range(rng.poisson(2.0 if stressed else 1.0)):
        length = rng.integers(*burst_frames)
        start = rng.integers(0, max(1, n_frames - length))
        au20[start:start + length] += expressiveness * rng.uniform(*burst_amplitude)
    df["AU20_r"] = au20

    shared_brow = 4 * np.convolve(rng.normal(0, 1, n_frames), np.ones(9) / 9, mode="same")
    brow_noise = 1.8 if stressed else 0.5
    df["y_38"] = 180.0
    df["y_44"] = 180.0
    df["y_20"] = df["y_38"] - (32 + shared_brow + rng.normal(0, brow_noise, n_frames))
    df["y_24"] = df["y_44"] - (32 + shared_brow + rng.normal(0, brow_noise, n_frames))

    motion = 1.6 if stressed else 1.0
    df["gaze_angle_x"] = rng.normal(0, 0.05 * motion, n_frames)
    df["gaze_angle_y"] = rng.normal(0.1, 0.05 * motion, n_frames)
    for axis in ("pose_Rx", "pose_Ry", "pose_Rz"):
        df[axis] = np.cumsum(rng.normal(0, 0.004 * motion, n_frames))
    df["AU45_c"] = (rng.random(n_frames) < 0.02).astype(float)
    return df


def synthetic_loader(source, window, output_dir):
    rng = np.random.default_rng(window["seed"])
    n_frames = int((window["end"] - window["start"]) * CLIP_FPS)
    return synthetic_window(n_frames, rng, window["kind"] == "risky", source["expressiveness"])


def mock_participants(count=15, seed=7):
    rng = np.random.default_rng(seed)
    questions = [("baseline", f"b{i}") for i in range(8)] + [("risky", f"r{i}") for i in range(6)]
    participants = []
    for i in range(count):
        windows = []
        for q, (kind, question) in enumerate(questions):
            start = q * 5.0
            windows.append({
                "question": question,
                "kind": kind,
                "start": start,
                "end": start + rng.uniform(1, 3),
                "seed": int(rng.integers(1_000_000_000)),
            })
        participants.append({"id": f"P{i + 1:02d}", "video": f"mock_P{i + 1:02d}.mov", "expressiveness": rng.uniform(0.6, 1.5), "windows": windows})
    return participants


def main():
    parser = argparse.ArgumentParser(description="Train the fear-AU deception/stress classifier from raw .mov recordings.")
    parser.add_argument("config", nargs="?", help="participants.json with videos and per-question response windows")
    parser.add_argument("--mock", action="store_true", help="use synthetic OpenFace output instead of real videos")
    parser.add_argument("--cache-dir", default="openface_cache", help="where OpenFace CSVs for each window are kept")
    parser.add_argument("--model-out", default="deception_rf_model.pkl")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    missing = [name for name, path in (("ffmpeg", FFMPEG), ("FeatureExtraction", OPENFACE)) if not path]
    if args.mock or not args.config or missing:
        if not args.mock:
            reason = f"{', '.join(missing)} not found" if missing else "no config given"
            print(f"Running in mock mode: {reason}.", file=sys.stderr)
        participants, loader = mock_participants(), synthetic_loader
    else:
        participants, loader = load_participants(args.config), load_window

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    table = build_window_table(participants, args.cache_dir, loader, args.workers)
    X = baseline_normalize(table)
    y = table["label"]
    print(f"Built feature matrix: {len(X)} response windows from {table['participant'].nunique()} participants, {X.shape[1]} features")

    features_path = Path(args.model_out).with_name("window_features.csv")
    table[["participant", "question", "kind", "label"]].join(X).to_csv(features_path, index=False)
    print(f"Saved features to {features_path}")

    niu_effect_report(participants, args.cache_dir, loader)
    train_loocv_classifier(X, y, table["participant"], args.model_out)


if __name__ == "__main__":
    main()
