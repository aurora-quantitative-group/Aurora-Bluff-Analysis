# Aurora

A local camera dashboard for the existing OpenFace classifier. The layout includes a head box with a floating score, a camera picker expanding from the top right, a reason panel, a Gemini report, and a timeline with captured-frame inspection.

## Run

The interface and setup checks run with Python alone:

```sh
python3 server.py
```

Open http://127.0.0.1:8000. Use `--port 8080` for a different port. Camera permissions require localhost or HTTPS.

On macOS, set up OpenFace and the dashboard dependencies with Homebrew installed:

```sh
./scripts/setup_openface.sh
python3 server.py
```

The server automatically uses the project virtual environment and discovers the local OpenFace build. For an existing installation, set `OPENFACE_BIN` to its `FeatureExtraction` executable. OpenFace runs from its binary directory so its landmark and AU models resolve correctly.

For Gemini reports, set `GEMINI_API_KEY` before starting the server.

`FeatureExtraction` can also be on PATH. Its model files must be installed. `GEMINI_MODEL` optionally overrides the default `gemini-2.5-flash`. The API key stays on the server; do not add it to frontend files or Git. Gemini uses Google's [generateContent REST API](https://ai.google.dev/api/generate-content).

## Use

1. Enable the camera. The top-right camera button lists available cameras; selecting one resets the session and baseline.
2. Start a session. Look forward and answer a neutral question while three reliable batches establish a personal baseline.
3. Continue speaking. OpenFace processes batches of 20 JPEG frames sampled at approximately 10 FPS. The box follows the latest reliable facial landmarks. Analysis updates after each OpenFace batch; this is not per-frame inference.
4. End the session. Drag the timeline slider to inspect a captured frame, its head box, and the corresponding observation.
5. Optionally add a transcript or notes and generate an in-depth Gemini report. The report uses measured observations and the context you supplied; microphone recording and transcription are not enabled.

The floating percentage is labeled **context score**: the saved random forest's probability for its risky-context training label. It is not a calibrated uncertainty estimate or a probability that someone is lying. Missing speech features are imputed by the saved model. Tracking confidence is shown separately. The server does not fabricate readings when OpenFace, tracking, or the baseline is unavailable.

Frames go to the local server and temporary OpenFace files are deleted after analysis. The browser retains up to 300 observation snapshots for review; the server retains up to 600 metric samples per session. Sessions expire after two hours of inactivity and disappear when the server exits. Gemini receives metrics and any explicitly entered transcript when you click Generate report; camera images are not sent to Gemini. The server only serves UI assets, never participant files or the model.

## Verification

```sh
python3 -m unittest discover -s tests
python3 -m py_compile server.py
```

To verify the installed OpenFace binary, models, calibration, and saved classifier against OpenFace’s sample sequence:

```sh
AURORA_TEST_OPENFACE=1 .venv/bin/python -m unittest discover -s tests -p test_openface.py
```

The UI uses the supplied Coolvetica font and logo from `assets/`. Only these selected assets are served.
