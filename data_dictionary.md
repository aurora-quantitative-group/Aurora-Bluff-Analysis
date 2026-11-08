# Data Dictionary: `collected_data.csv`

## Overview
`collected_data.csv` contains the model-ready feature matrix derived from participant survey responses. Each row represents a single participant's answer window for a specific question.

* **Rows:** 207 answer windows (119 baseline, 88 risky)
* **Participants:** 15 unique participants
* **Columns:** 64 total (4 identifier/meta columns, 30 `baseline_` features, 30 `delta_` features)
* **Target Variable:** `label`

---

## Dataset Structure & Feature Naming

Feature metrics are stored in pairs for each measurement:
* `baseline_<metric>`: The mean value of `<metric>` across the participant's baseline questions.
* `delta_<metric>`: The window's observed value relative to the participant's baseline ($\text{Value} - \text{Baseline}$).

---

## Identifier & Metadata Columns

| Column Name | Data Type | Permitted Values | Description |
| :--- | :--- | :--- | :--- |
| `participant` | String | e.g., `IMG_1066-003` | Unique participant identifier derived from source video filename. |
| `question` | String | 14 distinct keys | Identifier for the specific survey question answered (see Question Reference). |
| `kind` | String | `baseline`, `risky` | Phase of the survey protocol. |
| `label` | Integer | `0`, `1` | Classification target: `1` for risky phase questions, `0` for baseline phase questions. |

### Question Reference Keys

| Key | Phase (`kind`) | Survey Prompt |
| :--- | :--- | :--- |
| `name` | Baseline | What is your name? |
| `grade` | Baseline | What is your grade level? |
| `major` | Baseline | What is your major? |
| `state` | Baseline | Are you an in-state or out-of-state student? |
| `university` | Baseline | What university do you attend? |
| `food` | Baseline | What was the last food you ate? |
| `campus` | Baseline | Campus-related prompt |
| `class` | Baseline | What is your favorite class? |
| `gpa` | Risky | What is your GPA? |
| `screentime` | Risky | What is your daily screen time? |
| `sleep` | Risky | How much sleep did you get last night? |
| `ai` | Risky | Do you use AI on in-class assignments? |
| `dated` | Risky | Have you ever dated two people at the same time? |
| `bank` | Risky | What is your current bank account balance? |

---

## Feature Specifications

### 1. Facial Action Units (14 Features)
OpenFace regression intensities ($0.0$ to $5.0$). Each unit includes a window mean (`_mean`) and window standard deviation (`_std`).

| Feature Base | Action Unit | Description | Baseline Median (`_mean`) | Baseline Median (`_std`) |
| :--- | :--- | :--- | :--- | :--- |
| `AU01_r` | AU01 | Inner brow raiser | $0.125$ | $0.169$ |
| `AU02_r` | AU02 | Outer brow raiser | $0.059$ | $0.103$ |
| `AU04_r` | AU04 | Brow lowerer | $0.123$ | $0.163$ |
| `AU05_r` | AU05 | Upper lid raiser | $0.040$ | $0.090$ |
| `AU07_r` | AU07 | Lid tightener | $0.588$ | $0.285$ |
| `AU20_r` | AU20 | Lip stretcher | $0.083$ | $0.136$ |
| `AU26_r` | AU26 | Jaw drop | $0.398$ | $0.392$ |

### 2. Dynamics & Asymmetry (4 Features)

| Feature Base | Unit | Description | Baseline Median |
| :--- | :--- | :--- | :--- |
| `au20_event_count` | Count | Total occurrences of contiguous frames where $\text{AU20\_r} > 0.10$. | $2.6$ |
| `au20_avg_duration` | Milliseconds | Average duration of AU20 events (onset to offset). | $218\text{ ms}$ |
| `au20_transient_count` | Count | AU20 events lasting $\le 15$ frames ($\le 500\text{ ms}$ at 30 fps). | $2.5$ |
| `eyebrow_coherence` | Pearson $r$ ($-1$ to $1$) | Correlation between left and right eyebrow-to-eye distances. Values approaching $1$ indicate symmetric movement. | $0.954$ |

### 3. Head Posture, Gaze & Blinks (8 Features)

| Feature Base | Unit | Description | Baseline Median |
| :--- | :--- | :--- | :--- |
| `gaze_x_std` | Radians | Standard deviation of horizontal gaze angle. | $0.056\text{ rad}$ |
| `gaze_y_std` | Radians | Standard deviation of vertical gaze angle. | $0.049\text{ rad}$ |
| `gaze_y_mean` | Radians | Mean vertical gaze angle. | $0.184\text{ rad}$ |
| `head_pitch_std` | Radians | Standard deviation of head pitch (nodding motion). | $0.030\text{ rad}$ |
| `head_yaw_std` | Radians | Standard deviation of head yaw (turning motion). | $0.041\text{ rad}$ |
| `head_roll_std` | Radians | Standard deviation of head roll (tilting motion). | $0.026\text{ rad}$ |
| `head_speed` | Rad/sec | Average angular velocity across pitch, yaw, and roll. | $0.511\text{ rad/s}$ |
| `blink_rate` | Blinks/sec | Frequency of blink onsets per second of valid video footage. | $0.62\text{ blinks/s}$ |

### 4. Audio & Speech Dynamics (4 Features)

| Feature Base | Unit | Description | Baseline Median |
| :--- | :--- | :--- | :--- |
| `latency` | Seconds | Elapsed time between prompt completion and onset of first response word. | $0.94\text{ s}$ |
| `answer_words` | Count | Total word count spoken within the response window. | $2.5$ |
| `fillers` | Count | Frequency of target filler words and hedges (e.g., *uh, um, like, think, guess*). | $0.0$ |
| `answer_span` | Seconds | Duration from start of first word to completion of final word. | $0.78\text{ s}$ |