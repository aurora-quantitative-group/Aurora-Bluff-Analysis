# Research Methodology

## 1. Research Design & Approach

This study operates on the premise that physiological and facial micro-expressions corresponding to stress or cognitive load can be objectively measured.

A **mixed-methods research approach** was utilized, combining semi-structured video survey interviews  with algorithmic facial Action Unit (AU) analysis and Random Forest machine learning classification. The primary goal of this design is to quantify micro-expression variance between baseline, neutral responses and high-stimulus, "risky" responses.

## 2. Target Population & Sampling Procedure

* **Target Population:** Enrolled students at Texas A&M University.

* **Sample Size:** $N = 15$ participants ($207$ total analyzed response windows).

* **Sampling Method:** Systematic random sampling was conducted along a predetermined walking route across the Texas A&M campus. Researchers approached every fifth individual encountered along the designated path.

* **Inclusion Criteria:** Currently enrolled Texas A&M students walking individually along the sampling route on Saturday, November 7, 2026.

* **Exclusion Criteria:**

  * Non-student pedestrians.

  * Individuals using micro-mobility or vehicles (e.g., bicycles, scooters, skateboards, cars).

  * Individuals walking in groups.

  * Individuals actively engaged in distracting tasks (e.g., wearing headphones, visibly on a phone call) or appearing overtly busy.

## 3. Data Collection & Instrumentation

Data collection relied exclusively on primary video recording; no secondary datasets were utilized.

* **Primary Instrument:** Mobile camera positioned to capture high-resolution facial movements and micro-expressions during verbal responses.

* **Environmental Setting:** Field survey conducted along a fixed exterior campus route adjacent to the research team's operational facility to maintain consistent ambient lighting and environmental conditions.

## 4. Experimental Procedure & Survey Protocol

Each participant was administered a two-part structured verbal survey designed to establish an emotional/physiological control state followed by an elevated cognitive/stress state.

### Phase 1: Baseline Questions

Designed to establish low-arousal, neutral baseline facial dynamics under presumed truth-telling conditions:

1. What is your name?

2. What is your grade level?

3. What is your major?

4. Are you an in-state or out-of-state student?

5. What university do you attend?

6. What was the last food you ate?

7. What is your favorite class?

### Phase 2: High-Stimulus ("Risky") Questions

Designed to elicit cognitive load, discomfort, or affective physiological leakage:

1. What is your GPA?

2. What is your daily screen time?

3. How much sleep did you get last night?

4. Do you use AI on in-class assignments?

5. Have you ever dated two people at the same time?

6. What is your current bank account balance?

## 5. Data Analysis & Computational Pipeline

### 5.1 Data Cleaning & Feature Extraction

Video clips were trimmed into discrete answer windows ($1\text{--}3\text{ seconds}$ per response) and processed through the **OpenFace 2.0 CLI** (`FeatureExtraction`) to generate frame-by-frame Action Unit (AU) intensity and presence metrics.

1. **Quality Filtering:** Frames with a facial tracking confidence lower than 80% ($\text{confidence} < 0.80$) or tracking failure ($\text{success} == 0$) were removed from the analysis.

2. **Fear Action Unit Metrics:** Mean ($\mu$) and standard deviation ($\sigma$) values were calculated across seven primary fear-related Action Units derived from Niu et al. (2021):

   * $\text{AU01}$ (Inner Brow Raiser)

   * $\text{AU02}$ (Outer Brow Raiser)

   * $\text{AU04}$ (Brow Lowerer)

   * $\text{AU05}$ (Upper Lid Raiser)

   * $\text{AU07}$ (Lid Tightener)

   * $\text{AU20}$ (Lip Stretcher)

   * $\text{AU26}$ (Jaw Drop)

3. **Transient Micro-Expressions:** Peak detection algorithms identified transient $\text{AU20}$ intensity spikes ($> 0.10$). Total duration (onset-to-offset) and frequency of brief micro-expressions ($\le 15\text{ frames}$ / $500\text{ ms}$) were extracted.

4. **Eyebrow Asymmetry Index:** Vertical distances between eyebrow centers ($y_{20}, y_{24}$) and eye centers ($y_{38}, y_{44}$) were computed frame-by-frame. The Pearson correlation coefficient ($r$) between left and right side movements measured asymmetry, where lower correlation indicates higher facial asymmetry.

### 5.2 Within-Subject Baseline Normalization ($\Delta \text{Metric}$)

To control for individual expressiveness, resting facial asymmetry, and speech movement dynamics, a within-subject baseline subtraction was calculated:

$$
\Delta \text{Metric} = \text{Metric}_{\text{Risky}} - \text{Metric}_{\text{Baseline}}
$$

The resulting vector contains both differential shift scores ($\Delta \text{Metric}$) and raw baseline features to retain individual context.

### 5.3 Feature Weighting & Machine Learning Model

Feature vectors were scaled based on Cohen’s $d$ effect sizes established by Niu et al. (2021):

* **Eyebrow Asymmetry Shift:** Multiplier $= 2.0$ ($d = 1.314$).

* **Active AU Suppression (**$\text{AU05}, \text{AU20}$**):** Multiplier $= 1.8$ ($d \approx 0.77\text{--}0.81$).

* **Micro-Expression Duration & Frequency:** Multiplier $= 1.5$.

A Random Forest classification model ("face-only forest") was trained using a **Leave-One-Group-Out Cross-Validation (LOGOCV)** scheme across all 15 participants:

* Model trained on 14 participants and evaluated on the held-out 15th participant across all 207 answer windows.

* Held-out baseline scores for 119 control windows were sorted to establish an empirical baseline reference distribution ($\text{Median} = 0.395$, $90\text{th percentile} = 0.544$).

### 5.4 Application Integration & Percentile Scoring

The trained model outputs continuous stress/uncertainty probabilities bounded between $0\%$ and $100\%$, streamed frame-by-frame or window-by-window to a real-time user interface:

* A raw model score is converted into a percentile rank relative to the 119 held-out baseline reference scores.

* Percentile calculation formula:

$$
\text{Percentile} = \frac{\text{Count}(\text{Baseline Scores} < \text{Raw Score}) + 0.5 \times \text{Count}(\text{Ties})}{\text{Total Baseline Scores (119)}} \times 100
$$

* *Interpretation:* A percentile of $50\%$ represents a standard baseline response, whereas a percentile of $90\%$ indicates that only $10\%$ of baseline control responses scored higher (e.g., a raw score of $0.506 \approx 87\text{th percentile}$).

## 6. Ethical Considerations & Methodological Rigor

### 6.1 Ethical Compliance

* **Informed Consent:** Verbal informed consent was obtained from all 15 participants prior to recording. Participants were explicitly informed that their facial movements were being recorded and analyzed for research purposes.

* **Secondary Consent for Publication:** Participants were separately asked for permission to include their video footage in public project demonstrations or presentations. Requests to exclude video recordings from public presentation were strictly honored while preserving anonymized numerical feature vectors.

### 6.2 Validity & Methodological Limitations

* **Sample Size Constraints:** The small sample size ($N = 15$) limits generalizability across broader populations, functioning primarily as a proof-of-concept pilot study.

* **Sampling Bias:** Excluding individuals in groups, those wearing headphones, or those operating micro-mobility devices may introduce selection bias toward specific personality types or activity states.

* **Environmental Noise:** Outdoor handheld video collection introduces variability in lighting, camera distance, and natural ambient distractions, which was partially mitigated by OpenFace tracking confidence filtering ($\ge 80\%$) and within-subject baseline normalization.