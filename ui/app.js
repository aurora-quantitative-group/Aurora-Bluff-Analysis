const $ = (id) => document.getElementById(id);
const video = $('video');
const canvas = $('overlay');
const ctx = canvas.getContext('2d');
const capture = document.createElement('canvas');
const captureCtx = capture.getContext('2d');
const still = new Image();
still.className = 'session-still';
still.hidden = true;
$('viewport').insertBefore(still, canvas);
let stream;
let sessionId;
let status;
let running = false;
let phase = 'idle';
let generation = 0;
let cameraGeneration = 0;
let baselineCount = 0;
let started = 0;
let elapsed = 0;
let samples = [];
let tracking = null;
let trackingTime = 0;
let inspecting = false;
let reporting = false;
let switching = false;
let failures = 0;

function notice(message = '') {
  $('notice').textContent = message;
  $('notice').hidden = !message;
}

async function api(path, data, signal) {
  const response = await fetch(`/api/${path}`, data ? {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data), signal,
  } : { signal });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}

function formatTime(seconds) {
  return `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
}

function controls() {
  $('start-session').disabled = !stream || switching || reporting || !status?.openface || !status?.classifier;
  $('start-session').replaceChildren(document.createTextNode(running ? 'End session ' : 'Start session '));
  const arrow = document.createElement('span');
  arrow.textContent = running ? '■' : '→';
  $('start-session').append(arrow);
  $('reset-session').disabled = running || reporting || samples.length === 0;
  $('generate-report').disabled = running || reporting || !status?.gemini || samples.length === 0;
  $('enable-camera').disabled = switching;
  $('scrubber').disabled = running || samples.length === 0;
}

function setMenu(open) {
  $('camera-menu').classList.toggle('open', open);
  $('camera-menu').inert = !open;
  $('camera-toggle').setAttribute('aria-expanded', String(open));
  if (open) $('camera-list').querySelector('button')?.focus();
}

$('camera-toggle').addEventListener('click', async () => {
  const open = !$('camera-menu').classList.contains('open');
  setMenu(open);
  if (open) await listCameras();
});
document.addEventListener('click', (event) => {
  if (!$('camera-menu').contains(event.target) && !$('camera-toggle').contains(event.target)) setMenu(false);
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && $('camera-menu').classList.contains('open')) {
    setMenu(false);
    $('camera-toggle').focus();
  }
});

async function listCameras() {
  if (!navigator.mediaDevices) {
    $('camera-help').textContent = 'Camera access requires localhost or HTTPS.';
    return;
  }
  try {
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === 'videoinput');
    $('camera-count').textContent = `${devices.length} available`;
    $('camera-list').replaceChildren();
    const selected = stream?.getVideoTracks()[0]?.getSettings().deviceId;
    for (const [index, device] of devices.entries()) {
      const button = document.createElement('button');
      button.className = `device${selected && selected === device.deviceId ? ' selected' : ''}`;
      button.setAttribute('aria-pressed', String(Boolean(selected && selected === device.deviceId)));
      for (const value of ['◉', device.label || `Camera ${index + 1}`, selected && selected === device.deviceId ? '✓' : '']) {
        const span = document.createElement('span');
        span.textContent = value;
        button.append(span);
      }
      button.addEventListener('click', () => connectCamera(device.deviceId));
      $('camera-list').append(button);
    }
    $('camera-help').textContent = devices.length ? '' : 'No cameras found.';
  } catch (error) {
    $('camera-help').textContent = error.message;
  }
}

async function connectCamera(deviceId) {
  if (switching || reporting) return;
  switching = true;
  controls();
  const token = ++cameraGeneration;
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Open this dashboard on localhost to access your camera.');
    const next = await navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: 960 }, height: { ideal: 720 }, ...(deviceId ? { deviceId: { exact: deviceId } } : {}) } });
    if (token !== cameraGeneration) {
      next.getTracks().forEach((track) => track.stop());
      return;
    }
    stopSession();
    stream?.getTracks().forEach((track) => track.stop());
    stream = next;
    video.srcObject = stream;
    await video.play();
    await resetSession();
    $('camera-empty').hidden = true;
    $('live-dot').classList.add('active');
    $('feed-label').textContent = 'Camera';
    $('camera-name').textContent = stream.getVideoTracks()[0].label || 'Camera connected';
    $('resolution').textContent = `${video.videoWidth} × ${video.videoHeight}`;
    stream.getVideoTracks()[0].addEventListener('ended', () => {
      stopSession();
      stream = null;
      $('camera-empty').hidden = false;
      $('live-dot').classList.remove('active');
      notice('Camera disconnected. Reconnect a camera to continue.');
      controls();
    });
    notice(status?.openface && status?.classifier ? '' : status?.message);
    setMenu(false);
    await listCameras();
  } catch (error) {
    const messages = { NotAllowedError: 'Camera access was denied. Allow camera access in your browser and try again.', NotFoundError: 'No camera was found. Connect one and try again.', NotReadableError: 'Camera is unavailable. Close other apps using it and try again.' };
    notice(messages[error.name] || error.message);
  } finally {
    switching = false;
    controls();
  }
}

$('enable-camera').addEventListener('click', () => connectCamera());
navigator.mediaDevices?.addEventListener('devicechange', listCameras);

async function resetSession() {
  generation++;
  running = false;
  phase = 'idle';
  samples = [];
  tracking = null;
  baselineCount = 0;
  elapsed = 0;
  inspecting = false;
  still.hidden = true;
  video.style.visibility = 'visible';
  $('clock').textContent = '00:00';
  $('timeline-end').textContent = '00:00';
  $('scrubber').max = 0;
  $('scrubber').value = 0;
  $('reason').textContent = '';
  $('reason-detail').textContent = '';
  $('observation-state').textContent = 'Waiting';
  $('tracking-label').textContent = 'Ready';
  $('session-status').innerHTML = 'Ready';
  $('report-content').replaceChildren(reportPlaceholder.cloneNode(true));
  $('transcript').value = '';
  sessionId = null;
  drawTimeline();
  controls();
  try {
    const session = await api('session', {});
    sessionId = session.session_id;
  } catch (error) {
    notice(error.message);
  }
}

const reportPlaceholder = $('report-content').firstElementChild.cloneNode(true);
$('reset-session').addEventListener('click', resetSession);

function stopSession() {
  if (!running) return;
  elapsed = (performance.now() - started) / 1000;
  generation++;
  running = false;
  tracking = null;
  phase = 'complete';
  $('session-status').innerHTML = 'Complete';
  $('tracking-label').textContent = 'Paused';
  controls();
}

$('start-session').addEventListener('click', async () => {
  if (running) {
    stopSession();
    return;
  }
  if (samples.length) await resetSession();
  if (!sessionId) {
    notice('Could not create a session. Check that the local server is running.');
    return;
  }
  running = true;
  phase = 'calibration';
  started = performance.now();
  failures = 0;
  inspecting = false;
  still.hidden = true;
  video.style.visibility = 'visible';
  $('session-status').innerHTML = 'Calibrating';
  $('tracking-label').textContent = 'Calibrating';
  notice('');
  controls();
  collect(++generation);
});

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function collect(token) {
  while (running && token === generation) {
    const frames = [];
    const batchStart = performance.now();
    capture.width = 640;
    capture.height = Math.round(640 * video.videoHeight / video.videoWidth);
    for (let index = 0; index < 20; index++) {
      if (!running || token !== generation) return;
      if (video.readyState < 2) {
        stopSession();
        notice('The camera stopped delivering frames. Reconnect it to continue.');
        return;
      }
      captureCtx.drawImage(video, 0, 0, capture.width, capture.height);
      frames.push(capture.toDataURL('image/jpeg', .72).split(',')[1]);
      await delay(100);
    }
    const fps = frames.length * 1000 / (performance.now() - batchStart);
    try {
      const observation = await api('track', { session_id: sessionId, frames, fps, calibrating: phase === 'calibration' }, AbortSignal.timeout(55000));
      if (token !== generation) return;
      failures = 0;
      notice('');
      baselineCount = observation.baseline_count;
      tracking = observation.tracked ? observation : null;
      trackingTime = performance.now();
      $('reason').textContent = observation.reason;
      $('observation-state').textContent = observation.tracked ? 'Observing' : 'No face';
      if (observation.tracked) {
        samples.push({ ...observation, time: (performance.now() - started) / 1000, image: `data:image/jpeg;base64,${frames.at(-1)}`, width: capture.width, height: capture.height });
        if (samples.length > 300) samples.shift();
        $('scrubber').max = samples.length - 1;
        $('scrubber').value = samples.length - 1;
        $('reason-detail').textContent = observation.score === null ? `Baseline ${Math.round(Math.min(baselineCount, 3) / 3 * 100)}%` : `Experimental context score · ${Math.round(observation.tracking_confidence * 100)}% tracking confidence.`;
        if (baselineCount >= 3 && phase === 'calibration') {
          phase = 'observation';
          $('session-status').innerHTML = 'Observing';
          $('tracking-label').textContent = 'Tracking';
        }
        drawTimeline();
      } else {
        $('reason-detail').textContent = 'Tracking will resume when your face is clearly visible.';
      }
    } catch (error) {
      if (token !== generation) return;
      tracking = null;
      failures++;
      notice(error.message);
      if (failures >= 3) {
        stopSession();
        notice(`${error.message} Session paused after three failed tracking requests.`);
        return;
      }
      await delay(1000);
    }
  }
}

function drawOverlay() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const observation = inspecting ? samples[Number($('scrubber').value)] : tracking;
  if (observation?.box && (inspecting || performance.now() - trackingTime < 4000)) {
    const sourceWidth = inspecting ? observation.width : capture.width;
    const sourceHeight = inspecting ? observation.height : capture.height;
    const scale = Math.min(width / sourceWidth, height / sourceHeight);
    const box = observation.box;
    const x = box.x * scale + (width - sourceWidth * scale) / 2;
    const y = box.y * scale + (height - sourceHeight * scale) / 2;
    const w = box.width * scale;
    const h = box.height * scale;
    ctx.strokeStyle = '#dddddd';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x, y, w, h);
    const label = observation.score === null ? 'Establishing baseline' : `${Math.round(observation.score * 100)}% context score`;
    ctx.font = '12px -apple-system, sans-serif';
    const labelWidth = ctx.measureText(label).width + 22;
    const labelX = Math.max(6, Math.min(width - labelWidth - 6, x + (w - labelWidth) / 2));
    const labelY = Math.max(6, y - 38);
    ctx.fillStyle = '#dddddd';
    ctx.beginPath();
    ctx.roundRect(labelX, labelY, labelWidth, 27, 4);
    ctx.fill();
    ctx.fillStyle = '#111111';
    ctx.fillText(label, labelX + 11, labelY + 18);
  }
  if (running) {
    elapsed = (performance.now() - started) / 1000;
    $('clock').textContent = formatTime(elapsed);
    $('timeline-end').textContent = formatTime(elapsed);
  }
  requestAnimationFrame(drawOverlay);
}

function drawTimeline() {
  const end = Math.max(elapsed, samples.at(-1)?.time || 0, 1);
  let path = '';
  let area = '';
  let segment = [];
  const segments = [];
  for (const sample of samples) {
    if (sample.score === null) {
      if (segment.length) segments.push(segment);
      segment = [];
      continue;
    }
    segment.push([sample.time / end * 1000, 132 - sample.score * 124]);
  }
  if (segment.length) segments.push(segment);
  for (const points of segments) {
    const line = points.map(([x, y], index) => `${index ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`).join(' ');
    path += line;
    area += `${line} L${points.at(-1)[0]},132 L${points[0][0]},132 Z `;
  }
  $('line').setAttribute('d', path);
  $('area').setAttribute('d', area);
  $('chart-empty').hidden = Boolean(path);
  if (!path) $('chart-empty').textContent = samples.length ? 'Establishing baseline. Scores appear after calibration.' : '';
  $('timeline-selection').textContent = !running && samples.length ? 'Drag to inspect' : '';
  $('playhead').setAttribute('visibility', inspecting ? 'visible' : 'hidden');
}

$('scrubber').addEventListener('input', () => {
  const sample = samples[Number($('scrubber').value)];
  if (!sample) return;
  inspecting = true;
  still.src = sample.image;
  still.hidden = false;
  video.style.visibility = 'hidden';
  $('reason').textContent = sample.reason;
  $('reason-detail').textContent = `Observation at ${formatTime(sample.time)} · ${Math.round(sample.tracking_confidence * 100)}% tracking confidence`;
  $('observation-state').textContent = 'Reviewing';
  $('timeline-selection').textContent = `${formatTime(sample.time)}`;
  const x = sample.time / Math.max(elapsed, samples.at(-1).time, 1) * 1000;
  $('playhead').setAttribute('x1', x);
  $('playhead').setAttribute('x2', x);
  $('playhead').setAttribute('visibility', 'visible');
});

$('generate-report').addEventListener('click', async () => {
  reporting = true;
  controls();
  $('generate-report').textContent = 'Generating…';
  const token = generation;
  try {
    const report = await api('report', { session_id: sessionId, transcript: $('transcript').value }, AbortSignal.timeout(70000));
    if (token !== generation) return;
    const content = document.createElement('div');
    content.className = 'report-copy';
    const summary = document.createElement('p');
    summary.textContent = report.summary;
    content.append(summary);
    for (const section of report.sections) {
      const heading = document.createElement('h3');
      heading.textContent = section.title;
      const body = document.createElement('p');
      body.textContent = section.body;
      content.append(heading, body);
    }
    $('report-content').replaceChildren(content);
    $('reason').textContent = report.reason;
    $('reason-detail').textContent = 'Generated by Gemini';
    $('observation-state').textContent = 'Gemini';
    notice('');
  } catch (error) {
    notice(error.message);
  } finally {
    reporting = false;
    $('generate-report').textContent = 'Generate report ↗';
    controls();
  }
});

window.addEventListener('pagehide', () => {
  generation++;
  cameraGeneration++;
  stream?.getTracks().forEach((track) => track.stop());
});

async function init() {
  try {
    status = await api('status');
    if (!status.openface || !status.classifier) notice(status.message);
    if (!status.gemini) $('generate-report').title = 'Set GEMINI_API_KEY on the server to enable reports.';
    await resetSession();
  } catch (error) {
    notice(`Cannot connect to the local server: ${error.message}`);
  }
  controls();
  drawOverlay();
}

init();
