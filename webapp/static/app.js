// ── Boxing Hub ─────────────────────────────────────────────────────────────

const PUNCH_COLORS = {
  jab:        '#f97316',
  cross:      '#ef4444',
  hook:       '#eab308',
  uppercut:   '#06b6d4',
  block_high: '#3b82f6',
  idle:       '#5a5a78',
};

const GAME_TITLES = {
  memory:     'BOXING MEMORY',
  combo_rush: 'COMBO RUSH',
  defense:    'DEFENSE DRILL',
};

const GAME_LABELS = {
  memory:     'MEMORY',
  combo_rush: 'COMBO RUSH',
  defense:    'DEFENSE',
};

// ── State ──────────────────────────────────────────────────────────────────
let _token          = localStorage.getItem('boxing_token') || null;
let _username       = localStorage.getItem('boxing_user')  || null;
let _ws             = null;
let _currentGame    = null;
let _poseLandmarker = null;
let _videoStream    = null;
let _frameLoop      = null;
let _plans          = null;
let _lastGameState  = null;

// ── Screen navigation ──────────────────────────────────────────────────────
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

// ── API ────────────────────────────────────────────────────────────────────
async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (_token) opts.headers['Authorization'] = 'Bearer ' + _token;
  if (body)   opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

// ── Auth UI ────────────────────────────────────────────────────────────────
function setAuthError(msg) {
  const el = document.getElementById('auth-error');
  el.textContent = msg;
  el.classList.toggle('visible', !!msg);
}

document.querySelectorAll('.auth-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.auth-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const which = tab.dataset.tab;
    document.getElementById('form-login').classList.toggle('hidden',    which !== 'login');
    document.getElementById('form-register').classList.toggle('hidden', which !== 'register');
    document.getElementById('auth-subtitle').textContent = which === 'login'
      ? 'Sign in to your account to continue training.'
      : 'Create a free account to start training.';
    setAuthError('');
  });
});

['login-user', 'login-pass'].forEach(id =>
  document.getElementById(id).addEventListener('keydown', e => { if (e.key === 'Enter') App.login(); }));
['reg-user', 'reg-pass'].forEach(id =>
  document.getElementById(id).addEventListener('keydown', e => { if (e.key === 'Enter') App.register(); }));

// ── MediaPipe ──────────────────────────────────────────────────────────────
async function loadMediaPipe() {
  try {
    const { PoseLandmarker, FilesetResolver } =
      await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs');

    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );

    _poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence:  0.5,
      minTrackingConfidence:      0.5,
    });

    document.getElementById('mp-status').textContent = 'Pose detection active';
  } catch (e) {
    console.error('MediaPipe load failed:', e);
    document.getElementById('mp-status').textContent = 'Pose detection unavailable';
  }
}

// ── Camera ─────────────────────────────────────────────────────────────────
async function startCamera() {
  try {
    _videoStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
    const video = document.getElementById('cam-preview');
    video.srcObject = _videoStream;
    await video.play();
    return true;
  } catch (e) {
    document.getElementById('mp-status').textContent = 'Camera access denied';
    return false;
  }
}

function stopCamera() {
  if (_frameLoop) { cancelAnimationFrame(_frameLoop); _frameLoop = null; }
  if (_videoStream) { _videoStream.getTracks().forEach(t => t.stop()); _videoStream = null; }
}

// ── Frame loop ─────────────────────────────────────────────────────────────
let _lastFrameTs  = 0;
let _poseDetected = false;
const FRAME_MS = 33; // 30fps

function startFrameLoop() {
  const video  = document.getElementById('cam-preview');
  const canvas = document.getElementById('cam-canvas');

  function tick(now) {
    _frameLoop = requestAnimationFrame(tick);
    if (!_ws || _ws.readyState !== WebSocket.OPEN) return;
    if (now - _lastFrameTs < FRAME_MS) return;
    _lastFrameTs = now;

    if (!_poseLandmarker) {
      document.getElementById('mp-status').textContent = 'MediaPipe not loaded';
      return;
    }
    if (video.readyState < 2) {
      document.getElementById('mp-status').textContent = 'Camera starting...';
      return;
    }

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width  = video.videoWidth  || video.clientWidth;
      canvas.height = video.videoHeight || video.clientHeight;
    }

    let result;
    try {
      result = _poseLandmarker.detectForVideo(video, now);
    } catch(e) {
      document.getElementById('mp-status').textContent = 'Detection error: ' + e.message;
      return;
    }

    if (!result.landmarks || !result.landmarks.length) {
      _poseDetected = false;
      document.getElementById('mp-status').textContent = 'No pose detected — step back so full body is visible';
      return;
    }

    _poseDetected = true;
    document.getElementById('mp-status').textContent = 'Pose OK';

    const lms  = result.landmarks[0];
    const data = new Array(132);
    for (let i = 0; i < 33; i++) {
      // Mirror x to match training data (collected with cv2.flip horizontal)
      data[i*4]   = 1.0 - lms[i].x;
      data[i*4+1] = lms[i].y;
      data[i*4+2] = lms[i].z;
      data[i*4+3] = lms[i].visibility ?? 1.0;
    }
    _ws.send(JSON.stringify({ type: 'landmarks', data }));
  }

  _frameLoop = requestAnimationFrame(tick);
}

// ── WebSocket ──────────────────────────────────────────────────────────────
function connectWS(game) {
  if (_ws) { _ws.close(); _ws = null; }
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  _ws = new WebSocket(`${proto}://${location.host}/ws?token=${_token}`);

  _ws.onopen = () => {
    _ws.send(JSON.stringify({ type: 'start_game', game }));
    startFrameLoop();
  };

  _ws.onmessage = e => {
    let msg; try { msg = JSON.parse(e.data); } catch { return; }
    handleWsMessage(msg);
  };

  _ws.onerror = () => {
    document.getElementById('mp-status').textContent = 'Connection error';
  };

  _ws.onclose = () => {
    if (_frameLoop) { cancelAnimationFrame(_frameLoop); _frameLoop = null; }
  };

  const ping = setInterval(() => {
    if (_ws && _ws.readyState === WebSocket.OPEN)
      _ws.send(JSON.stringify({ type: 'ping' }));
    else clearInterval(ping);
  }, 25000);
}

function handleWsMessage(msg) {
  if (msg.type === 'cv') {
    updateCVStatus(msg);
    if (msg.action) updateActionFlash(msg.action, msg.confidence);
  } else if (msg.type === 'game_state') {
    _lastGameState = msg.state;
    renderGameState(_currentGame, msg.state);
  } else if (msg.type === 'game_over') {
    showGameOver(msg.game, msg.score, msg.last_state);
  }
}

// ── CV status ──────────────────────────────────────────────────────────────
let _prevMotionState = 'IDLE';

function updateCVStatus(msg) {
  const ms  = msg.motion_state || 'IDLE';
  const vel = msg.velocity || 0;

  // Badge: flash CLASSIFYING when transitioning MOTION→COOLDOWN
  const badge = document.getElementById('motion-badge');
  if (_prevMotionState === 'MOTION' && ms === 'COOLDOWN') {
    badge.textContent = 'CLASSIFYING...';
    badge.className   = 'motion-badge motion-CLASSIFYING';
    setTimeout(() => {
      badge.textContent = 'COOLDOWN';
      badge.className   = 'motion-badge motion-COOLDOWN';
    }, 300);
  } else {
    badge.textContent = ms;
    badge.className   = `motion-badge motion-${ms}`;
  }
  _prevMotionState = ms;

  // Velocity bar
  const pct   = Math.min(100, Math.round(vel / 0.04 * 100));
  const color = ms === 'MOTION' ? '#f87171' : ms === 'COOLDOWN' ? '#60a5fa' : '#4ade80';
  const velBar   = document.getElementById('vel-bar');
  const velStrip = document.getElementById('vel-strip');
  velBar.style.width = pct + '%'; velBar.style.background = color;
  if (velStrip) { velStrip.style.width = pct + '%'; velStrip.style.background = color; }

  // Show velocity number so user can verify detection is working
  if (_poseDetected) {
    document.getElementById('mp-status').textContent =
      `Pose OK  •  vel: ${vel.toFixed(4)}  •  ${ms}`;
  }
}

function updateActionFlash(action, conf) {
  const color = PUNCH_COLORS[action] || '#fff';
  document.getElementById('action-name').textContent   = action.replace(/_/g, ' ').toUpperCase();
  document.getElementById('action-name').style.color   = color;
  document.getElementById('conf-bar').style.width      = Math.round(conf * 100) + '%';
  document.getElementById('conf-bar').style.background = color;
  document.getElementById('conf-text').textContent     = Math.round(conf * 100) + '% confidence';
}

// ── Game renderers ──────────────────────────────────────────────────────────
function renderGameState(game, state) {
  if (!state) return;
  document.getElementById('game-score-val').textContent = state.score ?? 0;
  if (game === 'memory')     renderMemory(state);
  else if (game === 'combo_rush') renderCombo(state);
  else if (game === 'defense')    renderDefense(state);
}

function _slot(label, status, color) {
  const st = status === 'done'
    ? `background:${color}1a; border-color:${color}; color:${color};`
    : status === 'active'
    ? `background:${color}12; border-color:${color}88; color:${color};`
    : '';
  return `<div class="slot slot-${status}" style="${st}">${label}</div>`;
}

function renderMemory(s) {
  const area = document.getElementById('game-state-area');
  const seq  = s.sequence || [];
  const prog = s.progress || [];

  if (s.phase === 'SHOW') {
    const pct = Math.min(100, Math.round(s.timer / s.show_time * 100));
    area.innerHTML = `
      <div class="game-phase-label">Level ${s.level} &nbsp;&mdash;&nbsp; Memorise this sequence</div>
      <div class="slots">
        ${seq.map(p => _slot(p.replace(/_/g,' ').toUpperCase(), 'done', PUNCH_COLORS[p])).join('')}
      </div>
      <div class="timer-track" style="margin-top:.75rem">
        <div class="timer-fill" style="width:${pct}%;background:var(--muted)"></div>
      </div>
      <div class="timer-lbl">${(s.show_time - s.timer).toFixed(1)}s</div>`;

  } else if (s.phase === 'READY') {
    area.innerHTML = `
      <div class="countdown-big" style="color:var(--green)">GO</div>
      <div class="slots">
        ${seq.map((p, i) => _slot('?', i < prog.length ? 'done' : 'pending', PUNCH_COLORS[p])).join('')}
      </div>`;

  } else if (s.phase === 'PLAYING' || s.phase === 'FEEDBACK') {
    const idx = prog.length;
    area.innerHTML = `
      <div class="game-phase-label">Punch ${idx + 1} of ${seq.length} &mdash; Throw from memory</div>
      <div class="slots">
        ${seq.map((p, i) => {
          if (i < prog.length) return _slot(p.replace(/_/g,' ').toUpperCase(), 'done', PUNCH_COLORS[p]);
          if (i === idx)       return _slot('?', 'active', PUNCH_COLORS[p]);
          return _slot('?', 'pending', '');
        }).join('')}
      </div>
      <div style="color:var(--muted);font-size:.75rem;text-align:center;margin-top:.5rem">${prog.length} / ${seq.length} complete</div>`;

  } else if (s.phase === 'LEVEL_UP') {
    area.innerHTML = `
      <div style="text-align:center;padding:.75rem 0">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;font-weight:900;color:var(--green);letter-spacing:2px">LEVEL ${s.level} COMPLETE</div>
        <div style="color:var(--muted2);font-size:.85rem;margin-top:.4rem">Next level loading...</div>
      </div>`;

  } else if (s.phase === 'DONE') {
    const wrong = s.wrong_expected || '';
    const threw = s.last_act?.action || '';
    area.innerHTML = `
      <div style="text-align:center;padding:.5rem 0">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;font-weight:900;color:var(--red);letter-spacing:2px">GAME OVER</div>
        ${threw ? `<div style="color:var(--muted2);font-size:.85rem;margin-top:.6rem">
          Threw <strong style="color:${PUNCH_COLORS[threw]}">${threw.replace(/_/g,' ').toUpperCase()}</strong>
          &nbsp;&mdash;&nbsp; Expected <strong style="color:${PUNCH_COLORS[wrong]||'#fff'}">${wrong.replace(/_/g,' ').toUpperCase()}</strong>
        </div>` : ''}
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.5rem;font-weight:700;color:var(--gold);margin-top:.75rem">Level ${s.level} &nbsp;&middot;&nbsp; ${s.score} pts</div>
      </div>`;
  }
}

function renderCombo(s) {
  const area  = document.getElementById('game-state-area');
  const combo = s.combo    || [];
  const prog  = s.progress || [];

  if (s.phase === 'COUNTDOWN') {
    const n = Math.max(1, 3 - Math.floor(s.timer));
    const c = { 3: 'var(--red)', 2: 'var(--gold)', 1: 'var(--green)' };
    area.innerHTML = `
      <div class="countdown-big" style="color:${c[n]||'#fff'}">${n}</div>
      <div style="color:var(--muted2);font-size:.82rem;text-align:center">Throw the combo as fast as possible</div>`;

  } else if (s.phase === 'THROWING' || s.phase === 'RESULT') {
    const timerHtml = s.elapsed_ms != null && s.phase === 'THROWING'
      ? `<div style="font-family:'Barlow Condensed',sans-serif;font-size:3rem;font-weight:900;text-align:center;
           color:${s.elapsed_ms > 2000 ? 'var(--red)' : s.elapsed_ms > 1000 ? 'var(--gold)' : 'var(--green)'}">${s.elapsed_ms}<span style="font-size:1.2rem;font-weight:600;color:var(--muted2)"> ms</span></div>`
      : `<div style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;font-weight:700;color:var(--muted2);text-align:center;letter-spacing:2px">THROW</div>`;

    const resultHtml = s.phase === 'RESULT' && s.last_result
      ? (s.last_result.ok
          ? `<div class="result-msg result-ok">${s.last_result.elapsed_ms} ms &nbsp;+${s.last_result.pts} pts</div>`
          : `<div class="result-msg result-fail">Threw ${(s.last_result.wrong||'').replace(/_/g,' ').toUpperCase()} &mdash; wanted ${(s.last_result.expected||'').replace(/_/g,' ').toUpperCase()}</div>`)
      : '';

    area.innerHTML = `
      <div class="game-phase-label">Round ${s.round + 1} of ${s.total_rounds}</div>
      ${timerHtml}
      <div class="slots" style="margin-top:.6rem">
        ${combo.map((p, i) => {
          if (i < prog.length) return _slot(p.replace(/_/g,' ').toUpperCase(), 'done', PUNCH_COLORS[p]);
          if (i === prog.length) return _slot(p.replace(/_/g,' ').toUpperCase(), 'active', PUNCH_COLORS[p]);
          return _slot(p.replace(/_/g,' ').toUpperCase(), 'pending', PUNCH_COLORS[p]);
        }).join('')}
      </div>
      ${resultHtml}`;

  } else if (s.phase === 'DONE') {
    area.innerHTML = `
      <div style="text-align:center;padding:.75rem 0">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;font-weight:900;color:var(--white);letter-spacing:2px">COMPLETE</div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:4rem;font-weight:900;color:var(--gold);line-height:1.1">${s.score}</div>
        <div style="color:var(--muted2);font-size:.78rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase">Points</div>
      </div>`;
  }
}

function renderDefense(s) {
  const area = document.getElementById('game-state-area');

  if (s.phase === 'GUIDE') {
    area.innerHTML = `
      <div class="game-phase-label">Counter Guide &mdash; Memorise before you play</div>
      <table class="guide-table">
        <thead>
          <tr>
            <th>Incoming</th>
            <th style="color:var(--green)">100 pts</th>
            <th style="color:var(--gold)">80 pts</th>
            <th style="color:var(--block)">40 pts</th>
          </tr>
        </thead>
        <tbody>
          ${[
            ['JAB',      'var(--jab)',      'CROSS',    'HOOK',     'BLOCK HIGH'],
            ['CROSS',    'var(--cross)',    'HOOK',     'UPPERCUT', 'BLOCK HIGH'],
            ['HOOK',     'var(--hook)',     'UPPERCUT', 'CROSS',    'BLOCK HIGH'],
            ['UPPERCUT', 'var(--uppercut)', 'JAB',      'CROSS',    'BLOCK HIGH'],
          ].map(([atk, c, b1, b2, b3]) => `
            <tr>
              <td style="color:${c}">${atk}</td>
              <td style="color:var(--green)">${b1}</td>
              <td style="color:var(--gold)">${b2}</td>
              <td style="color:var(--block)">${b3}</td>
            </tr>`).join('')}
        </tbody>
      </table>
      <div style="color:var(--muted2);font-size:.75rem;margin:.75rem 0 .9rem;text-align:center">Wrong punch or timeout costs 1 life</div>
      <button class="btn btn-primary btn-block btn-lg" onclick="App.sendReady()">Ready — Start Drill</button>`;

  } else if (s.phase === 'WAITING' || s.phase === 'JUDGING') {
    const atk   = s.attack || '';
    const col   = PUNCH_COLORS[atk] || '#fff';
    const tpct  = s.time_left != null ? Math.round(s.time_left / s.attack_window * 100) : 0;
    const tcol  = tpct < 35 ? 'var(--red)' : tpct < 65 ? 'var(--gold)' : 'var(--green)';

    const hpHtml = Array.from({ length: s.max_hp || 5 }, (_, i) =>
      `<div class="hp-pip ${i >= s.hp ? 'lost' : ''}"></div>`).join('');

    const dotHtml = (s.results || []).map(r =>
      `<div class="rd ${r.tag}"></div>`).join('') +
      Array.from({ length: Math.max(0, s.total_rounds - (s.results||[]).length) }, () =>
        `<div class="rd"></div>`).join('');

    const resultHtml = s.phase === 'JUDGING' && s.last_result
      ? (() => {
          const { tag, action, pts } = s.last_result;
          if (tag === 'miss')  return `<div class="result-msg result-fail">Too slow &mdash; &minus;1 life</div>`;
          if (tag === 'wrong') return `<div class="result-msg result-fail">${(action||'').replace(/_/g,' ').toUpperCase()} won&apos;t work here &mdash; &minus;1 life</div>`;
          if (tag === 'block') return `<div class="result-msg result-block">Blocked &nbsp;+${pts} pts</div>`;
          return `<div class="result-msg result-ok">Perfect counter &nbsp;+${pts} pts</div>`;
        })()
      : '';

    area.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.6rem">
        <div class="game-phase-label" style="margin:0">Round ${s.round + 1} / ${s.total_rounds}</div>
        <div class="hp-row" style="margin:0">${hpHtml}</div>
      </div>
      <div class="game-punch-name" style="color:${col};margin:.4rem 0">${atk.replace(/_/g,' ').toUpperCase()}</div>
      ${s.phase === 'WAITING' ? `
        <div class="timer-track">
          <div class="timer-fill" style="width:${tpct}%;background:${tcol}"></div>
        </div>
        <div class="timer-lbl">${(s.time_left||0).toFixed(1)}s to counter</div>` : ''}
      ${resultHtml}
      <div class="round-dots" style="margin-top:.75rem">${dotHtml}</div>`;

  } else if (s.phase === 'DONE' || s.phase === 'KO') {
    const ko  = s.phase === 'KO';
    const res = s.results || [];
    const hits   = res.filter(r => r.tag === 'hit').length;
    const blocks = res.filter(r => r.tag === 'block').length;
    const misses = res.filter(r => ['miss','wrong'].includes(r.tag)).length;
    area.innerHTML = `
      <div style="text-align:center;padding:.5rem 0">
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;font-weight:900;letter-spacing:2px;color:${ko ? 'var(--red)' : 'var(--green)'}">
          ${ko ? 'KNOCKED OUT' : 'DRILL COMPLETE'}
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem;margin:1rem 0">
          ${[['Counters', hits, 'var(--green)'], ['Blocks', blocks, 'var(--gold)'], ['Misses', misses, 'var(--red)']].map(
            ([l, v, c]) => `<div style="background:var(--bg3);border:1px solid var(--border);border-radius:7px;padding:.6rem .4rem;text-align:center">
              <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:800;color:${c}">${v}</div>
              <div style="font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--muted2)">${l}</div>
            </div>`
          ).join('')}
        </div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:3rem;font-weight:900;color:var(--gold)">${s.score}</div>
        <div style="color:var(--muted2);font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em">Points</div>
      </div>`;
  }
}

// ── Game over overlay ──────────────────────────────────────────────────────
function showGameOver(game, score, lastState) {
  document.getElementById('go-game-name').textContent = GAME_LABELS[game] || game.toUpperCase();
  document.getElementById('go-title').textContent     = 'GAME OVER';
  document.getElementById('go-score').textContent     = score;

  // Build reason string from last game state
  let reason = '';
  if (lastState) {
    if (game === 'memory' && lastState.last_act) {
      const threw    = lastState.last_act.action;
      const expected = lastState.wrong_expected;
      if (threw && expected)
        reason = `You threw <strong style="color:${PUNCH_COLORS[threw]}">${threw.replace(/_/g,' ').toUpperCase()}</strong>
                  — needed <strong style="color:${PUNCH_COLORS[expected]}">${expected.replace(/_/g,' ').toUpperCase()}</strong>`;
    } else if (game === 'defense' && lastState.last_result) {
      const { tag, action, attack } = lastState.last_result;
      if (tag === 'miss')
        reason = `Too slow on the <strong style="color:${PUNCH_COLORS[attack]}">${(attack||'').replace(/_/g,' ').toUpperCase()}</strong>`;
      else if (tag === 'wrong' && action)
        reason = `<strong style="color:${PUNCH_COLORS[action]}">${action.replace(/_/g,' ').toUpperCase()}</strong>
                  won't counter a <strong style="color:${PUNCH_COLORS[attack]}">${(attack||'').replace(/_/g,' ').toUpperCase()}</strong>`;
    }
  }
  document.getElementById('go-reason').innerHTML = reason;
  document.getElementById('game-over-overlay').classList.remove('hidden');
  App.loadMyScores();
}

// ── App ────────────────────────────────────────────────────────────────────
const App = {

  async init() {
    if (_token) this.showHub();
  },

  async login() {
    setAuthError('');
    const u = document.getElementById('login-user').value.trim();
    const p = document.getElementById('login-pass').value;
    try {
      const res = await api('POST', '/api/auth/login', { username: u, password: p });
      _token = res.token; _username = res.username;
      localStorage.setItem('boxing_token', _token);
      localStorage.setItem('boxing_user',  _username);
      this.showHub();
    } catch (e) { setAuthError(e.message); }
  },

  async register() {
    setAuthError('');
    const u = document.getElementById('reg-user').value.trim();
    const p = document.getElementById('reg-pass').value;
    try {
      const res = await api('POST', '/api/auth/register', { username: u, password: p });
      _token = res.token; _username = res.username;
      localStorage.setItem('boxing_token', _token);
      localStorage.setItem('boxing_user',  _username);
      this.showHub();
    } catch (e) { setAuthError(e.message); }
  },

  logout() {
    _token = null; _username = null;
    localStorage.removeItem('boxing_token');
    localStorage.removeItem('boxing_user');
    showScreen('screen-auth');
  },

  showHub() {
    if (_ws) { _ws.close(); _ws = null; }
    stopCamera();
    document.getElementById('hub-username').textContent = _username || 'Fighter';
    showScreen('screen-hub');
    this.loadLeaderboard('memory');
    this.loadMyScores();
  },

  async loadMyScores() {
    try {
      const data = await api('GET', '/api/scores/me');
      for (const [game, scores] of Object.entries(data)) {
        const el = document.getElementById('best-' + game);
        const sv = document.getElementById('stat-' + game);
        if (scores.length) {
          const best = Math.max(...scores.map(s => s.score));
          if (el) el.textContent = best.toLocaleString();
          if (sv) sv.textContent = best.toLocaleString();
        } else {
          if (el) el.textContent = '—';
          if (sv) sv.textContent = '—';
        }
      }
    } catch {}
  },

  async loadLeaderboard(game) {
    document.querySelectorAll('.lb-tab').forEach(t =>
      t.classList.toggle('active', t.dataset.game === game));
    try {
      const rows = await api('GET', `/api/scores/leaderboard/${game}`);
      const el   = document.getElementById('lb-content');
      if (!rows.length) {
        el.innerHTML = '<div style="color:var(--muted);font-size:.85rem;padding:.5rem 0">No scores yet — be the first.</div>';
        return;
      }
      el.innerHTML = `<table class="lb-table">
        <thead><tr><th>#</th><th>Fighter</th><th>Score</th><th>Date</th></tr></thead>
        <tbody>
          ${rows.map((r, i) => `
            <tr class="${i < 3 ? 'lb-rank-' + (i+1) : ''}">
              <td>${i + 1}</td>
              <td>${r.username}</td>
              <td class="lb-score">${r.score.toLocaleString()}</td>
              <td style="color:var(--muted);font-size:.78rem">${(r.played_at||'').split('T')[0]}</td>
            </tr>`).join('')}
        </tbody>
      </table>`;
    } catch {
      document.getElementById('lb-content').innerHTML = '<div style="color:var(--muted);font-size:.85rem">Failed to load.</div>';
    }
  },

  async showTraining() {
    showScreen('screen-training');
    if (!_plans) {
      try { _plans = await api('GET', '/api/training/plans'); } catch {}
    }
    this.showPlan('Beginner');
  },

  showPlan(level) {
    document.querySelectorAll('.level-tab').forEach(t =>
      t.classList.toggle('active', t.dataset.level === level));
    const p = _plans?.[level];
    if (!p) return;
    document.getElementById('plan-content').innerHTML = `
      <div class="plan-panel">
        <div class="plan-panel-title">Drills &mdash; ${p.sessions_per_week}x per week</div>
        <div style="color:var(--muted2);font-size:.8rem;margin-bottom:1rem;font-style:italic">${p.focus}</div>
        ${p.drills.map(d => `
          <div class="drill-item">
            <div class="drill-name">${d.name}</div>
            <div class="drill-meta">${d.sets} sets &times; ${d.reps} reps</div>
            <div class="drill-tip">${d.tip}</div>
          </div>`).join('')}
      </div>
      <div class="plan-panel">
        <div class="plan-panel-title">Weekly Schedule</div>
        ${p.week_plan.map(day => `<div class="week-item">${day}</div>`).join('')}
        <div class="plan-panel-title" style="margin-top:1.25rem">Score Targets</div>
        ${Object.entries(p.game_goals).map(([g, v]) =>
          `<div class="goal-row">
             <span class="goal-game">${GAME_LABELS[g] || g}</span>
             <span class="goal-pts">${v.toLocaleString()} pts</span>
           </div>`).join('')}
      </div>`;
  },

  async startGame(game) {
    _currentGame = game;
    document.getElementById('game-title').textContent     = GAME_TITLES[game];
    document.getElementById('game-score-val').textContent = '0';
    document.getElementById('game-over-overlay').classList.add('hidden');
    document.getElementById('game-state-area').innerHTML  =
      '<div style="color:var(--muted);text-align:center;padding:2rem;font-size:.875rem">Starting...</div>';
    document.getElementById('action-name').textContent   = '—';
    document.getElementById('action-name').style.color   = 'var(--muted)';
    document.getElementById('conf-text').textContent     = 'Waiting for movement...';
    document.getElementById('conf-bar').style.width      = '0%';
    document.getElementById('mp-status').textContent     = 'Initialising...';

    showScreen('screen-game');

    if (!_poseLandmarker) {
      document.getElementById('mp-status').textContent = 'Loading pose detection...';
      await loadMediaPipe();
    }

    const ok = await startCamera();
    if (!ok) return;

    connectWS(game);
  },

  sendReady() {
    if (_ws && _ws.readyState === WebSocket.OPEN)
      _ws.send(JSON.stringify({ type: 'game_action', action: 'advance' }));
  },

  leaveGame() {
    if (_ws) { _ws.close(); _ws = null; }
    stopCamera();
    this.showHub();
  },

  playAgain() {
    document.getElementById('game-over-overlay').classList.add('hidden');
    const game = _currentGame;
    if (_ws) { _ws.close(); _ws = null; }
    stopCamera();
    this.startGame(game);
  },
};

App.init();
