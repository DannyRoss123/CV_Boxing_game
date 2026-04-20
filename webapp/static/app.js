// ── Boxing Hub — Frontend App ──────────────────────────────────────────────

const PUNCH_COLORS = {
  jab:        '#ff9b32',
  cross:      '#ff3737',
  hook:       '#ffd732',
  uppercut:   '#32dcff',
  block_high: '#3282ff',
  idle:       '#6b6b88',
};

const GAME_TITLES = {
  memory:     'MEMORY',
  combo_rush: 'COMBO RUSH',
  defense:    'DEFENSE DRILL',
};

// ── State ──────────────────────────────────────────────────────────────────
let _token       = localStorage.getItem('boxing_token') || null;
let _username    = localStorage.getItem('boxing_user')  || null;
let _ws          = null;
let _currentGame = null;
let _poseLandmarker = null;
let _videoStream    = null;
let _frameLoop      = null;
let _lbGame      = 'memory';
let _plans       = null;
let _currentPlan = 'Beginner';
let _lastAction  = null;
let _lastGameState = null;

// ── Screen navigation ──────────────────────────────────────────────────────
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

// ── API helpers ────────────────────────────────────────────────────────────
async function api(method, path, body) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (_token) opts.headers['Authorization'] = 'Bearer ' + _token;
  if (body)   opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

// ── Auth ───────────────────────────────────────────────────────────────────
function setAuthError(msg) {
  document.getElementById('auth-error').textContent = msg;
}

document.querySelectorAll('.auth-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.auth-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const which = tab.dataset.tab;
    document.getElementById('form-login').classList.toggle('hidden',    which !== 'login');
    document.getElementById('form-register').classList.toggle('hidden', which !== 'register');
    setAuthError('');
  });
});

// Enter key on auth inputs
['login-user','login-pass'].forEach(id =>
  document.getElementById(id).addEventListener('keydown', e => { if (e.key === 'Enter') App.login(); }));
['reg-user','reg-pass'].forEach(id =>
  document.getElementById(id).addEventListener('keydown', e => { if (e.key === 'Enter') App.register(); }));

// ── MediaPipe setup ────────────────────────────────────────────────────────
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
      runningMode:           'VIDEO',
      numPoses:              1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence:  0.5,
      minTrackingConfidence:      0.5,
    });

    document.getElementById('mp-status').textContent = 'Pose detection ready';
  } catch (e) {
    console.error('MediaPipe load failed:', e);
    document.getElementById('mp-status').textContent = 'Pose detection unavailable — using server fallback';
  }
}

// ── Camera ─────────────────────────────────────────────────────────────────
async function startCamera() {
  try {
    _videoStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 320, height: 240, facingMode: 'user' },
      audio: false,
    });
    const video = document.getElementById('cam-preview');
    video.srcObject = _videoStream;
    await video.play();
    return true;
  } catch (e) {
    console.error('Camera error:', e);
    document.getElementById('mp-status').textContent = 'Camera access denied — game requires webcam';
    return false;
  }
}

function stopCamera() {
  if (_frameLoop) { cancelAnimationFrame(_frameLoop); _frameLoop = null; }
  if (_videoStream) {
    _videoStream.getTracks().forEach(t => t.stop());
    _videoStream = null;
  }
}

// ── Frame loop: extract landmarks, send to server ──────────────────────────
let _lastFrameTs = 0;
const FRAME_INTERVAL_MS = 66; // ~15 fps

function startFrameLoop() {
  const video = document.getElementById('cam-preview');
  const canvas = document.getElementById('cam-canvas');

  function tick(now) {
    _frameLoop = requestAnimationFrame(tick);
    if (!_ws || _ws.readyState !== WebSocket.OPEN) return;
    if (now - _lastFrameTs < FRAME_INTERVAL_MS) return;
    _lastFrameTs = now;

    if (!_poseLandmarker || video.readyState < 2) return;

    // Match canvas size to video
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width  = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    // Run pose detection
    let result;
    try {
      result = _poseLandmarker.detectForVideo(video, now);
    } catch (e) { return; }

    if (!result.landmarks || result.landmarks.length === 0) return;

    // Flatten landmarks to 132 floats
    const lms = result.landmarks[0];
    const data = new Array(132);
    for (let i = 0; i < 33; i++) {
      data[i*4]   = lms[i].x;
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
  const url   = `${proto}://${location.host}/ws?token=${_token}`;
  _ws = new WebSocket(url);

  _ws.onopen = () => {
    _ws.send(JSON.stringify({ type: 'start_game', game }));
    startFrameLoop();
  };

  _ws.onmessage = e => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    handleWsMessage(msg);
  };

  _ws.onerror = () => {
    document.getElementById('mp-status').textContent = 'Connection error — check network';
  };

  _ws.onclose = () => {
    if (_frameLoop) { cancelAnimationFrame(_frameLoop); _frameLoop = null; }
  };

  // Keepalive ping
  const ping = setInterval(() => {
    if (_ws && _ws.readyState === WebSocket.OPEN) {
      _ws.send(JSON.stringify({ type: 'ping' }));
    } else {
      clearInterval(ping);
    }
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
    showGameOver(msg.game, msg.score);
  }
}

// ── CV status UI ───────────────────────────────────────────────────────────
function updateCVStatus(msg) {
  const badge = document.getElementById('motion-badge');
  const ms    = msg.motion_state || 'IDLE';
  badge.textContent  = ms;
  badge.className    = `motion-badge motion-${ms}`;

  const pct = Math.min(100, Math.round((msg.velocity || 0) / 0.04 * 100));
  document.getElementById('vel-bar').style.width = pct + '%';
}

function updateActionFlash(action, conf) {
  const color = PUNCH_COLORS[action] || '#fff';
  const label = action.replace('_', ' ').toUpperCase();
  document.getElementById('action-name').textContent  = label;
  document.getElementById('action-name').style.color  = color;
  document.getElementById('conf-bar').style.width     = Math.round(conf * 100) + '%';
  document.getElementById('conf-bar').style.background = color;
  document.getElementById('conf-text').textContent    = Math.round(conf * 100) + '% confidence';
  _lastAction = { action, conf };
}

// ── Game state renderers ───────────────────────────────────────────────────
function renderGameState(game, state) {
  if (!state) return;
  document.getElementById('game-score-val').textContent = state.score ?? 0;

  if (game === 'memory')     renderMemory(state);
  else if (game === 'combo_rush') renderCombo(state);
  else if (game === 'defense')    renderDefense(state);
}

function slotHtml(label, status, color) {
  const cls = `slot slot-${status}`;
  const style = status !== 'pending'
    ? `background:${color}22; border-color:${color}; color:${color}`
    : '';
  return `<div class="${cls}" style="${style}">${label}</div>`;
}

function renderMemory(s) {
  const area = document.getElementById('game-state-area');
  const seq  = s.sequence || [];
  const prog = s.progress || [];

  let html = '';

  if (s.phase === 'SHOW') {
    const pct = Math.min(100, Math.round(s.timer / s.show_time * 100));
    html = `
      <div style="text-align:center">
        <div style="color:var(--muted);font-size:.85rem;margin-bottom:.8rem">LEVEL ${s.level} — MEMORISE THIS COMBO</div>
        <div class="slots">
          ${seq.map(p => slotHtml(p.toUpperCase(), 'done', PUNCH_COLORS[p])).join('')}
        </div>
        <div style="margin-top:.8rem">
          <div class="vel-bar-bg"><div class="vel-bar-fill" style="width:${pct}%;background:var(--muted)"></div></div>
          <div style="color:var(--muted);font-size:.75rem;margin-top:.3rem">Memorising... ${(s.show_time - s.timer).toFixed(1)}s</div>
        </div>
      </div>`;
  } else if (s.phase === 'READY') {
    html = `<div style="text-align:center;padding:1.5rem">
      <div style="font-size:4rem;font-weight:900;color:#fff">GO!</div>
      <div class="slots" style="margin-top:1rem">
        ${seq.map((p, i) => slotHtml('?', i < prog.length ? 'done' : 'pending', PUNCH_COLORS[p])).join('')}
      </div>
    </div>`;
  } else if (s.phase === 'PLAYING' || s.phase === 'FEEDBACK') {
    const idx = prog.length;
    html = `<div style="text-align:center">
      <div style="color:var(--muted);font-size:.85rem;margin-bottom:.8rem">
        Punch ${idx + 1} of ${seq.length}
      </div>
      <div class="slots">
        ${seq.map((p, i) => {
          if (i < prog.length) return slotHtml(p.toUpperCase(), 'done', PUNCH_COLORS[p]);
          if (i === prog.length) return slotHtml(p.toUpperCase(), 'active', PUNCH_COLORS[p]);
          return slotHtml('?', 'pending', '');
        }).join('')}
      </div>
      <div style="color:var(--muted);font-size:.8rem;margin-top:.8rem">${prog.length} / ${seq.length}</div>
    </div>`;
  } else if (s.phase === 'LEVEL_UP') {
    html = `<div style="text-align:center;padding:1rem">
      <div style="font-size:1.8rem;font-weight:900;color:var(--green)">LEVEL ${s.level} COMPLETE! ✓</div>
      <div style="color:var(--gold);margin-top:.5rem">Score: ${s.score}</div>
      <div style="color:var(--muted);font-size:.85rem;margin-top:.4rem">Next level loading...</div>
    </div>`;
  } else if (s.phase === 'DONE') {
    html = `<div style="text-align:center;padding:1rem">
      <div style="font-size:1.6rem;font-weight:900;color:var(--red)">GAME OVER</div>
      ${s.last_act ? `<div style="color:var(--muted);font-size:.9rem;margin-top:.5rem">
        Threw <strong style="color:${PUNCH_COLORS[s.last_act.action]}">${s.last_act.action.toUpperCase()}</strong>
        — Expected <strong style="color:${PUNCH_COLORS[s.wrong_expected] || '#fff'}">${(s.wrong_expected||'').toUpperCase()}</strong>
      </div>` : ''}
      <div style="color:var(--gold);margin-top:.6rem;font-size:1.1rem">Level ${s.level} &middot; ${s.score} pts</div>
    </div>`;
  }

  area.innerHTML = html;
}

function renderCombo(s) {
  const area  = document.getElementById('game-state-area');
  const combo = s.combo   || [];
  const prog  = s.progress || [];

  let html = '';

  if (s.phase === 'COUNTDOWN') {
    const n = Math.max(1, 3 - Math.floor(s.timer));
    const colors = { 3: 'var(--red)', 2: 'var(--gold)', 1: 'var(--green)' };
    html = `<div style="text-align:center">
      <div class="countdown-num" style="color:${colors[n] || '#fff'}">${n}</div>
      <div style="color:var(--muted);font-size:.85rem">Get ready — throw the combo as fast as possible!</div>
    </div>`;
  } else if (s.phase === 'THROWING' || s.phase === 'RESULT') {
    const timerHtml = s.elapsed_ms != null && s.phase === 'THROWING'
      ? `<div style="font-size:2.5rem;font-weight:900;color:${s.elapsed_ms > 2000 ? 'var(--red)' : s.elapsed_ms > 1000 ? 'var(--gold)' : 'var(--green)'}">${s.elapsed_ms}ms</div>`
      : `<div style="font-size:1.4rem;font-weight:700;color:var(--muted)">THROW!</div>`;

    const resultHtml = s.phase === 'RESULT' && s.last_result
      ? (s.last_result.ok
          ? `<div class="result-msg result-ok">✓ ${s.last_result.elapsed_ms}ms &nbsp;+${s.last_result.pts} pts</div>`
          : `<div class="result-msg result-fail">✗ Threw ${s.last_result.wrong?.toUpperCase()} — wanted ${s.last_result.expected?.toUpperCase()}</div>`)
      : '';

    html = `<div style="text-align:center">
      <div style="color:var(--muted);font-size:.8rem;margin-bottom:.5rem">
        Round ${s.round + 1} / ${s.total_rounds}
      </div>
      ${timerHtml}
      <div class="slots" style="margin-top:.8rem">
        ${combo.map((p, i) => {
          if (i < prog.length) return slotHtml(p.toUpperCase(), 'done', PUNCH_COLORS[p]);
          if (i === prog.length) return slotHtml(p.toUpperCase(), 'active', PUNCH_COLORS[p]);
          return slotHtml(p.toUpperCase(), 'pending', PUNCH_COLORS[p]);
        }).join('')}
      </div>
      ${resultHtml}
    </div>`;
  } else if (s.phase === 'DONE') {
    html = `<div style="text-align:center;padding:1rem">
      <div style="font-size:1.5rem;font-weight:900;color:var(--text)">COMBO RUSH DONE!</div>
      <div style="font-size:3.5rem;font-weight:900;color:var(--gold);margin:.3rem 0">${s.score}</div>
      <div style="color:var(--muted)">points</div>
    </div>`;
  }

  area.innerHTML = html;
}

function renderDefense(s) {
  const area = document.getElementById('game-state-area');

  let html = '';

  if (s.phase === 'GUIDE') {
    html = `<div style="text-align:center">
      <div style="font-size:1.2rem;font-weight:800;margin-bottom:.8rem">COUNTER GUIDE</div>
      <div style="font-size:.8rem;color:var(--muted);margin-bottom:1rem">Memorise — you won't be told mid-fight!</div>
      <table style="width:100%;border-collapse:collapse;font-size:.85rem;text-align:left">
        <tr style="color:var(--muted);font-size:.75rem">
          <th style="padding:.3rem .5rem">INCOMING</th>
          <th style="padding:.3rem .5rem;color:var(--green)">100 pts</th>
          <th style="padding:.3rem .5rem;color:var(--gold)">80 pts</th>
          <th style="padding:.3rem .5rem;color:var(--block)">40 pts</th>
        </tr>
        ${[
          ['JAB',      'var(--jab)',      'CROSS',    'HOOK',     'BLOCK'],
          ['CROSS',    'var(--cross)',    'HOOK',     'UPPERCUT', 'BLOCK'],
          ['HOOK',     'var(--hook)',     'UPPERCUT', 'CROSS',    'BLOCK'],
          ['UPPERCUT', 'var(--uppercut)', 'JAB',      'CROSS',    'BLOCK'],
        ].map(([atk, c, b1, b2]) => `
          <tr style="border-top:1px solid var(--border)">
            <td style="padding:.4rem .5rem;font-weight:700;color:${c}">${atk}</td>
            <td style="padding:.4rem .5rem;color:var(--green)">${b1}</td>
            <td style="padding:.4rem .5rem;color:var(--gold)">${b2}</td>
            <td style="padding:.4rem .5rem;color:var(--block)">BLOCK HIGH</td>
          </tr>`).join('')}
      </table>
      <button class="btn btn-primary btn-lg" style="margin-top:1.2rem;width:100%"
        onclick="App.sendReady()">READY — LET'S GO</button>
      <div style="color:var(--red);font-size:.75rem;margin-top:.5rem">Wrong punch or timeout = -1 HP</div>
    </div>`;
  } else if (s.phase === 'WAITING' || s.phase === 'JUDGING') {
    const atk   = s.attack || '';
    const col   = PUNCH_COLORS[atk] || '#fff';
    const tpct  = s.time_left != null ? Math.round(s.time_left / s.attack_window * 100) : 0;
    const tcolor = tpct < 35 ? 'var(--red)' : tpct < 65 ? 'var(--gold)' : 'var(--green)';

    const resultHtml = s.phase === 'JUDGING' && s.last_result
      ? (() => {
          const { tag, action, pts } = s.last_result;
          if (tag === 'miss')   return `<div class="result-msg result-fail">TOO SLOW! -1 HP</div>`;
          if (tag === 'wrong')  return `<div class="result-msg result-fail">WRONG! ${action?.toUpperCase()} won't work here. -1 HP</div>`;
          if (tag === 'block')  return `<div class="result-msg result-block">BLOCKED! +${pts} pts</div>`;
          return `<div class="result-msg result-ok">PERFECT COUNTER! +${pts} pts</div>`;
        })()
      : '';

    const hpHtml = Array.from({ length: s.max_hp }, (_, i) =>
      `<div class="hp-dot ${i >= s.hp ? 'lost' : ''}"></div>`).join('');

    const dotHtml = (s.results || []).map(r =>
      `<div class="rd ${r.tag}"></div>`).join('') +
      Array.from({ length: Math.max(0, s.total_rounds - (s.results||[]).length) }, () =>
        `<div class="rd"></div>`).join('');

    html = `<div style="text-align:center">
      <div class="hp-row">${hpHtml}</div>
      <div style="color:var(--muted);font-size:.8rem;margin-bottom:.3rem">
        Round ${s.round + 1} / ${s.total_rounds}
      </div>
      <div style="font-size:3.5rem;font-weight:900;color:${col};letter-spacing:4px">${atk.toUpperCase()}</div>
      ${s.phase === 'WAITING' ? `
      <div class="timer-bar-wrap">
        <div class="timer-bar-bg"><div class="timer-bar-fill" style="width:${tpct}%;background:${tcolor}"></div></div>
        <div class="timer-label">${(s.time_left || 0).toFixed(1)}s to counter</div>
      </div>` : ''}
      ${resultHtml}
      <div class="round-dots">${dotHtml}</div>
    </div>`;
  } else if (s.phase === 'DONE' || s.phase === 'KO') {
    const ko = s.phase === 'KO';
    const res = s.results || [];
    const hits   = res.filter(r => r.tag === 'hit').length;
    const blocks = res.filter(r => r.tag === 'block').length;
    const misses = res.filter(r => r.tag === 'miss' || r.tag === 'wrong').length;
    html = `<div style="text-align:center;padding:.5rem">
      <div style="font-size:1.8rem;font-weight:900;color:${ko ? 'var(--red)' : 'var(--green)'}">
        ${ko ? 'K.O.!!' : 'DRILL COMPLETE!'}
      </div>
      <div style="color:var(--muted);font-size:.85rem;margin:.5rem 0">
        Counters: ${hits} &nbsp;·&nbsp; Blocks: ${blocks} &nbsp;·&nbsp; Misses: ${misses}
      </div>
      <div style="font-size:3rem;font-weight:900;color:var(--gold)">${s.score}</div>
      <div style="color:var(--muted)">points</div>
    </div>`;
  }

  area.innerHTML = html;
}

// ── Game over overlay ──────────────────────────────────────────────────────
function showGameOver(game, score) {
  document.getElementById('go-title').textContent   = GAME_TITLES[game] || game.toUpperCase();
  document.getElementById('go-score').textContent   = score;
  document.getElementById('game-over-overlay').classList.remove('hidden');
  // Refresh hub bests
  App.loadMyScores();
}

// ── The main App object ────────────────────────────────────────────────────
const App = {

  async init() {
    if (_token) {
      this.showHub();
    }
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
    } catch (e) {
      setAuthError(e.message);
    }
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
    } catch (e) {
      setAuthError(e.message);
    }
  },

  logout() {
    _token = null; _username = null;
    localStorage.removeItem('boxing_token');
    localStorage.removeItem('boxing_user');
    showScreen('screen-auth');
  },

  async showHub() {
    document.getElementById('hub-username').textContent = _username || 'Fighter';
    showScreen('screen-hub');
    this.loadLeaderboard('memory');
    this.loadMyScores();
  },

  async loadMyScores() {
    try {
      const data = await api('GET', '/api/scores/me');
      for (const [game, scores] of Object.entries(data)) {
        if (scores.length > 0) {
          const best = Math.max(...scores.map(s => s.score));
          const el   = document.getElementById('best-' + game);
          if (el) el.textContent = 'Best: ' + best;
        }
      }
    } catch {}
  },

  async loadLeaderboard(game) {
    _lbGame = game;
    document.querySelectorAll('.lb-tab').forEach(t =>
      t.classList.toggle('active', t.dataset.game === game));
    try {
      const rows = await api('GET', `/api/scores/leaderboard/${game}`);
      const el   = document.getElementById('lb-content');
      if (!rows.length) {
        el.innerHTML = '<div style="color:var(--muted);padding:1rem;text-align:center">No scores yet — be the first!</div>';
        return;
      }
      const rankColors = ['rank-1', 'rank-2', 'rank-3'];
      el.innerHTML = `<table class="lb-table">
        <thead><tr><th>#</th><th>Fighter</th><th>Score</th><th>Date</th></tr></thead>
        <tbody>
          ${rows.map((r, i) => `
            <tr>
              <td class="${rankColors[i] || ''}">${i+1}</td>
              <td>${r.username}</td>
              <td style="font-weight:700;color:var(--gold)">${r.score}</td>
              <td style="color:var(--muted);font-size:.8rem">${r.played_at?.split('T')[0] || r.played_at || ''}</td>
            </tr>`).join('')}
        </tbody>
      </table>`;
    } catch (e) {
      document.getElementById('lb-content').innerHTML = '<div style="color:var(--muted)">Failed to load</div>';
    }
  },

  async showTraining() {
    showScreen('screen-training');
    if (!_plans) {
      try {
        _plans = await api('GET', '/api/training/plans');
      } catch {}
    }
    this.showPlan('Beginner');
  },

  showPlan(level) {
    _currentPlan = level;
    document.querySelectorAll('.level-tab').forEach(t =>
      t.classList.toggle('active', t.dataset.level === level));
    const p = _plans?.[level];
    if (!p) return;
    const el = document.getElementById('plan-content');
    el.innerHTML = `
      <div class="plan-card card">
        <h3>Drills — ${p.sessions_per_week}x / week</h3>
        <div style="color:var(--muted);font-size:.82rem;margin-bottom:.8rem">${p.focus}</div>
        ${p.drills.map(d => `
          <div class="drill-item">
            <div class="drill-name">${d.name}</div>
            <div class="drill-meta">${d.sets} sets × ${d.reps} reps</div>
            <div class="drill-tip">${d.tip}</div>
          </div>`).join('')}
      </div>
      <div class="plan-card card">
        <h3>Weekly Plan</h3>
        ${p.week_plan.map(day => `<div class="week-item">${day}</div>`).join('')}
        <h3 style="margin-top:1rem">Score Goals</h3>
        ${Object.entries(p.game_goals).map(([g, v]) =>
          `<div class="week-item"><strong>${GAME_TITLES[g]}</strong>: ${v} pts</div>`
        ).join('')}
      </div>`;
  },

  async startGame(game) {
    _currentGame = game;
    document.getElementById('game-title').textContent = GAME_TITLES[game] || game.toUpperCase();
    document.getElementById('game-score-val').textContent = '0';
    document.getElementById('game-over-overlay').classList.add('hidden');
    document.getElementById('game-state-area').innerHTML =
      '<div style="color:var(--muted);text-align:center;padding:2rem">Starting...</div>';
    document.getElementById('action-name').textContent = '—';
    document.getElementById('action-name').style.color = 'var(--muted)';
    document.getElementById('conf-text').textContent   = 'waiting...';
    document.getElementById('conf-bar').style.width    = '0%';

    showScreen('screen-game');

    // Load MediaPipe if not loaded yet
    if (!_poseLandmarker) {
      document.getElementById('mp-status').textContent = 'Loading pose detection...';
      await loadMediaPipe();
    }

    const camOk = await startCamera();
    if (!camOk) return;

    connectWS(game);
  },

  sendReady() {
    if (_ws && _ws.readyState === WebSocket.OPEN) {
      _ws.send(JSON.stringify({ type: 'game_action', action: 'advance' }));
    }
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

  showHub() {
    if (_ws) { _ws.close(); _ws = null; }
    stopCamera();
    document.getElementById('hub-username').textContent = _username || 'Fighter';
    showScreen('screen-hub');
    this.loadLeaderboard('memory');
    this.loadMyScores();
  },
};

// Boot
App.init();
