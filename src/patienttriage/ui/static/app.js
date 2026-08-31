"use strict";

/* PatientTriage.ai console — talks to the FastAPI endpoints in ui/server.py and
 * renders exactly what they return. No triage logic lives here: every acuity,
 * confidence label and red flag is computed server-side by service.pipeline. */

const state = {
  profile: null,
  profiles: [],
  cases: [],
  lastAssessment: null, // the payload rendered on screen right now
  overrideLevel: 3,
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = JSON.stringify((await res.json()).detail ?? detail); } catch (_) {}
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function pct(x) {
  return x == null ? "—" : `${(x * 100).toFixed(0)}%`;
}

// ------------------------------------------------------------------------ chrome: clock, tabs

function startClock() {
  const tick = () => {
    const now = new Date();
    const clock = $("#live-clock");
    if (clock) clock.textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  };
  tick();
  setInterval(tick, 1000);
}

function initTabs() {
  $$(".tab-btn").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab-btn").forEach((t) => t.classList.remove("active"));
      $$(".view").forEach((v) => v.classList.remove("active"));
      tab.classList.add("active");
      $(`#view-${tab.dataset.view}`).classList.add("active");
      $("#page-title").textContent = tab.dataset.title;
      $("#page-sub").textContent = tab.dataset.sub;

      if (tab.dataset.view === "waitingroom") startWaitingRoomPolling();
      else stopWaitingRoomPolling();
      if (tab.dataset.view === "command") loadCommandCenter();
      if (tab.dataset.view === "routing") loadRouting();
      if (tab.dataset.view === "site") renderSite();
    });
  });
}

function switchTab(view) {
  $(`.tab-btn[data-view="${view}"]`)?.click();
}

// ------------------------------------------------------------------------ site / model status

async function initProfiles() {
  state.profiles = await api("/api/profiles");
  const select = $("#profile-select");
  select.innerHTML = state.profiles.map((p) => `<option value="${p.id}">${p.name}</option>`).join("");
  // District runs the full model; it's the more representative first impression than
  // the rules-only rural tier, which is still one click away in this same list.
  const defaultProfile = state.profiles.find((p) => p.id === "district") ?? state.profiles[0];
  select.value = defaultProfile.id;
  state.profile = select.value;
  select.addEventListener("change", () => {
    state.profile = select.value;
    watchProfileStatus();
    renderSite();
    updateTierBanner();
    seenEscalations = new Set(); // a different site is a different board, not a continuation
  });
  watchProfileStatus();
  updateTierBanner();
}

function currentProfile() {
  return state.profiles.find((p) => p.id === state.profile);
}

// The single most common source of "why didn't this give me a number" confusion:
// a rules-only site has no statistical model at all, by design (see README). That is
// only ever visible in the sidebar pill otherwise, which is easy to miss mid-intake.
function updateTierBanner() {
  const banner = $("#tier-banner");
  const p = currentProfile();
  const isRulesOnly = p && p.model_tier === "rules_only";
  banner.style.display = isRulesOnly ? "flex" : "none";
  if (isRulesOnly) {
    banner.innerHTML = `⚠&nbsp; <span><strong>${escapeHtml(p.name)}</strong> runs on the deterministic
      safety-rule layer only — there is no statistical risk model configured for this
      site. That's intentional (see Site &amp; governance), not a fault: a patient with
      no red flag and normal vitals will show "suggest at least level 5" with no exact
      number, rather than a model score nobody has validated locally. Pick District or
      Urban above for the full model.</span>`;
  }
}

async function watchProfileStatus() {
  const pill = $("#model-status");
  const text = $("#model-status-text");
  await api(`/api/profiles/${state.profile}/prepare`, { method: "POST" });

  const poll = async () => {
    const status = await api(`/api/profiles/${state.profile}/status`);
    if (status.rules_only) {
      pill.className = "safety-pill rules";
      text.textContent = "rules-only site";
      return;
    }
    if (status.preparing) {
      pill.className = "safety-pill busy";
      text.textContent = "training on NHAMCS…";
      setTimeout(poll, 900);
      return;
    }
    if (status.ready) {
      pill.className = "safety-pill";
      text.textContent = "AI safety guardrails active";
      return;
    }
    if (!status.data_available) {
      pill.className = "safety-pill rules";
      text.textContent = "no data — rules only";
      return;
    }
    setTimeout(poll, 900);
  };
  poll();
}

function renderSite() {
  const p = currentProfile();
  if (!p) return;
  $("#site-name").textContent = p.name;
  $("#site-describe").textContent = p.describe;
  $("#site-governance").innerHTML = p.governance
    .map((line) => {
      const [key, ...rest] = line.split(/\s{2,}/);
      const val = rest.join(" ").trim();
      return `<li><span class="g-key">${escapeHtml(key.trim())}</span><span class="g-val">${escapeHtml(val || "—")}</span></li>`;
    })
    .join("");
}

// ------------------------------------------------------------------------ example cases

async function initCases() {
  state.cases = await api("/api/cohort");
  const grid = $("#cases-grid");
  grid.innerHTML = state.cases
    .map((c) => {
      const band = c.nurse_acuity <= 2 ? "l1" : c.nurse_acuity === 3 ? "l3" : "l5";
      return `
      <div class="scenario-card" data-id="${c.id}">
        <span class="scenario-badge" style="background:var(--${band}-tint);color:var(--${band});">${c.id} · nurse level ${c.nurse_acuity}</span>
        <h3>${escapeHtml(c.label)}</h3>
        <div class="scenario-quote">age ${c.age_years < 1 ? Math.round(c.age_years * 365.25) + "d" : Math.round(c.age_years) + "y"}</div>
      </div>`;
    })
    .join("");

  $$(".scenario-card", grid).forEach((card) => {
    card.addEventListener("click", async () => {
      const doc = await api(`/api/cohort/${card.dataset.id}`);
      populateForm(doc.snapshot);
      switchTab("triage");
    });
  });
}

function populateForm(snapshot) {
  const form = $("#triage-form");
  form.reset();
  for (const [key, value] of Object.entries(snapshot)) {
    const field = form.elements.namedItem(key);
    if (!field || value === null || value === undefined) continue;
    if (field.tagName === "SELECT" && (value === true || value === false)) {
      field.value = String(value);
    } else if (Array.isArray(value)) {
      continue; // chief_complaint_codes — not exposed in this form
    } else {
      field.value = value;
    }
  }
}

// ------------------------------------------------------------------------ assessment form

function collectSnapshot() {
  const form = $("#triage-form");
  const data = new FormData(form);
  const snapshot = {};
  for (const [key, raw] of data.entries()) {
    const value = raw.trim();
    if (value === "") continue;
    const field = form.elements.namedItem(key);
    if (field.tagName === "SELECT" && (value === "true" || value === "false")) {
      snapshot[key] = value === "true";
    } else if (field.type === "number") {
      snapshot[key] = Number(value);
    } else {
      snapshot[key] = value;
    }
  }
  return snapshot;
}

// ------------------------------------------------------------------------ voice dictation

// A microphone button that fills the free-text complaint field by speech instead of
// typing — useful the moment a nurse's hands are on a patient rather than a keyboard.
// It only ever fills the field; the real rule engine on the server is what reads the
// text afterwards, same as if it had been typed. This never fabricates a transcript
// when no microphone is available — it says so and leaves the field for typing,
// because inventing plausible-sounding clinical text is the one thing this must not do.
function initVoiceDictation() {
  const button = $("#voice-btn");
  const textarea = $("#f-complaint");
  const hint = $("#dictate-hint");
  if (!button || !textarea) return;

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    button.disabled = true;
    button.title = "This browser does not support speech recognition";
    hint.textContent = "dictation unavailable in this browser — type the complaint instead";
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  let recording = false;
  let finalTranscript = "";
  const micIcon = `<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" viewBox="0 0 24 24"><path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z"/><path d="M19 10v2a7 7 0 01-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/></svg>`;

  recognition.onresult = (event) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      if (event.results[i].isFinal) finalTranscript += event.results[i][0].transcript + " ";
      else interim += event.results[i][0].transcript;
    }
    textarea.value = (textarea.dataset.preDictation || "") + finalTranscript + interim;
  };
  recognition.onerror = (event) => {
    hint.textContent = `dictation stopped (${event.error}) — type to continue`;
    stop();
  };
  recognition.onend = () => stop();

  function start() {
    recording = true;
    finalTranscript = "";
    textarea.dataset.preDictation = textarea.value ? textarea.value + " " : "";
    button.classList.add("recording");
    button.innerHTML = "⏺ Listening…";
    hint.textContent = "listening — speak the handover as you would say it aloud";
    try {
      recognition.start();
    } catch (err) {
      hint.textContent = "could not start the microphone — type the complaint instead";
      stop();
    }
  }

  function stop() {
    recording = false;
    button.classList.remove("recording");
    button.innerHTML = `${micIcon} Dictate`;
    try { recognition.stop(); } catch (_) {}
    if (hint.textContent.startsWith("listening")) hint.textContent = "";
  }

  button.addEventListener("click", () => (recording ? stop() : start()));
}

// ------------------------------------------------------------------------ ambient ECG

// Decorative only — a bedside-monitor motif for the intake screen, not a real feed
// and not a claim about this patient. No numbers are ever drawn on it.
function initEcgCanvas() {
  const canvas = $("#ecg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const H = 56;
  let w = 600, x = 0;

  function beatY(px, mid) {
    const c = ((px % 70) + 70) % 70;
    if (c > 28 && c < 31) return mid + 6;
    if (c >= 31 && c <= 36) return mid - 18;
    if (c > 36 && c < 39) return mid + 9;
    if (c > 46 && c < 54) return mid - 4;
    return mid;
  }

  function stroke(from, to) {
    const mid = H / 2;
    ctx.strokeStyle = "#0d6e73";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(from, beatY(from, mid));
    for (let px = from + 2; px <= to; px += 2) ctx.lineTo(px, beatY(px, mid));
    ctx.stroke();
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    w = canvas.clientWidth || 600;
    canvas.width = w * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, H);
    stroke(0, w);
    x = 0;
  }
  resize();
  window.addEventListener("resize", resize);

  function frame() {
    ctx.clearRect(x, 0, 22, H);
    stroke(Math.max(x - 2, 0), x);
    x += 2;
    if (x > w) x = 0;
    requestAnimationFrame(frame);
  }
  frame();
}

// ------------------------------------------------------------------------ triage form

function initTriageForm() {
  $("#triage-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = event.submitter;
    button.disabled = true;
    try {
      const snapshot = collectSnapshot();
      const started = performance.now();
      const assessment = await api("/api/assess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile: state.profile, snapshot }),
      });
      const roundTrip = (performance.now() - started).toFixed(0);
      $("#assess-latency").textContent = `answered in ${roundTrip} ms (model: ${assessment.latency_ms} ms)`;
      state.lastAssessment = assessment;
      renderAssessment(assessment);
    } catch (err) {
      $("#result-body").innerHTML = `<div class="callout"><strong>Could not assess this patient.</strong>&nbsp;${escapeHtml(err.message)}</div>`;
    } finally {
      button.disabled = false;
    }
  });

  $("#btn-clear-form").addEventListener("click", () => {
    $("#triage-form").reset();
    $("#result-body").innerHTML = '<div class="result-empty">Enter or load a patient, then <strong>Generate triage recommendation</strong>.</div>';
    $("#result-patient-id").textContent = "";
    state.lastAssessment = null;
  });
}

const ACUITY_WORD = { 1: "resuscitation", 2: "emergent", 3: "urgent", 4: "less urgent", 5: "non-urgent" };
const SOURCE_LABEL = { rule: "red flag rule", model: "AI risk model", abstained: "abstained", degraded: "rules only" };

function renderAssessment(a) {
  $("#result-patient-id").textContent = a.patient_id;
  const level = a.recommended_acuity;
  // A rules-only site (or any other fallback) still has a real number to show: the
  // red-flag floor. Only true abstention-to-nothing (an empty conformal set with no
  // rule behind it) has genuinely nothing to display.
  const floorOnly = level == null && a.degraded;
  const displayLevel = level ?? (floorOnly ? a.rule_floor : null);
  const badgeCls = displayLevel ? `acuity-${displayLevel}` : "acuity-na";
  const badgeText = displayLevel ?? "—";
  state.overrideLevel = displayLevel ?? 3;

  const chips = [`<span class="chip source-${a.source}">${SOURCE_LABEL[a.source] || a.source}</span>`];
  if (a.abstained) chips.push(`<span class="chip">insufficient confidence to refine further</span>`);
  // "no model loaded" on a rules-only site is explained by the banner above the form
  // already — repeating it here as a second chip only made a designed tier read as
  // an error. A genuinely unexpected fallback (an exception, a latency breach) still
  // gets flagged, because that one *is* worth a nurse's attention.
  if (a.degraded && a.degraded_reason !== "no model loaded") {
    chips.push(`<span class="chip">unexpected fallback: ${escapeHtml(a.degraded_reason ?? "")}</span>`);
  }

  const warning = a.capability_warning
    ? `<div class="callout">⚠&nbsp; <span>${escapeHtml(a.capability_warning)}</span></div>`
    : "";
  const missing = a.missing_vitals.length
    ? `<div class="callout">⚠&nbsp; <span>Not recorded: ${a.missing_vitals.map((v) => v.replace(/_/g, " ")).join(", ")}</span></div>`
    : "";

  const NEWS2_COLOR = { low: "var(--l5)", "low-medium": "var(--l4)", medium: "var(--l3)", high: "var(--l1)" };
  const news2Color = NEWS2_COLOR[a.news2_band] || "var(--text-secondary)";
  const news2Value = a.news2_score != null
    ? `<span style="color:${news2Color}">${a.news2_score}</span>`
    : `<span style="font-size:.85rem;color:var(--text-muted);">n/a</span>`;

  const news2 = `
    <div class="news2-card">
      <div>
        <h5>National Early Warning Score</h5>
        <div class="sub">${a.news2_score != null ? "Second opinion, RCP 2017" : "not applicable — under 18 or pregnant"}</div>
      </div>
      <div style="text-align:right;">
        <div class="news2-val">${news2Value}</div>
        ${a.news2_score != null ? `<div style="font-size:.68rem;font-weight:700;color:${news2Color};">${escapeHtml(a.news2_band)} risk</div>` : ""}
      </div>
    </div>`;

  const stats = `
    <div class="stat-row">
      <div class="stat-tile"><div class="stat-label">Confidence</div><div class="stat-value">${escapeHtml(a.confidence_label)}</div></div>
      <div class="stat-tile"><div class="stat-label">Data completeness</div><div class="stat-value">${escapeHtml(a.data_completeness)}</div></div>
      <div class="stat-tile"><div class="stat-label">Deterioration risk</div><div class="stat-value">${a.deterioration_risk != null ? (a.deterioration_risk * 100).toFixed(1) + "%" : "not modelled"}</div></div>
      <div class="stat-tile"><div class="stat-label">Prediction set</div><div class="stat-value">${a.prediction_set.length ? a.prediction_set.join(", ") : "—"}</div></div>
    </div>`;

  // Where, not just how urgently. Bed occupancy is never simulated here — this is the
  // zone a patient of this acuity normally goes to, for a charge nurse to check
  // against the real board, not a claim that a bed is actually free right now.
  const destination = a.destination
    ? `<div class="destination-card">
        <div class="destination-zone">→ ${escapeHtml(a.destination.zone)}</div>
        <div class="destination-detail">${escapeHtml(a.destination.bed_type)} · target ${a.destination.target_minutes} min to clinician</div>
        ${a.destination.note ? `<div class="destination-note">⚠ ${escapeHtml(a.destination.note)}</div>` : ""}
      </div>`
    : "";

  const rules = a.rule_hits.length
    ? `<div class="section-label">Red flags fired</div><div class="rule-hit-list">${a.rule_hits
        .map((h) => `<div class="rule-hit"><div class="rule-name">${h.rule_id} — ${escapeHtml(h.name)}</div><div>${escapeHtml(h.reason)}</div><div class="rule-citation">${escapeHtml(h.citation)}</div></div>`)
        .join("")}</div>`
    : "";

  const drivers = a.drivers.length
    ? `<div class="section-label">Explainable AI · what drove this</div><div class="risk-drivers-list">${a.drivers
        .map((d) => `<div class="risk-driver"><span class="arrow">→</span><span>${escapeHtml(d.sentence)}</span></div>`)
        .join("")}</div>`
    : "";

  const subText = level
    ? (ACUITY_WORD[level] ?? "")
    : floorOnly
      ? "at least this level — rules only, no statistical model for this site"
      : "no acuity assigned — see reason above";

  const nurseBox = `
    <div class="nurse-box">
      <div class="card-title" style="font-size:.88rem;margin-bottom:.35rem;">
        <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M16 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="8.5" cy="7" r="4"/><path d="M20 8v6M23 11h-6"/></svg>
        Nurse authority
      </div>
      <p style="font-size:.75rem;color:var(--text-secondary);margin-bottom:.5rem;">Triage nurses retain full clinical authority to accept or adjust this recommendation.</p>

      <div class="hint" style="text-transform:uppercase;font-weight:700;letter-spacing:.05em;">Final level</div>
      <div class="override-options" id="ov-level-options">
        ${[1, 2, 3, 4, 5].map((n) => `<button type="button" class="override-btn ${n === state.overrideLevel ? "active" : ""}" data-level="${n}">${n}</button>`).join("")}
      </div>

      <div class="hint" style="text-transform:uppercase;font-weight:700;letter-spacing:.05em;">Clinician</div>
      <input id="ov-clinician" class="override-select" value="RN-UI" style="margin-top:.3rem;" />

      <div class="hint" style="text-transform:uppercase;font-weight:700;letter-spacing:.05em;">Reason, if different</div>
      <select id="ov-reason" class="override-select">
        <option value="clinical_gestalt">Clinical gestalt</option>
        <option value="history_not_in_system">History not in system</option>
        <option value="data_wrong">Data wrong</option>
        <option value="complaint_misread">Complaint misread</option>
        <option value="department_context">Department context</option>
        <option value="patient_preference">Patient preference</option>
        <option value="other" selected>Other / agrees</option>
      </select>
      <input id="ov-note" class="override-select" placeholder="note (optional)" />

      <div class="form-actions">
        <button class="btn-primary" id="btn-record-override" type="button">
          <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>
          Record decision
        </button>
        <button class="btn-secondary" id="btn-add-waitingroom" type="button">+ Waiting room</button>
      </div>
      <div id="override-result"></div>
      <div id="waitingroom-add-result"></div>
    </div>`;

  $("#result-body").innerHTML = `
    <div class="esi-hero-card">
      <div class="esi-badge-giant ${badgeCls}">${badgeText}</div>
      <div>
        <div class="esi-title-text">${escapeHtml(a.headline)}</div>
        <div class="esi-subtext">${subText}</div>
      </div>
    </div>
    <div class="chip-row">${chips.join("")}</div>
    ${warning}
    ${missing}
    ${destination}
    ${news2}
    ${stats}
    ${rules}
    ${drivers}
    ${nurseBox}
  `;

  $$("#ov-level-options .override-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.overrideLevel = Number(btn.dataset.level);
      $$("#ov-level-options .override-btn").forEach((b) => b.classList.toggle("active", b === btn));
    });
  });

  $("#btn-record-override").addEventListener("click", async () => {
    const button = $("#btn-record-override");
    button.disabled = true;
    try {
      const payload = {
        patient_id: a.patient_id,
        assigned_acuity: state.overrideLevel,
        clinician_id: $("#ov-clinician").value || "RN-UI",
        reason: $("#ov-reason").value,
        note: $("#ov-note").value,
      };
      const result = await api("/api/override", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const flagNote = result.overrode_a_red_flag ? " — this overrides a fired red flag and is routed for review" : "";
      $("#override-result").innerHTML = `<div class="result-note ${result.direction}">Recorded: ${result.direction}${flagNote}</div>`;
    } catch (err) {
      $("#override-result").innerHTML = `<div class="result-note error">${escapeHtml(err.message)}</div>`;
    } finally {
      button.disabled = false;
    }
  });

  $("#btn-add-waitingroom").addEventListener("click", async () => {
    const button = $("#btn-add-waitingroom");
    button.disabled = true;
    try {
      const payload = { patient_id: a.patient_id, assigned_acuity: state.overrideLevel };
      await api("/api/waitingroom/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      $("#waitingroom-add-result").innerHTML = `<div class="result-note sent_for_treatment">Added to the waiting room at level ${payload.assigned_acuity} — see Waiting room.</div>`;
      button.textContent = "✓ On the board";
    } catch (err) {
      $("#waitingroom-add-result").innerHTML = `<div class="result-note error">${escapeHtml(err.message)}</div>`;
      button.disabled = false;
    }
  });
}

// ------------------------------------------------------------------------ bed routing

async function loadRouting() {
  const [routing, board] = await Promise.all([
    api(`/api/routing?profile=${state.profile}`),
    api(`/api/waitingroom?profile=${state.profile}`),
  ]);

  $("#routing-notes").innerHTML = routing.site_notes.length
    ? routing.site_notes.map((n) => `<div class="callout">⚠&nbsp; <span>${escapeHtml(n)}</span></div>`).join("")
    : "";

  const counts = {};
  for (const row of [...board.board, ...board.in_treatment]) {
    counts[row.assigned_acuity] ??= { waiting: 0, in_treatment: 0 };
    counts[row.assigned_acuity][row.in_treatment ? "in_treatment" : "waiting"] += 1;
  }

  $("#routing-grid").innerHTML = routing.zones
    .map((z) => {
      const c = counts[z.acuity] || { waiting: 0, in_treatment: 0 };
      return `
      <div class="zone-card">
        <div class="zone-header">
          <span class="zone-title"><span class="acuity-pill acuity-${z.acuity}">L${z.acuity}</span> ${escapeHtml(z.zone)}</span>
          <span class="zone-target">target ${z.target_minutes} min</span>
        </div>
        <div class="zone-count-row">
          <div class="zone-count"><div class="num">${c.waiting}</div><div class="lbl">Waiting</div></div>
          <div class="zone-count"><div class="num">${c.in_treatment}</div><div class="lbl">In treatment</div></div>
        </div>
        <div class="zone-bed-type">${escapeHtml(z.bed_type)}</div>
      </div>`;
    })
    .join("");
}

// ------------------------------------------------------------------------ waiting room

// A short synthesised alert tone — no audio file to ship, and it only ever fires for
// an escalating alert (vitals worsening enough to lower the suggested level), not for
// a routine wait-time reminder, so it stays meaningful rather than constant.
let audioCtx = null;
function playAlertTone() {
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    if (!audioCtx) audioCtx = new Ctx();
    const now = audioCtx.currentTime;
    [0, 0.22].forEach((offset) => {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "triangle";
      osc.frequency.setValueAtTime(880, now + offset);
      gain.gain.setValueAtTime(0.25, now + offset);
      gain.gain.exponentialRampToValueAtTime(0.01, now + offset + 0.16);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(now + offset);
      osc.stop(now + offset + 0.16);
    });
  } catch (err) {
    console.warn("alert tone unavailable:", err);
  }
}

let seenEscalations = new Set();
let waitingRoomTimer = null;

const RESOLUTION_LABEL = {
  sent_for_treatment: "Sent for treatment",
  stabilised_on_recheck: "Stabilised on recheck",
  discharged: "Discharged",
  transferred: "Transferred",
  other: "Other",
};

function miniKpi(title, value, sub) {
  return `<div class="kpi-card" style="padding:.8rem 1rem;">
    <div class="kpi-title" style="margin-bottom:.3rem;">${title}</div>
    <div class="kpi-body" style="border-top:none;padding-top:0;">
      <div class="kpi-num" style="font-size:1.4rem;">${value}</div>
      <div class="kpi-sub">${sub || ""}</div>
    </div>
  </div>`;
}

async function loadWaitingRoom() {
  const data = await api(`/api/waitingroom?profile=${state.profile}`);
  const asOf = new Date(data.as_of);
  $("#wr-clock").textContent = `as of ${asOf.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;

  const breachEntries = Object.entries(data.breach_rate || {}).sort((a, b) => a[0] - b[0]);
  $("#wr-summary").innerHTML = [
    miniKpi("Waiting for a space", data.total_waiting),
    miniKpi("In treatment", data.total_in_treatment, "moved to a bed, still monitored"),
    ...breachEntries.map(([lvl, rate]) => miniKpi(`Level ${lvl} breach rate`, pct(rate))),
  ].join("");

  $("#wr-board").innerHTML = data.board.length
    ? data.board.map(rowHtml).join("")
    : '<tr><td colspan="5" class="empty-note">Nobody waiting — add a patient from Triage, or load the demo scenario.</td></tr>';

  $("#wr-treatment-board").innerHTML = data.in_treatment.length
    ? data.in_treatment.map(rowHtml).join("")
    : '<tr><td colspan="5" class="empty-note">Nobody in treatment yet.</td></tr>';

  $("#wr-resolved").innerHTML = data.recently_resolved.length
    ? data.recently_resolved
        .map((r) => {
          const when = new Date(r.resolved_at);
          return `<div class="feed-row">
            <span class="feed-time">${when.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
            <span class="feed-body"><span class="pid">${r.patient_id}</span> — ${RESOLUTION_LABEL[r.reason] || r.reason}${r.note ? ": " + escapeHtml(r.note) : ""} · waited ${r.waited_minutes}m</span>
          </div>`;
        })
        .join("")
    : '<div class="empty-note">Nobody has left the department yet.</div>';

  // A visible + audible alert on a *new* escalation only — polling again a few
  // seconds later must not re-trigger for an alert already on screen. Checked across
  // both boards: a patient already moved to a bed can still deteriorate.
  const currentEscalations = new Map();
  for (const row of [...data.board, ...data.in_treatment]) {
    for (const alert of row.alerts) {
      if (alert.escalates) currentEscalations.set(`${row.patient_id}|${alert.reason}|${alert.detail}`, row);
    }
  }
  const newKeys = [...currentEscalations.keys()].filter((key) => !seenEscalations.has(key));
  if (newKeys.length) {
    if ($("#wr-sound").checked) playAlertTone();
    showEscalationToast(currentEscalations.get(newKeys[0]));
  }
  seenEscalations = new Set(currentEscalations.keys());
}

function showEscalationToast(row) {
  const alert = row.alerts.find((a) => a.escalates);
  $("#wr-toast").innerHTML = `
    <div class="deterioration-banner">
      <div style="display:flex;align-items:center;gap:.8rem;">
        <svg width="22" height="22" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        <div>
          <h4>Deterioration detected — ${row.patient_id}</h4>
          <p>${escapeHtml(alert.detail)}</p>
        </div>
      </div>
      <button class="btn-secondary" type="button" onclick="this.closest('.deterioration-banner').remove()">Acknowledge</button>
    </div>`;
}

function rowHtml(row) {
  const rowCls = row.alerts.some((a) => a.escalates) ? "row-escalation" : row.alerts.length ? "row-alert" : "";
  const tags = row.alerts.length
    ? row.alerts.map((a) => `<span class="alert-tag ${a.reason}">${escapeHtml(a.detail)}</span>`).join("")
    : `<span class="alert-tag clear">no alerts</span>`;
  // A patient already in a bed isn't "waiting against a target" any more — the clock
  // that number describes has already ended for them. waited_minutes still counts
  // from their original arrival, not from the bed move, so it's labelled as a total.
  const waitedLabel = row.in_treatment ? `${row.waited_minutes}m total` : `${row.waited_minutes}m / ${row.target_minutes}m`;
  return `
    <tr class="${rowCls}">
      <td><span class="acuity-pill acuity-${row.assigned_acuity}">L${row.assigned_acuity}</span></td>
      <td class="mono">${row.patient_id}</td>
      <td class="mono">${waitedLabel}</td>
      <td>${tags}</td>
      <td>
        <div class="resolve-control">
          <select data-role="resolve-reason" aria-label="Resolution reason for ${row.patient_id}">
            <option value="sent_for_treatment">Sent for treatment</option>
            <option value="stabilised_on_recheck">Stabilised on recheck</option>
            <option value="discharged">Discharged</option>
            <option value="transferred">Transferred</option>
            <option value="other">Other</option>
          </select>
          <button class="btn-secondary" data-role="resolve-btn" data-pid="${row.patient_id}" type="button">✓ Resolve</button>
        </div>
      </td>
    </tr>`;
}

// Event delegation on each board body: rows are replaced wholesale on every poll, so
// listeners attached to individual rows would need re-binding every time.
function initWaitingRoomBoard() {
  ["#wr-board", "#wr-treatment-board"].forEach((sel) => {
    $(sel).addEventListener("click", async (event) => {
      const button = event.target.closest('[data-role="resolve-btn"]');
      if (!button) return;
      const row = button.closest("tr");
      const reason = row.querySelector('[data-role="resolve-reason"]').value;
      button.disabled = true;
      button.textContent = "…";
      try {
        await api("/api/waitingroom/resolve", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ patient_id: button.dataset.pid, reason }),
        });
        await loadWaitingRoom();
      } catch (err) {
        button.disabled = false;
        button.textContent = "✓ Resolve";
        alert(`Could not resolve ${button.dataset.pid}: ${err.message}`);
      }
    });
  });
}

function initWaitingRoom() {
  initWaitingRoomBoard();
  $("#btn-seed-demo").addEventListener("click", async () => {
    const button = $("#btn-seed-demo");
    button.disabled = true;
    try {
      await api("/api/waitingroom/seed_demo", { method: "POST" });
      seenEscalations = new Set(); // a freshly loaded scenario starts its own baseline
      await loadWaitingRoom();
    } finally {
      button.disabled = false;
    }
  });
}

// Live means live: poll while the tab is on screen, and stop the moment it isn't —
// a monitoring board nobody is looking at should not keep working in the background.
function startWaitingRoomPolling() {
  loadWaitingRoom();
  if (waitingRoomTimer) clearInterval(waitingRoomTimer);
  waitingRoomTimer = setInterval(loadWaitingRoom, 12000);
}

function stopWaitingRoomPolling() {
  if (waitingRoomTimer) clearInterval(waitingRoomTimer);
  waitingRoomTimer = null;
}

// ------------------------------------------------------------------------ command center

const ACUITY_COLOR = { 1: "var(--l1)", 2: "var(--l2)", 3: "var(--l3)", 4: "var(--l4)", 5: "var(--l5)", abstained: "var(--neutral)" };
const ACUITY_TINT = { 1: "var(--l1-tint)", 2: "var(--l2-tint)", 3: "var(--l3-tint)", 4: "var(--l4-tint)", 5: "var(--l5-tint)", abstained: "var(--neutral-tint)" };

const KPI_ICONS = {
  clipboard: '<path d="M9 2h6a1 1 0 011 1v2H8V3a1 1 0 011-1z"/><path d="M8 4H6a2 2 0 00-2 2v14a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-2"/>',
  shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
  alert: '<path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
  "user-check": '<path d="M16 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="8.5" cy="7" r="4"/><polyline points="17 11 19 13 23 9"/>',
};

function renderCommandKpis(stats) {
  const overrides = stats.overrides;
  const cards = [
    {
      title: "Assessments this session", icon: "clipboard", color: "var(--brand)", tint: "var(--brand-tint)",
      num: stats.assessment_count,
      sub: stats.assessment_count ? "every one written to the audit log" : "assess a patient in Triage to begin",
    },
    {
      title: "Red flag rate", icon: "shield", color: "var(--l1)", tint: "var(--l1-tint)",
      num: pct(stats.red_flag_rate),
      sub: "share of assessments a deterministic rule decided",
    },
    {
      title: "Abstention rate", icon: "alert", color: "var(--l3)", tint: "var(--l3-tint)",
      num: pct(stats.abstention_rate),
      sub: "the model declined rather than guess",
    },
    {
      title: "Nurse override rate", icon: "user-check", color: "#7c3aed", tint: "#f1e9fe",
      num: overrides.n ? pct(overrides.override_rate) : "—",
      sub: overrides.n ? `${overrides.n} decision(s) recorded — human-in-the-loop authority retained` : "no decisions recorded yet",
    },
  ];
  $("#kpi-grid").innerHTML = cards
    .map(
      (c) => `
    <div class="kpi-card">
      <div class="kpi-head">
        <div class="kpi-icon" style="background:${c.tint};">
          <svg width="18" height="18" fill="none" stroke="${c.color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">${KPI_ICONS[c.icon]}</svg>
        </div>
        <div class="kpi-title">${c.title}</div>
      </div>
      <div class="kpi-body"><div class="kpi-num">${c.num}</div><div class="kpi-sub">${c.sub}</div></div>
    </div>`
    )
    .join("");
}

function renderAcuityDonut(distribution) {
  const host = $("#donut-container");
  const total = Object.values(distribution).reduce((a, b) => a + b, 0);
  if (!total) {
    host.innerHTML = '<div class="empty-note">No assessments yet this session.</div>';
    return;
  }
  const order = ["1", "2", "3", "4", "5", "abstained"];
  const R = 58, STROKE = 15, C = 2 * Math.PI * R, GAP = 2;
  let offset = 0;
  const segments = order
    .filter((k) => distribution[k])
    .map((k) => {
      const n = distribution[k];
      const frac = n / total;
      const len = Math.max(frac * C - GAP, 1);
      const seg = `<circle cx="76" cy="76" r="${R}" fill="none" stroke="${ACUITY_COLOR[k]}" stroke-width="${STROKE}" stroke-linecap="round" stroke-dasharray="${len} ${C - len}" stroke-dashoffset="${-offset}" transform="rotate(-90 76 76)"></circle>`;
      offset += frac * C;
      return seg;
    })
    .join("");
  const legend = order
    .filter((k) => distribution[k])
    .map((k) => {
      const n = distribution[k];
      const label = k === "abstained" ? "Abstained" : `Level ${k}`;
      return `<div class="risk-legend-item">
        <span class="risk-dot" style="border-color:${ACUITY_COLOR[k]};"></span>
        <div><div class="risk-legend-count">${n}</div><div class="risk-legend-label">${label} · ${Math.round((n / total) * 100)}%</div></div>
      </div>`;
    })
    .join("");
  host.innerHTML = `
    <div class="risk-legend">${legend}</div>
    <div class="donut-wrap">
      <svg width="152" height="152" viewBox="0 0 152 152" role="img" aria-label="Acuity distribution over ${total} assessments">
        <circle cx="76" cy="76" r="${R}" fill="none" stroke="var(--border)" stroke-width="${STROKE}"></circle>
        ${segments}
      </svg>
      <div class="donut-center"><div class="donut-center-label">${total}</div><div class="donut-center-sub">assessed</div></div>
    </div>`;
}

function renderAcuityBars(distribution) {
  const total = Object.values(distribution).reduce((a, b) => a + b, 0) || 1;
  const order = ["1", "2", "3", "4", "5", "abstained"];
  const max = Math.max(...order.map((k) => distribution[k] || 0), 1);
  $("#bar-chart").innerHTML = order
    .map((k) => {
      const n = distribution[k] || 0;
      const pctW = Math.round((n / total) * 100);
      const h = Math.max((n / max) * 100, n ? 12 : 4);
      const label = k === "abstained" ? "Abst." : `L${k}`;
      return `<div class="bar-col" title="${label}: ${n} (${pctW}%)">
        <div class="bar-value">${pctW}%</div>
        <div class="bar-shape" style="height:${h}%;border-top-color:${ACUITY_COLOR[k]};background:linear-gradient(180deg, ${ACUITY_TINT[k]}, #ffffff);">
          <span class="bar-pill">${n}</span>
        </div>
      </div>`;
    })
    .join("");
  $("#bar-legend").innerHTML = order
    .map((k) => `<div class="bar-legend-item"><span class="bar-legend-swatch" style="background:${ACUITY_COLOR[k]};"></span>${k === "abstained" ? "Abstained" : `Level ${k}`}</div>`)
    .join("");
}

async function loadCommandCenter() {
  const data = await api("/api/audit?limit=100");
  $("#audit-integrity").innerHTML = data.intact
    ? `✓ chain intact — ${data.count} record(s)`
    : `<span style="color:var(--danger);">✗ ${escapeHtml(data.problem)}</span>`;

  renderCommandKpis(data.stats);
  renderAcuityDonut(data.stats.acuity_distribution);
  renderAcuityBars(data.stats.acuity_distribution);

  $("#audit-rows").innerHTML = data.records.length
    ? data.records
        .map(
          (r) => `<tr>
            <td class="mono">${r.timestamp.replace("T", " ").slice(0, 19)}</td>
            <td><span class="evt-tag ${r.event}">${r.event.replace(/_/g, " ")}</span></td>
            <td class="mono">${r.patient_id}</td>
            <td class="mono">${r.record_hash}…</td>
          </tr>`
        )
        .join("")
    : '<tr><td colspan="4" class="empty-note">No records yet — assess a patient first.</td></tr>';
}

// ------------------------------------------------------------------------ boot

(async function boot() {
  startClock();
  initTabs();
  initTriageForm();
  initVoiceDictation();
  initEcgCanvas();
  initWaitingRoom();
  await initProfiles();
  await initCases();
})();
