(() => {
  "use strict";

  const API = "";

  // --------------------------------------------------------------------
  // Stage rail configuration — mirrors pipeline.py exactly
  // --------------------------------------------------------------------
  const STAGES = [
    {
      id: "stage1", code: "1", label: "Guardrails",
      telemetryKey: "stage1_pre_guardrails", latencyKey: "stage1_pre_guardrails_ms",
      title: "Stage 1 — Pre-Execution Guardrails",
      risk: (t) => (t?.is_blocked ? 10 : (t?.injection_score ?? 0)),
      detail: (t) => t ? [
        ["PII detected", (t.pii_detected || []).map(p => p.entity_type).join(", ") || "None"],
        ["Sanitized prompt", t.sanitized_prompt],
        ["Injection score", `${t.injection_score} / 10`],
        ["Blocked", t.is_blocked ? `Yes — ${t.block_reason}` : "No"],
      ] : [],
    },
    {
      id: "gen", code: "2", label: "Primary LLM",
      telemetryKey: null, latencyKey: "primary_llm_generation_ms",
      title: "Stage 2 — Primary LLM Generation",
      risk: () => null,
      detail: (t, full) => [
        ["Candidate response", (full?.candidate_response || "").slice(0, 400) || "—"],
      ],
    },
    {
      id: "stage3a", code: "3A", label: "Fast Checks",
      telemetryKey: "stage3a_fast_checks", latencyKey: "stage3a_fast_checks_ms",
      title: "Stage 3A — Fast Parallel Checks",
      risk: (t) => t ? Math.max(t.heuristic_risk ?? 0, t.stat_risk ?? 0) : null,
      detail: (t) => t ? [
        ["Heuristic risk", `${t.heuristic_risk} / 10`],
        ["Statistical risk", `${t.stat_risk} / 10`],
        ["Banned lexicon hits", (t.banned_lexicon_hits || []).join(", ") || "None"],
        ["Output PII", (t.output_pii || []).map(p => p.entity_type).join(", ") || "None"],
        ["N-gram repetition", t.ngram_repetition],
        ["Cosine similarity", t.cosine_similarity],
      ] : [],
    },
    {
      id: "stage3b", code: "3B", label: "RAG Grounding",
      telemetryKey: "stage3b_rag_grounding", latencyKey: "stage3b_rag_grounding_ms",
      title: "Stage 3B — RAG Factual Grounding",
      risk: (t) => t ? (t.rag_risk ?? 0) : null,
      detail: (t) => t ? [
        ["Grounding score", `${t.grounding_score} / 10`],
        ["Retrieved chunks", (t.retrieved_chunks || []).map(c => c.doc_id).join(", ") || "None"],
        ["Unsupported claims", (t.unsupported_claims || []).join("; ") || "None"],
        ["Numeric mismatches", (t.numeric_mismatches || []).join("; ") || "None"],
      ] : [],
    },
    {
      id: "stage3c", code: "3C", label: "AI Judge",
      telemetryKey: "stage3c_ai_judge", latencyKey: "stage3c_ai_judge_ms",
      title: "Stage 3C — AI-as-a-Judge Evaluation",
      risk: (t) => t ? (t.judge_risk_score ?? 0) : null,
      detail: (t) => t ? [
        ["Bias score", `${t.bias_score} / 10`],
        ["Tone score", `${t.tone_score} / 10`],
        ["Policy risk", `${t.policy_risk_score} / 10`],
        ["Judge notes", t.judge_notes],
      ] : [],
    },
    {
      id: "stage4", code: "4", label: "Arbitration",
      telemetryKey: "stage4_arbitration", latencyKey: "stage4_arbitration_ms",
      title: "Stage 4 — Policy Arbitration",
      risk: (t, full) => t ? t.composite_score : full?.composite_score ?? null,
      detail: (t, full) => [
        ["Composite score", `${(t?.composite_score ?? full?.composite_score ?? 0)} / 10`],
        ["Decision", t?.decision ?? full?.decision],
        ["Financial trigger", (t?.is_financial_trigger ?? full?.is_financial_trigger) ? "Yes" : "No"],
        ["Reason", t?.reason ?? "—"],
      ],
    },
    {
      id: "stage5", code: "5", label: "Governance",
      telemetryKey: null, latencyKey: "stage5_governance_ms",
      title: "Stage 5 — Governance & Audit",
      risk: () => null,
      detail: (t, full) => [
        ["Audit hash", full?.audit_hash],
        ["HITL ticket", full?.ticket_id || "None — not quarantined"],
      ],
    },
  ];

  const DECISION_CLASS = { ALLOW: "status-clean", HITL: "status-caution", BLOCK: "status-risk" };
  const DECISION_DOT = { ALLOW: "dot-green", HITL: "dot-amber", BLOCK: "dot-red" };
  const DECISION_COLOR_VAR = { ALLOW: "var(--allow)", HITL: "var(--hitl)", BLOCK: "var(--block)" };

  function riskClass(score) {
    if (score === null || score === undefined) return null;
    if (score >= 7) return "status-risk";
    if (score > 2.5) return "status-caution";
    return "status-clean";
  }

  // --------------------------------------------------------------------
  // State
  // --------------------------------------------------------------------
  let sessionHistory = [];
  let lastFullResult = null;
  let scenarios = [];

  // --------------------------------------------------------------------
  // DOM refs
  // --------------------------------------------------------------------
  const $ = (sel) => document.querySelector(sel);
  const rail = $("#rail");
  const stageDetail = $("#stageDetail");
  const railSub = $("#railSub");
  const verdictPanel = $("#verdictPanel");
  const runBtn = $("#runBtn");
  const runBtnLabel = $("#runBtnLabel");
  const promptInput = $("#promptInput");
  const toastEl = $("#toast");

  function toast(msg) {
    toastEl.textContent = msg;
    toastEl.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => toastEl.classList.remove("show"), 2200);
  }

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
  }

  // --------------------------------------------------------------------
  // Rail rendering
  // --------------------------------------------------------------------
  function buildRail() {
    rail.innerHTML = "";
    const track = document.createElement("div");
    track.className = "rail-track";
    track.innerHTML = '<div class="rail-track-fill" id="railTrackFill"></div>';
    rail.appendChild(track);

    STAGES.forEach((stage) => {
      const node = document.createElement("button");
      node.type = "button";
      node.className = "rail-node";
      node.id = `node-${stage.id}`;
      node.innerHTML = `
        <span class="node-circle">${stage.code}</span>
        <span class="node-label">${stage.label}</span>
        <span class="node-score" id="score-${stage.id}"></span>
      `;
      node.addEventListener("click", () => showStageDetail(stage.id));
      rail.appendChild(node);
    });
  }

  function resetRailVisual() {
    STAGES.forEach((s) => {
      const node = $(`#node-${s.id}`);
      node.className = "rail-node";
      $(`#score-${s.id}`).textContent = "";
    });
    const fill = $("#railTrackFill");
    if (fill) fill.style.width = "0%";
    stageDetail.hidden = true;
    stageDetail.classList.remove("show");
  }

  function showStageDetail(stageId) {
    if (!lastFullResult) return;
    STAGES.forEach((s) => $(`#node-${s.id}`)?.classList.remove("selected"));
    $(`#node-${stageId}`)?.classList.add("selected");

    const stage = STAGES.find((s) => s.id === stageId);
    const t = stage.telemetryKey ? lastFullResult.telemetry?.[stage.telemetryKey] : null;
    const rows = stage.detail(t, lastFullResult) || [];

    stageDetail.hidden = false;
    stageDetail.innerHTML = `
      <h4>${esc(stage.title)}</h4>
      <dl>
        ${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}
        <dt>Stage latency</dt><dd>${lastFullResult.telemetry?.waterfall_latency_ms?.[stage.latencyKey] ?? "—"} ms</dd>
      </dl>
    `;
  }

  // --------------------------------------------------------------------
  // Run inspection
  // --------------------------------------------------------------------
  function delay(ms) { return new Promise((r) => setTimeout(r, ms)); }

  async function animateRailProgress() {
    const fill = $("#railTrackFill");
    const stepDelay = 190;
    for (let i = 0; i < STAGES.length; i++) {
      const stage = STAGES[i];
      const node = $(`#node-${stage.id}`);
      node.classList.add("pending");
      await delay(stepDelay);
      node.classList.remove("pending");
      node.classList.add("active");
      if (fill) fill.style.width = `${((i + 1) / STAGES.length) * 100}%`;
    }
  }

  function finalizeRail(result) {
    const telemetry = result.telemetry || {};
    const earlyBlock = !!telemetry.early_block;

    STAGES.forEach((stage) => {
      const node = $(`#node-${stage.id}`);
      const scoreEl = $(`#score-${stage.id}`);
      node.classList.remove("active", "pending");

      const skipped = earlyBlock && !["stage1", "stage4", "stage5"].includes(stage.id);
      if (skipped) {
        node.classList.add("skipped");
        scoreEl.textContent = "skipped";
        return;
      }

      const t = stage.telemetryKey ? telemetry[stage.telemetryKey] : null;

      if (stage.id === "stage4") {
        const cls = DECISION_CLASS[result.decision] || riskClass(result.composite_score);
        node.classList.add(cls);
        scoreEl.textContent = `${result.composite_score.toFixed(1)}`;
        return;
      }
      if (stage.id === "stage5") {
        node.classList.add(DECISION_CLASS[result.decision] || "status-clean");
        scoreEl.textContent = result.ticket_id ? "queued" : "logged";
        return;
      }

      const risk = stage.risk(t, result);
      if (risk === null || risk === undefined) {
        node.classList.add("status-clean");
        scoreEl.textContent = "pass";
      } else {
        node.classList.add(riskClass(risk));
        scoreEl.textContent = `${Number(risk).toFixed(1)}`;
      }
    });

    const fill = $("#railTrackFill");
    if (fill) fill.style.width = "100%";
  }

  async function runInspection(promptText) {
    if (!promptText || !promptText.trim()) {
      toast("Enter a prompt first");
      return;
    }
    runBtn.classList.add("running");
    runBtnLabel.textContent = "Running…";
    resetRailVisual();
    railSub.textContent = "Streaming the prompt through all seven inspection points…";

    const animPromise = animateRailProgress();
    const fetchPromise = fetch(`${API}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: promptText }),
    }).then(async (r) => {
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed (${r.status})`);
      }
      return r.json();
    });

    try {
      const [, result] = await Promise.all([animPromise, fetchPromise]);
      lastFullResult = result;
      finalizeRail(result);
      renderVerdict(result);
      addHistory(promptText, result);
      railSub.textContent = "Click any node on the rail to inspect its raw telemetry.";
      refreshHealth();
      if (result.decision === "HITL") refreshHitlQueue();
    } catch (e) {
      railSub.textContent = `Inspection failed: ${e.message}`;
      toast(`Error: ${e.message}`);
    } finally {
      runBtn.classList.remove("running");
      runBtnLabel.textContent = "Run Inspection";
    }
  }

  function renderVerdict(result) {
    verdictPanel.hidden = false;
    const badge = $("#verdictBadge");
    badge.textContent = result.decision;
    badge.className = `verdict-badge ${result.decision}`;

    const t4 = result.telemetry?.stage4_arbitration;
    $("#verdictReason").textContent = t4?.reason || (result.telemetry?.stage1_pre_guardrails?.block_reason) || "";

    // Gauge needle: -90deg at score 0 -> +90deg at score 10
    const score = Math.max(0, Math.min(10, result.composite_score));
    const deg = -90 + (score / 10) * 180;
    $("#gaugeNeedle").style.transform = `rotate(${deg}deg)`;
    $("#gaugeScoreText").textContent = score.toFixed(1);
    $("#gaugeScoreText").parentElement.style.color = DECISION_COLOR_VAR[result.decision] || "var(--text-primary)";

    // Chips
    const chips = [];
    if (result.is_financial_trigger) chips.push(`<span class="vchip warn">⚑ Financial trigger</span>`);
    if (result.ticket_id) chips.push(`<span class="vchip warn">Ticket ${esc(result.ticket_id)}</span>`);
    chips.push(`<span class="vchip info">${esc(result.telemetry?.total_latency_ms ?? "—")} ms total</span>`);
    $("#verdictChips").innerHTML = chips.join("");

    // Breakdown bars — weighted contribution matching Config weights in models.py
    const breakdown = t4?.score_breakdown || {};
    const weights = { heuristic_risk: 0.25, stat_risk: 0.15, rag_risk: 0.35, judge_risk: 0.25 };
    const names = {
      heuristic_risk: "Heuristic (w 0.25)",
      stat_risk: "Statistical (w 0.15)",
      rag_risk: "RAG Grounding (w 0.35)",
      judge_risk: "AI Judge (w 0.25)",
    };
    const barsHtml = Object.keys(names).map((key) => {
      const raw = breakdown[key] ?? 0;
      const contribution = raw * (weights[key] ?? 0);
      const pct = Math.min(100, (raw / 10) * 100);
      const color = riskClass(raw) === "status-risk" ? "var(--block)" : riskClass(raw) === "status-caution" ? "var(--hitl)" : "var(--allow)";
      return `
        <div class="bar-row">
          <span class="bar-name">${names[key]}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
          <span class="bar-val">${raw.toFixed(1)}/10</span>
        </div>`;
    }).join("");
    $("#breakdownBars").innerHTML = barsHtml || `<p class="empty-hint">No breakdown available for this early-blocked request.</p>`;

    $("#deliveredResponse").textContent = result.final_response || "";
    $("#auditHashText").textContent = (result.audit_hash || "").slice(0, 24) + "…";
    $("#auditHashText").dataset.full = result.audit_hash || "";
    $("#latencyText").textContent = `SHA-256 chained · genesis-linked`;
  }

  function addHistory(promptText, result) {
    sessionHistory.unshift({ prompt: promptText, decision: result.decision, score: result.composite_score, result });
    sessionHistory = sessionHistory.slice(0, 25);
    const list = $("#historyList");
    list.innerHTML = sessionHistory.map((h, i) => `
      <div class="history-item" data-idx="${i}">
        <span class="h-dot" style="background:${DECISION_COLOR_VAR[h.decision]}"></span>
        <span class="h-text">${esc(h.prompt)}</span>
        <span class="h-score">${h.score.toFixed(1)}</span>
      </div>
    `).join("");
    list.querySelectorAll(".history-item").forEach((el) => {
      el.addEventListener("click", () => {
        const h = sessionHistory[Number(el.dataset.idx)];
        promptInput.value = h.prompt;
        lastFullResult = h.result;
        finalizeRail(h.result);
        renderVerdict(h.result);
        railSub.textContent = "Showing a past run from this session. Click a node for details.";
      });
    });
  }

  // --------------------------------------------------------------------
  // Scenarios
  // --------------------------------------------------------------------
  async function loadScenarios() {
    try {
      const res = await fetch(`${API}/api/scenarios`);
      scenarios = await res.json();
      $("#scenarioChips").innerHTML = scenarios.map((s) =>
        `<button class="chip" data-prompt="${esc(s.prompt)}" title="${esc(s.prompt)}">${esc(s.label)}</button>`
      ).join("");
      $("#scenarioChips").querySelectorAll(".chip").forEach((chip) => {
        chip.addEventListener("click", () => {
          promptInput.value = chip.dataset.prompt;
          promptInput.focus();
        });
      });
    } catch (e) {
      $("#scenarioChips").innerHTML = `<span class="chip-loading">Could not load scenarios (${e.message})</span>`;
    }
  }

  // --------------------------------------------------------------------
  // Health / status pills
  // --------------------------------------------------------------------
  async function refreshHealth() {
    try {
      const res = await fetch(`${API}/api/audit-log?limit=1`);
      const data = await res.json();
      const dot = $("#chainStatusDot");
      const text = $("#chainStatusText");
      dot.className = `dot ${data.chain_valid ? "dot-green" : "dot-red"}`;
      text.textContent = data.chain_valid ? `Chain verified · ${data.total_entries} logged` : "Chain integrity FAILED";
    } catch (e) {
      $("#chainStatusText").textContent = "Ledger unreachable";
    }
    try {
      const res2 = await fetch(`${API}/api/hitl-queue`);
      const data2 = await res2.json();
      const n = data2.pending.length;
      $("#hitlStatusText").textContent = `${n} pending`;
      const badge = $("#hitlTabBadge");
      badge.textContent = n;
      badge.hidden = n === 0;
    } catch (e) { /* noop */ }
  }

  // --------------------------------------------------------------------
  // Ledger tab
  // --------------------------------------------------------------------
  async function refreshLedger() {
    const list = $("#ledgerList");
    try {
      const res = await fetch(`${API}/api/audit-log?limit=100`);
      const data = await res.json();
      if (!data.entries.length) {
        list.innerHTML = `<p class="empty-hint">No audit entries yet. Run an inspection from the Live tab.</p>`;
        return;
      }
      list.innerHTML = data.entries.map((e, i) => `
        <div class="ledger-entry">
          <div class="ledger-rail">
            <span class="node-dot" style="background:${DECISION_COLOR_VAR[e.decision] || "var(--text-dim)"}"></span>
            <span class="node-line"></span>
          </div>
          <div class="ledger-card" data-idx="${i}">
            <div class="ledger-card-top">
              <span class="ledger-card-decision ${e.decision}">${e.decision}</span>
              <span class="ledger-card-score">score ${Number(e.composite_score).toFixed(2)}/10 ${e.is_financial_trigger ? "· $ trigger" : ""}</span>
              <span class="ledger-card-time">${new Date(e.timestamp).toLocaleString()}</span>
            </div>
            <div class="ledger-card-hash">entry ${e.entry_hash.slice(0, 20)}… ← prev ${e.prev_hash.slice(0, 20)}…</div>
            <div class="ledger-card-detail">
              <div><strong>Entry ID:</strong> ${esc(e.entry_id)}</div>
              <div><strong>Prompt hash:</strong> ${esc(e.prompt_hash)}</div>
              <pre>${esc(JSON.stringify(e.trace, null, 2)).slice(0, 4000)}</pre>
            </div>
          </div>
        </div>
      `).join("");
      list.querySelectorAll(".ledger-card").forEach((card) => {
        card.addEventListener("click", () => card.classList.toggle("expanded"));
      });
    } catch (e) {
      list.innerHTML = `<p class="empty-hint">Could not load ledger: ${esc(e.message)}</p>`;
    }
  }

  async function verifyChain() {
    const banner = $("#verifyBanner");
    banner.hidden = false;
    banner.className = "verify-banner";
    banner.textContent = "Verifying SHA-256 chain…";
    try {
      const res = await fetch(`${API}/api/audit-log/verify`, { method: "POST" });
      const data = await res.json();
      banner.className = `verify-banner ${data.chain_valid ? "ok" : "fail"}`;
      banner.textContent = (data.chain_valid ? "✓ " : "✗ ") + data.verification_message;
      refreshHealth();
    } catch (e) {
      banner.className = "verify-banner fail";
      banner.textContent = `Verification request failed: ${e.message}`;
    }
  }

  // --------------------------------------------------------------------
  // HITL tab
  // --------------------------------------------------------------------
  async function refreshHitlQueue() {
    try {
      const res = await fetch(`${API}/api/hitl-queue`);
      const data = await res.json();

      $("#pendingCount").textContent = data.pending.length;

      const tuning = data.tuning_metrics || {};
      $("#tuningStrip").innerHTML = `
        <div class="tuning-metric"><span class="t-val">${tuning.total_reviews ?? 0}</span><span class="t-label">Total reviews</span></div>
        <div class="tuning-metric"><span class="t-val">${tuning.approval_rate !== undefined ? (tuning.approval_rate * 100).toFixed(0) + "%" : "—"}</span><span class="t-label">Approval rate</span></div>
        <div class="tuning-metric"><span class="t-val">${tuning.override_rate !== undefined ? (tuning.override_rate * 100).toFixed(0) + "%" : "—"}</span><span class="t-label">Override rate</span></div>
        <div class="tuning-metric"><span class="t-val">${tuning.recommended_allow_threshold_adjustment ?? "—"}</span><span class="t-label">Suggested threshold Δ</span></div>
      `;

      const pendingList = $("#pendingTicketList");
      pendingList.innerHTML = data.pending.length ? data.pending.map((t) => `
        <div class="ticket-card" data-id="${esc(t.ticket_id)}">
          <div class="ticket-card-top">
            <span class="ticket-id">${esc(t.ticket_id)}</span>
            <span class="ticket-score">risk ${Number(t.composite_score).toFixed(2)}/10 ${t.is_financial_trigger ? "· $ trigger" : ""}</span>
          </div>
          <div class="ticket-reason">${esc(t.reason)}</div>
          <div class="ticket-prompt">${esc(t.prompt)}</div>
          <div class="ticket-actions">
            <button class="btn btn-tiny resolve-btn" data-action="APPROVE" style="border-color:var(--allow);color:var(--allow)">Approve</button>
            <button class="btn btn-tiny resolve-btn" data-action="EDIT" style="border-color:var(--accent);color:var(--accent)">Edit &amp; release</button>
            <button class="btn btn-tiny resolve-btn" data-action="OVERRIDE" style="border-color:var(--block);color:var(--block)">Override &amp; block</button>
          </div>
        </div>
      `).join("") : `<p class="empty-hint">Queue is empty. Try the “Financial Trigger” or “Ungrounded Hallucination” scenario from Live Inspection.</p>`;

      pendingList.querySelectorAll(".resolve-btn").forEach((btn) => {
        btn.addEventListener("click", () => resolveTicket(btn.closest(".ticket-card").dataset.id, btn.dataset.action));
      });

      const resolvedList = $("#resolvedTicketList");
      resolvedList.innerHTML = data.resolved.length ? data.resolved.map((t) => `
        <div class="ticket-card">
          <div class="ticket-card-top">
            <span class="ticket-id">${esc(t.ticket_id)}</span>
            <span class="ticket-status ${t.status}">${esc(t.status)}</span>
          </div>
          <div class="ticket-reason">${esc(t.reviewer_notes || "")}</div>
          <div class="ticket-prompt">${esc(t.final_delivered_text || "")}</div>
        </div>
      `).join("") : `<p class="empty-hint">No resolved tickets yet.</p>`;

      refreshHealth();
    } catch (e) {
      toast(`Could not load HITL queue: ${e.message}`);
    }
  }

  async function resolveTicket(ticketId, action) {
    try {
      const res = await fetch(`${API}/api/hitl-resolve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticket_id: ticketId, action }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || "Resolve failed");
      toast(`Ticket ${ticketId} resolved: ${action}`);
      refreshHitlQueue();
    } catch (e) {
      toast(`Error: ${e.message}`);
    }
  }

  // --------------------------------------------------------------------
  // Tabs
  // --------------------------------------------------------------------
  function setupTabs() {
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((t) => { t.classList.remove("active"); t.setAttribute("aria-selected", "false"); });
        document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
        tab.classList.add("active");
        tab.setAttribute("aria-selected", "true");
        const target = document.getElementById(`view-${tab.dataset.tab}`);
        target.classList.add("active");
        if (tab.dataset.tab === "ledger") refreshLedger();
        if (tab.dataset.tab === "hitl") refreshHitlQueue();
      });
    });
  }

  // --------------------------------------------------------------------
  // Wire up
  // --------------------------------------------------------------------
  function init() {
    buildRail();
    setupTabs();
    loadScenarios();
    refreshHealth();

    runBtn.addEventListener("click", () => runInspection(promptInput.value));
    promptInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) runInspection(promptInput.value);
    });

    $("#verifyChainBtn").addEventListener("click", verifyChain);

    $("#copyHashBtn").addEventListener("click", () => {
      const full = $("#auditHashText").dataset.full;
      if (!full) return;
      navigator.clipboard?.writeText(full);
      toast("Audit hash copied");
    });

    $("#resetDemoBtn").addEventListener("click", async () => {
      if (!confirm("Clear the audit ledger and HITL queue? This cannot be undone.")) return;
      await fetch(`${API}/api/reset-demo`, { method: "POST" });
      sessionHistory = [];
      lastFullResult = null;
      $("#historyList").innerHTML = `<p class="empty-hint">Runs you trigger will appear here.</p>`;
      verdictPanel.hidden = true;
      resetRailVisual();
      railSub.textContent = "Awaiting first run — every response is checked at seven points before it reaches a user.";
      refreshHealth();
      refreshLedger();
      refreshHitlQueue();
      toast("Demo state reset");
    });

    setInterval(refreshHealth, 15000);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
