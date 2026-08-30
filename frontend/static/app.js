/**
 * frontend/static/app.js - Demonstration UI Interactive Controller
 * Handles Step-by-Step Pipeline Walkthrough, Scenario Presets, Custom Prompts, Theme Toggle, Sticky Stepper & HITL Triage
 */

// Global State
let currentPipelineData = null;
let currentStepIndex = 1;
const TOTAL_STEPS = 7;
let autoPlayInterval = null;
let activeScenarios = [];
let allKnowledgeDocs = [];
let activeHITLTicketId = null;

// Stage Metadata Definitions
const STAGE_META = {
    1: {
        title: "Stage 1: Pre-Execution Guardrails (pii.py)",
        desc: "Scanning input prompt for sensitive PII entities (Luhn validated) and adversarial prompt injection signatures (<1ms)",
        cardId: "card-stage-1",
        nodeId: "node-stage-1"
    },
    2: {
        title: "Stage 2: Primary LLM Generation & Context Retrieval (llm_client.py)",
        desc: "BM25 enterprise knowledge base retrieval + Corrective RAG (CRAG) retrieval confidence evaluation",
        cardId: "card-stage-2",
        nodeId: "node-stage-2"
    },
    3: {
        title: "Stage 3A: Fast Parallel Checks (fast_checks.py)",
        desc: "Concurrent scatter-gather worker bus (<20ms SLA) executing Heuristic Agent and Statistical Loop/Entropy Scorer",
        cardId: "card-stage-3a",
        nodeId: "node-stage-3a"
    },
    4: {
        title: "Stage 3B: RAG Grounding Verification (rag_verifier.py)",
        desc: "Exact numeric entity extraction, policy cross-referencing and semantic NLI claim entailment layer",
        cardId: "card-stage-3b",
        nodeId: "node-stage-3b"
    },
    5: {
        title: "Stage 3C: AI-as-a-Judge Evaluation (ai_judge.py)",
        desc: "Secondary compliance LLM with JSON mode evaluating demographic bias, tone hostility, and company policy breaches",
        cardId: "card-stage-3c",
        nodeId: "node-stage-3c"
    },
    6: {
        title: "Stage 4: Policy Arbitration & Risk Assessment (arbitrator.py)",
        desc: "Mathematical composite risk score calculation, FinCheck financial trigger check & 3-tier matrix routing",
        cardId: "card-stage-4",
        nodeId: "node-stage-4"
    },
    7: {
        title: "Stage 5: Delivered Output & Cryptographic Governance (audit_hitl.py)",
        desc: "Delivering finalized text, enqueuing HITL quarantine ticket if needed, and generating SHA-256 hash chained audit log",
        cardId: "card-stage-5",
        nodeId: "node-stage-5"
    }
};

// ============================================================================
// Initialization & Theme Management
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    lucide.createIcons();
    loadScenarios();
    checkHealth();
    loadHITLTickets();
    loadKnowledgeBase();
    loadAuditLogs();

    // Textarea char counter & Ctrl+Enter shortcut
    const promptInput = document.getElementById("prompt-input");
    if (promptInput) {
        promptInput.addEventListener("input", () => {
            const count = promptInput.value.length;
            document.getElementById("char-count").innerText = `${count} characters`;
        });

        promptInput.addEventListener("keydown", (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                e.preventDefault();
                startDemonstration(false); // Instant run
            }
        });
    }
});

function initTheme() {
    const savedTheme = localStorage.getItem("controlplane_theme") || "light";
    if (savedTheme === "dark") {
        document.documentElement.classList.add("dark");
        updateThemeUI(true);
    } else {
        document.documentElement.classList.remove("dark");
        updateThemeUI(false);
    }
}

function toggleTheme() {
    const isDark = document.documentElement.classList.toggle("dark");
    localStorage.setItem("controlplane_theme", isDark ? "dark" : "light");
    updateThemeUI(isDark);
}

function updateThemeUI(isDark) {
    const icon = document.getElementById("theme-icon");
    const label = document.getElementById("theme-label");
    if (!icon || !label) return;

    if (isDark) {
        label.innerText = "Dark Mode";
        icon.setAttribute("data-lucide", "moon");
        icon.className = "w-4 h-4 text-indigo-400";
    } else {
        label.innerText = "Bright Mode";
        icon.setAttribute("data-lucide", "sun");
        icon.className = "w-4 h-4 text-amber-500";
    }
    refreshIcons();
}

function refreshIcons() {
    setTimeout(() => {
        lucide.createIcons();
    }, 50);
}

// ============================================================================
// Custom Prompt Helper Functions
// ============================================================================

function clearPrompt() {
    const input = document.getElementById("prompt-input");
    input.value = "";
    document.getElementById("char-count").innerText = "0 characters";
    input.focus();
}

function setCustomPrompt(text) {
    const input = document.getElementById("prompt-input");
    input.value = text;
    document.getElementById("char-count").innerText = `${text.length} characters`;
    input.focus();

    // Visual highlight on textarea
    input.classList.add("border-indigo-500", "ring-2", "ring-indigo-500/40");
    setTimeout(() => {
        input.classList.remove("border-indigo-500", "ring-2", "ring-indigo-500/40");
    }, 600);
}

// ============================================================================
// Tab Navigation
// ============================================================================

function switchTab(tabKey) {
    const tabs = ["demo", "hitl", "kb", "audit", "arch"];
    tabs.forEach(t => {
        const btn = document.getElementById(`tab-btn-${t}`);
        const content = document.getElementById(`tab-content-${t}`);
        if (t === tabKey) {
            btn.classList.add("bg-indigo-600", "text-white", "shadow-sm");
            btn.classList.remove("text-slate-600", "dark:text-slate-400");
            content.classList.remove("hidden");
        } else {
            btn.classList.remove("bg-indigo-600", "text-white", "shadow-sm");
            btn.classList.add("text-slate-600", "dark:text-slate-400");
            content.classList.add("hidden");
        }
    });

    if (tabKey === "hitl") loadHITLTickets();
    if (tabKey === "kb") loadKnowledgeBase();
    if (tabKey === "audit") loadAuditLogs();

    refreshIcons();
}

// ============================================================================
// Health & Scenarios
// ============================================================================

async function checkHealth() {
    try {
        const res = await fetch("/api/health");
        const data = await res.json();
        if (data.has_groq_api_key) {
            document.getElementById("llm-status-text").innerText = "Groq Live API Active";
            document.getElementById("llm-mode-badge").classList.add("border-indigo-500/40", "bg-indigo-50", "dark:bg-indigo-950/30");
        } else {
            document.getElementById("llm-status-text").innerText = "Local Fallback Mode";
        }
    } catch (e) {
        console.error("Health check error:", e);
    }
}

async function loadScenarios() {
    try {
        const res = await fetch("/api/scenarios");
        activeScenarios = await res.json();
        const grid = document.getElementById("scenarios-grid");
        grid.innerHTML = "";

        activeScenarios.forEach((s) => {
            const badgeClasses = {
                emerald: "bg-emerald-100 text-emerald-800 border-emerald-300 dark:bg-emerald-500/20 dark:text-emerald-300 dark:border-emerald-500/30",
                amber: "bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-500/20 dark:text-amber-300 dark:border-amber-500/30",
                blue: "bg-sky-100 text-sky-800 border-sky-300 dark:bg-blue-500/20 dark:text-blue-300 dark:border-blue-500/30",
                purple: "bg-purple-100 text-purple-800 border-purple-300 dark:bg-purple-500/20 dark:text-purple-300 dark:border-purple-500/30",
                rose: "bg-rose-100 text-rose-800 border-rose-300 dark:bg-rose-500/20 dark:text-rose-300 dark:border-rose-500/30"
            }[s.badge_color] || "bg-indigo-100 text-indigo-800 border-indigo-300 dark:bg-indigo-500/20 dark:text-indigo-300 dark:border-indigo-500/30";

            const btn = document.createElement("button");
            btn.className = "glass-card-interactive text-left p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/70 hover:bg-slate-50 dark:hover:bg-slate-850 flex flex-col justify-between group shadow-sm";
            btn.onclick = () => selectScenario(s.id);

            btn.innerHTML = `
                <div>
                    <div class="flex items-center justify-between mb-1.5">
                        <span class="text-[10px] font-extrabold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">Scenario ${s.id}</span>
                        <span class="px-2 py-0.5 text-[9px] font-extrabold rounded-full border ${badgeClasses}">${s.expected_tier}</span>
                    </div>
                    <h4 class="font-extrabold text-xs text-slate-900 dark:text-slate-100 group-hover:text-indigo-600 dark:group-hover:text-indigo-300 transition-colors">${s.title}</h4>
                    <p class="text-[11px] text-slate-600 dark:text-slate-400 mt-1 line-clamp-2">${s.description}</p>
                </div>
                <div class="flex items-center space-x-1 text-[10px] font-bold text-indigo-600 dark:text-indigo-400 mt-2.5">
                    <span>Load Scenario</span>
                    <i data-lucide="arrow-right" class="w-3 h-3 group-hover:translate-x-1 transition-transform"></i>
                </div>
            `;
            grid.appendChild(btn);
        });
        refreshIcons();
    } catch (e) {
        console.error("Failed loading scenarios:", e);
    }
}

function selectScenario(scenarioId) {
    const s = activeScenarios.find(item => item.id === scenarioId);
    if (!s) return;

    setCustomPrompt(s.prompt);
}

// ============================================================================
// Pipeline Demonstration & Step Walkthrough
// ============================================================================

async function startDemonstration(isStepWalkthrough = true) {
    const promptInput = document.getElementById("prompt-input");
    const promptText = promptInput.value.trim();

    if (!promptText) {
        alert("Please enter a custom prompt or click one of the quick scenario buttons above.");
        promptInput.focus();
        return;
    }

    const execMode = document.getElementById("exec-mode-select").value;
    const btnStep = document.getElementById("btn-step-run");
    const btnInstant = document.getElementById("btn-instant-run");

    btnStep.disabled = true;
    btnInstant.disabled = true;
    btnStep.classList.add("opacity-60");
    btnInstant.classList.add("opacity-60");

    try {
        const res = await fetch("/api/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prompt: promptText,
                user_id: "demo_reviewer",
                execution_mode: execMode
            })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Pipeline execution failed");
        }

        const data = await res.json();
        currentPipelineData = data;

        // Populate all data fields
        populatePipelineData(data);

        // Show demonstration containers
        document.getElementById("demonstration-stepper-container").classList.remove("hidden");
        document.getElementById("pipeline-summary-panel").classList.remove("hidden");
        document.getElementById("stage-cards-container").classList.remove("hidden");

        if (isStepWalkthrough) {
            // Start at step 1 and allow step-by-step navigation
            jumpToStep(1);
        } else {
            // Instant mode: reveal all steps and jump to summary/stage 5
            jumpToStep(7);
        }

        refreshIcons();
    } catch (e) {
        alert(`Execution Error: ${e.message}`);
    } finally {
        btnStep.disabled = false;
        btnInstant.disabled = false;
        btnStep.classList.remove("opacity-60");
        btnInstant.classList.remove("opacity-60");
    }
}

function populatePipelineData(data) {
    const steps = data.steps;
    const s1 = steps.step1_guardrails;
    const s2 = steps.step2_generation;
    const s3a = steps.step3a_fast_checks;
    const s3b = steps.step3b_rag_grounding;
    const s3c = steps.step3c_ai_judge;
    const s4 = steps.step4_arbitration;
    const s5 = steps.step5_governance;

    // 1. Top Summary Panel
    const decisionEl = document.getElementById("summary-decision-text");
    const badgeIconEl = document.getElementById("summary-badge-icon");
    const scoreBarEl = document.getElementById("summary-score-bar");
    const reasonEl = document.getElementById("summary-reason-text");

    document.getElementById("summary-score-val").innerText = `${data.composite_score.toFixed(2)} / 10.0`;
    document.getElementById("summary-latency-val").innerText = `${data.total_latency_ms.toFixed(2)} ms`;
    document.getElementById("summary-hash-val").innerText = `${data.audit_hash.substring(0, 16)}...`;
    document.getElementById("summary-hash-val").title = data.audit_hash;
    document.getElementById("summary-path-val").innerText = `${data.active_path} PATH`;

    const scorePct = Math.min(100, Math.max(0, (data.composite_score / 10.0) * 100));
    scoreBarEl.style.width = `${scorePct}%`;

    decisionEl.innerText = data.decision;
    reasonEl.innerText = s4.reason || "Processed through arbitration matrix";

    if (data.decision === "ALLOW") {
        decisionEl.className = "text-lg font-black text-emerald-600 dark:text-emerald-400 tracking-tight";
        badgeIconEl.className = "w-12 h-12 rounded-xl flex items-center justify-center text-xl font-black bg-emerald-100 dark:bg-emerald-500/20 text-emerald-700 dark:text-emerald-400 border border-emerald-300 dark:border-emerald-500/30 pulse-emerald";
        badgeIconEl.innerText = "✓";
        scoreBarEl.className = "score-fill h-full bg-emerald-500";
    } else if (data.decision === "HITL") {
        decisionEl.className = "text-lg font-black text-amber-600 dark:text-amber-400 tracking-tight";
        badgeIconEl.className = "w-12 h-12 rounded-xl flex items-center justify-center text-xl font-black bg-amber-100 dark:bg-amber-500/20 text-amber-800 dark:text-amber-400 border border-amber-300 dark:border-amber-500/30 pulse-amber";
        badgeIconEl.innerText = "!";
        scoreBarEl.className = "score-fill h-full bg-amber-500";
    } else {
        decisionEl.className = "text-lg font-black text-rose-600 dark:text-rose-400 tracking-tight";
        badgeIconEl.className = "w-12 h-12 rounded-xl flex items-center justify-center text-xl font-black bg-rose-100 dark:bg-rose-500/20 text-rose-700 dark:text-rose-400 border border-rose-300 dark:border-rose-500/30 pulse-rose";
        badgeIconEl.innerText = "✕";
        scoreBarEl.className = "score-fill h-full bg-rose-500";
    }

    // 2. Stage 1 Details
    document.getElementById("s1-latency-badge").innerText = `${s1.latency_ms.toFixed(2)} ms`;
    document.getElementById("s1-sanitized-text").innerText = s1.sanitized_prompt;
    document.getElementById("s1-pii-count").innerText = `${s1.pii_detected.length} entities detected`;

    const piiListEl = document.getElementById("s1-pii-list");
    if (s1.pii_detected.length > 0) {
        piiListEl.innerHTML = s1.pii_detected.map(p => `
            <div class="flex items-center justify-between p-2 rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-xs">
                <span class="pii-chip">${p.entity_type}</span>
                <span class="font-mono text-[11px] text-slate-700 dark:text-slate-400">"${p.text}" [${p.start}:${p.end}]</span>
                <span class="sanitized-chip">REDACTED</span>
            </div>
        `).join("");
    } else {
        piiListEl.innerHTML = '<p class="text-slate-400 dark:text-slate-500 italic">No sensitive personal identifiers detected in prompt.</p>';
    }

    const injScoreBadge = document.getElementById("s1-injection-score-badge");
    injScoreBadge.innerText = `Score: ${s1.injection_score.toFixed(1)}/10`;
    if (s1.is_blocked) {
        injScoreBadge.className = "px-2 py-0.5 text-[10px] font-bold rounded bg-rose-100 dark:bg-rose-500/20 text-rose-700 dark:text-rose-400 border border-rose-300 dark:border-rose-500/30";
        document.getElementById("s1-injection-alert").innerHTML = `<span class="text-rose-600 dark:text-rose-400 font-extrabold">CRITICAL JAILBREAK DETECTED:</span> ${s1.block_reason} (Early Termination Triggered)`;
    } else if (s1.is_injection) {
        injScoreBadge.className = "px-2 py-0.5 text-[10px] font-bold rounded bg-amber-100 dark:bg-amber-500/20 text-amber-800 dark:text-amber-400 border border-amber-300 dark:border-amber-500/30";
        document.getElementById("s1-injection-alert").innerHTML = `<span class="text-amber-700 dark:text-amber-400 font-extrabold">MILD PROBE FLAGGED:</span> ${s1.block_reason}`;
    } else {
        injScoreBadge.className = "px-2 py-0.5 text-[10px] font-bold rounded bg-emerald-100 dark:bg-emerald-500/20 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-500/30";
        document.getElementById("s1-injection-alert").innerText = "Prompt is clean. No jailbreak signatures triggered.";
    }

    // 3. Stage 2 Details
    document.getElementById("s2-latency-badge").innerText = `${s2.latency_ms.toFixed(2)} ms`;
    document.getElementById("s2-crag-status").innerText = s2.crag_status;
    document.getElementById("s2-crag-conf").innerText = s2.crag_confidence.toFixed(2);
    document.getElementById("s2-candidate-text").innerText = s2.candidate_response || "(Skipped due to early guardrail block)";

    // 4. Stage 3A Details
    document.getElementById("s3a-latency-badge").innerText = `${s3a.latency_ms.toFixed(2)} ms`;
    document.getElementById("s3a-heur-risk-badge").innerText = `Risk: ${s3a.heuristic_risk.toFixed(2)}/10`;
    document.getElementById("s3a-banned-hits").innerText = s3a.banned_lexicon_hits.length > 0 ? s3a.banned_lexicon_hits.join(", ") : "None";
    document.getElementById("s3a-output-pii").innerText = s3a.output_pii.length > 0 ? s3a.output_pii.map(p => p.entity_type).join(", ") : "None detected";
    document.getElementById("s3a-stat-risk-badge").innerText = `Risk: ${s3a.stat_risk.toFixed(2)}/10`;
    document.getElementById("s3a-ngram-val").innerText = s3a.ngram_repetition.toFixed(3);
    document.getElementById("s3a-entropy-val").innerText = `${s3a.perplexity_score.toFixed(2)} bits`;
    document.getElementById("s3a-overlap-val").innerText = s3a.cosine_similarity.toFixed(2);

    // 5. Stage 3B Details
    document.getElementById("s3b-latency-badge").innerText = `${s3b.latency_ms.toFixed(2)} ms`;
    document.getElementById("s3b-grounding-score").innerText = `${s3b.grounding_score.toFixed(1)} / 10.0`;
    document.getElementById("s3b-verif-status").innerText = s3b.verification_status;
    document.getElementById("s3b-rag-risk").innerText = `${s3b.rag_risk.toFixed(1)} / 10.0`;

    const chunksListEl = document.getElementById("s3b-chunks-list");
    if (s3b.retrieved_chunks && s3b.retrieved_chunks.length > 0) {
        chunksListEl.innerHTML = s3b.retrieved_chunks.map(c => `
            <div class="p-2.5 rounded-lg bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-[11px] shadow-xs">
                <div class="font-bold text-indigo-700 dark:text-indigo-300">[${c.doc_id}] ${c.title}</div>
                <div class="text-slate-600 dark:text-slate-400 truncate mt-0.5">${c.content}</div>
            </div>
        `).join("");
    } else {
        chunksListEl.innerHTML = '<p class="text-slate-400 dark:text-slate-500 italic">No knowledge chunks retrieved (General inquiry).</p>';
    }

    const mismatchesListEl = document.getElementById("s3b-mismatches-list");
    if (s3b.numeric_mismatches && s3b.numeric_mismatches.length > 0) {
        mismatchesListEl.innerHTML = s3b.numeric_mismatches.map(m => `
            <div class="p-2 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800/40 text-rose-700 dark:text-rose-300 font-mono text-[11px] font-bold">
                ⚠ Mismatch: "${m}"
            </div>
        `).join("");
    } else {
        mismatchesListEl.innerHTML = '<p class="text-slate-400 dark:text-slate-500 italic">All numerical terms align with authoritative policies.</p>';
    }

    // 6. Stage 3C Details
    document.getElementById("s3c-latency-badge").innerText = `${s3c.latency_ms.toFixed(2)} ms`;
    document.getElementById("s3c-bias-val").innerText = `${s3c.bias_score.toFixed(1)} / 10.0`;
    document.getElementById("s3c-tone-val").innerText = `${s3c.tone_score.toFixed(1)} / 10.0`;
    document.getElementById("s3c-policy-val").innerText = `${s3c.policy_risk_score.toFixed(1)} / 10.0`;
    document.getElementById("s3c-notes-val").innerText = s3c.judge_notes || "Evaluated clean across compliance dimensions.";

    // 7. Stage 4 Details
    document.getElementById("s4-latency-badge").innerText = `${s4.latency_ms.toFixed(2)} ms`;
    document.getElementById("s4-decision-badge").innerText = s4.decision;
    document.getElementById("s4-reason-text").innerText = s4.reason;

    const finBadge = document.getElementById("s4-fin-trigger-badge");
    if (s4.is_financial_trigger) {
        finBadge.innerText = "FINANCIAL TRIGGER (FORCED ESCALATION)";
        finBadge.className = "px-2 py-0.5 rounded text-[10px] font-black bg-amber-100 dark:bg-amber-500/20 text-amber-800 dark:text-amber-300 border border-amber-300 dark:border-amber-500/30";
    } else {
        finBadge.innerText = "NO FINANCIAL TRIGGER";
        finBadge.className = "px-2 py-0.5 rounded text-[10px] font-bold bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-400";
    }

    const bd = s4.score_breakdown || {};
    document.getElementById("s4-breakdown-list").innerHTML = `
        <div>• Heuristic Risk: ${(bd.heuristic_risk || 0).toFixed(2)} &times; 0.25 = ${( (bd.heuristic_risk || 0) * 0.25 ).toFixed(2)}</div>
        <div>• Statistical Risk: ${(bd.stat_risk || 0).toFixed(2)} &times; 0.15 = ${( (bd.stat_risk || 0) * 0.15 ).toFixed(2)}</div>
        <div>• RAG Risk: ${(bd.rag_risk || 0).toFixed(2)} &times; 0.35 = ${( (bd.rag_risk || 0) * 0.35 ).toFixed(2)}</div>
        <div>• AI Judge Risk: ${(bd.judge_risk || 0).toFixed(2)} &times; 0.25 = ${( (bd.judge_risk || 0) * 0.25 ).toFixed(2)}</div>
    `;

    // 8. Stage 5 Details
    document.getElementById("s5-latency-badge").innerText = `${s5.latency_ms.toFixed(2)} ms`;
    document.getElementById("s5-delivered-text").innerText = s5.final_delivered_response;
    document.getElementById("s5-hash-full").innerText = s5.audit_hash;

    const quarantineBox = document.getElementById("s5-quarantine-box");
    if (s5.quarantined_ticket_id) {
        quarantineBox.classList.remove("hidden");
        document.getElementById("s5-ticket-id").innerText = s5.quarantined_ticket_id;
        activeHITLTicketId = s5.quarantined_ticket_id;
    } else {
        quarantineBox.classList.add("hidden");
        activeHITLTicketId = null;
    }
}

// ============================================================================
// Stepper Navigation Functions
// ============================================================================

function jumpToStep(stepIdx) {
    if (stepIdx < 1 || stepIdx > TOTAL_STEPS) return;
    currentStepIndex = stepIdx;
    updateStepperView();
}

function nextStep() {
    if (currentStepIndex < TOTAL_STEPS) {
        currentStepIndex++;
        updateStepperView();
    }
}

function prevStep() {
    if (currentStepIndex > 1) {
        currentStepIndex--;
        updateStepperView();
    }
}

function updateStepperView() {
    const meta = STAGE_META[currentStepIndex];
    if (!meta) return;

    // Update Header Text
    document.getElementById("current-step-title").innerText = meta.title;
    document.getElementById("current-step-desc").innerText = meta.desc;

    // Update Prev / Next buttons state
    document.getElementById("btn-prev-step").disabled = (currentStepIndex === 1);
    const nextBtn = document.getElementById("btn-next-step");
    if (currentStepIndex === TOTAL_STEPS) {
        nextBtn.innerHTML = `<span>Complete</span> <i data-lucide="check" class="w-3.5 h-3.5"></i>`;
    } else {
        nextBtn.innerHTML = `<span>Next Stage</span> <i data-lucide="chevron-right" class="w-3.5 h-3.5"></i>`;
    }

    // Update Stage Nodes
    for (let i = 1; i <= TOTAL_STEPS; i++) {
        const m = STAGE_META[i];
        const nodeEl = document.getElementById(m.nodeId);
        const statusEl = document.getElementById(`node-status-${i === 3 ? '3a' : i === 4 ? '3b' : i === 5 ? '3c' : i}`);
        const cardEl = document.getElementById(m.cardId);

        if (!nodeEl || !cardEl) continue;

        if (i === currentStepIndex) {
            nodeEl.className = "stage-node active-stage p-2.5 rounded-xl text-center shadow-md";
            if (statusEl) statusEl.innerText = "Inspecting";
            cardEl.classList.add("stage-card-spotlight");

            // Smart smooth scroll directly below the sticky frozen stepper header
            const stepperEl = document.getElementById("demonstration-stepper-container");
            const stepperHeight = stepperEl ? stepperEl.offsetHeight : 140;
            const navHeaderHeight = 64;
            const topOffset = navHeaderHeight + stepperHeight + 16;

            const cardRect = cardEl.getBoundingClientRect();
            const targetScrollY = window.pageYOffset + cardRect.top - topOffset;

            window.scrollTo({
                top: Math.max(0, targetScrollY),
                behavior: "smooth"
            });
        } else if (i < currentStepIndex) {
            nodeEl.className = "stage-node passed-stage p-2.5 rounded-xl text-center";
            if (statusEl) statusEl.innerText = "Passed ✓";
            cardEl.classList.remove("stage-card-spotlight");
        } else {
            nodeEl.className = "stage-node p-2.5 rounded-xl text-center";
            if (statusEl) statusEl.innerText = "Pending";
            cardEl.classList.remove("stage-card-spotlight");
        }
    }

    refreshIcons();
}

function toggleAutoPlay() {
    const icon = document.getElementById("autoplay-icon");
    const text = document.getElementById("autoplay-text");

    if (autoPlayInterval) {
        clearInterval(autoPlayInterval);
        autoPlayInterval = null;
        text.innerText = "Auto Play";
        icon.setAttribute("data-lucide", "play");
    } else {
        text.innerText = "Pause";
        icon.setAttribute("data-lucide", "pause");
        autoPlayInterval = setInterval(() => {
            if (currentStepIndex < TOTAL_STEPS) {
                nextStep();
            } else {
                toggleAutoPlay();
            }
        }, 1800);
    }
    refreshIcons();
}

function resetDemonstration() {
    if (autoPlayInterval) toggleAutoPlay();
    jumpToStep(1);
}

// ============================================================================
// HITL Review Queue Logic
// ============================================================================

async function loadHITLTickets() {
    try {
        const res = await fetch("/api/hitl/tickets");
        const data = await res.json();

        // Update counts
        document.getElementById("hitl-stat-pending").innerText = data.pending_count;
        document.getElementById("hitl-stat-total").innerText = data.metrics.total_reviews;
        document.getElementById("hitl-stat-approval").innerText = `${(data.metrics.allow_rate * 100).toFixed(1)}%`;
        
        const adj = data.metrics.recommended_allow_threshold_adjustment || 0.0;
        document.getElementById("hitl-stat-threshold").innerText = adj === 0 ? "Optimal Balance (±0.0)" : `Adjust ${adj > 0 ? '+' : ''}${adj}`;

        // Top nav badge
        const navBadge = document.getElementById("nav-pending-badge");
        if (data.pending_count > 0) {
            navBadge.innerText = data.pending_count;
            navBadge.classList.remove("hidden");
        } else {
            navBadge.classList.add("hidden");
        }

        const listEl = document.getElementById("hitl-tickets-list");
        if (data.all_tickets && data.all_tickets.length > 0) {
            listEl.innerHTML = data.all_tickets.map(t => {
                const isPending = (t.status === "PENDING");
                const statusBadge = isPending 
                    ? '<span class="px-2 py-0.5 text-[10px] font-bold rounded-full bg-amber-100 dark:bg-amber-500/20 text-amber-800 dark:text-amber-300 border border-amber-300 dark:border-amber-500/30">PENDING REVIEW</span>'
                    : `<span class="px-2 py-0.5 text-[10px] font-bold rounded-full bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700">${t.status}</span>`;

                return `
                    <div class="p-4 rounded-xl bg-white dark:bg-slate-900/80 border border-slate-200 dark:border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-3 shadow-sm">
                        <div class="space-y-1 max-w-2xl">
                            <div class="flex items-center space-x-2">
                                <span class="font-mono font-bold text-xs text-indigo-600 dark:text-indigo-400">${t.ticket_id}</span>
                                ${statusBadge}
                                ${t.is_financial_trigger ? '<span class="px-1.5 py-0.2 text-[9px] font-bold rounded bg-rose-100 dark:bg-rose-500/20 text-rose-700 dark:text-rose-300">FINANCIAL TRIGGER</span>' : ''}
                                <span class="text-[11px] text-slate-400">${t.timestamp ? t.timestamp.substring(0, 19) : ''}</span>
                            </div>
                            <p class="text-xs text-slate-800 dark:text-slate-200 font-bold">Prompt: "${t.prompt}"</p>
                            <p class="text-[11px] text-slate-600 dark:text-slate-400 truncate">Draft: "${t.candidate_response}"</p>
                            <p class="text-[11px] text-amber-700 dark:text-amber-400 font-medium">Reason: ${t.reason} (Score: ${t.composite_score.toFixed(2)}/10)</p>
                        </div>
                        ${isPending ? `
                            <button onclick="openHITLModal('${t.ticket_id}', '${escapeQuotes(t.prompt)}', '${escapeQuotes(t.candidate_response)}')" class="px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-bold whitespace-nowrap shadow-sm">
                                Review &amp; Resolve
                            </button>
                        ` : `
                            <span class="text-xs text-slate-500 italic">Resolved: ${t.reviewer_notes || 'No notes'}</span>
                        `}
                    </div>
                `;
            }).join("");
        } else {
            listEl.innerHTML = '<p class="text-xs text-slate-400 dark:text-slate-500 italic">No tickets in the HITL review queue.</p>';
        }
        refreshIcons();
    } catch (e) {
        console.error("Failed loading HITL tickets:", e);
    }
}

function escapeQuotes(str) {
    if (!str) return "";
    return str.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

function openHITLModal(ticketId, promptText, candidateText) {
    activeHITLTicketId = ticketId;
    document.getElementById("modal-ticket-id").innerText = ticketId;
    document.getElementById("modal-prompt-text").innerText = promptText;
    document.getElementById("modal-candidate-text").value = candidateText;
    document.getElementById("modal-reviewer-notes").value = "";
    document.getElementById("hitl-action-modal").classList.remove("hidden");
    refreshIcons();
}

function openHITLModalFromDemo() {
    if (!activeHITLTicketId || !currentPipelineData) return;
    openHITLModal(
        activeHITLTicketId,
        currentPipelineData.prompt,
        currentPipelineData.steps.step2_generation.candidate_response
    );
}

function closeHITLModal() {
    document.getElementById("hitl-action-modal").classList.add("hidden");
    activeHITLTicketId = null;
}

async function submitHITLResolution(action) {
    if (!activeHITLTicketId) return;

    const notes = document.getElementById("modal-reviewer-notes").value.trim();
    const editedText = document.getElementById("modal-candidate-text").value.trim();

    try {
        const res = await fetch("/api/hitl/resolve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                ticket_id: activeHITLTicketId,
                action: action,
                reviewer_notes: notes || `Resolved as ${action}`,
                edited_text: (action === "EDIT") ? editedText : null
            })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Failed resolving ticket");
        }

        closeHITLModal();
        loadHITLTickets();
        alert(`Ticket '${activeHITLTicketId}' successfully resolved via ${action}!`);
    } catch (e) {
        alert(`Error: ${e.message}`);
    }
}

// ============================================================================
// Knowledge Base Explorer Logic
// ============================================================================

async function loadKnowledgeBase() {
    try {
        const res = await fetch("/api/kb");
        const data = await res.json();
        allKnowledgeDocs = data.documents;

        // Populate category dropdown
        const catSelect = document.getElementById("kb-category-filter");
        catSelect.innerHTML = '<option value="all">All Categories</option>' + 
            data.categories.map(c => `<option value="${c}">${c.toUpperCase()}</option>`).join("");

        renderKnowledgeDocs(allKnowledgeDocs);
    } catch (e) {
        console.error("Failed loading KB:", e);
    }
}

function renderKnowledgeDocs(docs) {
    const grid = document.getElementById("kb-docs-grid");
    if (!docs || docs.length === 0) {
        grid.innerHTML = '<p class="text-xs text-slate-400 dark:text-slate-500 italic">No matching policy documents found.</p>';
        return;
    }

    grid.innerHTML = docs.map(d => `
        <div class="glass-card-interactive p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white/90 dark:bg-slate-900/60 space-y-2 shadow-sm">
            <div class="flex items-center justify-between">
                <span class="font-mono text-xs font-bold text-indigo-600 dark:text-indigo-400">[${d.doc_id}]</span>
                <span class="px-2 py-0.5 text-[9px] font-bold rounded-full bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 uppercase border border-slate-200 dark:border-slate-700">${d.category}</span>
            </div>
            <h4 class="font-extrabold text-xs text-slate-900 dark:text-slate-100">${d.title}</h4>
            <p class="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed">${d.content}</p>
            <div class="pt-2 border-t border-slate-100 dark:border-slate-800/80 flex flex-wrap gap-1">
                ${(d.keywords || []).slice(0, 5).map(k => `
                    <span class="px-1.5 py-0.5 text-[9px] font-mono bg-indigo-50 dark:bg-indigo-950/40 text-indigo-700 dark:text-indigo-300 rounded border border-indigo-200 dark:border-indigo-800/30">${k}</span>
                `).join("")}
            </div>
        </div>
    `).join("");
    refreshIcons();
}

function filterKnowledgeBase() {
    const query = document.getElementById("kb-search-input").value.toLowerCase().trim();
    const cat = document.getElementById("kb-category-filter").value;

    let filtered = allKnowledgeDocs;
    if (cat !== "all") {
        filtered = filtered.filter(d => d.category.toLowerCase() === cat.toLowerCase());
    }
    if (query) {
        filtered = filtered.filter(d => 
            d.title.toLowerCase().includes(query) || 
            d.content.toLowerCase().includes(query) ||
            (d.keywords && d.keywords.some(k => k.toLowerCase().includes(query)))
        );
    }
    renderKnowledgeDocs(filtered);
}

// ============================================================================
// Cryptographic Audit Verification Logic
// ============================================================================

async function loadAuditLogs() {
    try {
        const res = await fetch("/api/audit/logs?limit=15");
        const data = await res.json();
        const tableContainer = document.getElementById("audit-logs-table");

        if (!data.entries || data.entries.length === 0) {
            tableContainer.innerHTML = '<p class="text-xs text-slate-400 dark:text-slate-500 italic">No audit records logged yet.</p>';
            return;
        }

        tableContainer.innerHTML = `
            <table class="w-full text-left text-xs font-mono">
                <thead>
                    <tr class="border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400 font-bold">
                        <th class="py-2.5 px-3">Entry Hash (H_i)</th>
                        <th class="py-2.5 px-3">Previous Hash (H_i-1)</th>
                        <th class="py-2.5 px-3">Decision</th>
                        <th class="py-2.5 px-3">Score</th>
                        <th class="py-2.5 px-3">Timestamp</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-100 dark:divide-slate-850">
                    ${data.entries.map(e => `
                        <tr class="hover:bg-slate-50 dark:hover:bg-slate-900/50 transition-colors">
                            <td class="py-2.5 px-3 font-bold text-indigo-700 dark:text-indigo-300 truncate max-w-[160px]" title="${e.entry_hash}">${e.entry_hash.substring(0, 16)}...</td>
                            <td class="py-2.5 px-3 text-slate-400 dark:text-slate-500 truncate max-w-[160px]" title="${e.prev_hash}">${e.prev_hash.substring(0, 16)}...</td>
                            <td class="py-2.5 px-3 font-sans font-black ${e.decision === 'ALLOW' ? 'text-emerald-600 dark:text-emerald-400' : e.decision === 'HITL' ? 'text-amber-600 dark:text-amber-400' : 'text-rose-600 dark:text-rose-400'}">${e.decision}</td>
                            <td class="py-2.5 px-3 text-slate-800 dark:text-slate-200 font-bold">${e.composite_score.toFixed(2)}</td>
                            <td class="py-2.5 px-3 text-slate-400 dark:text-slate-500">${e.timestamp ? e.timestamp.substring(0, 19) : ''}</td>
                        </tr>
                    `).join("")}
                </tbody>
            </table>
        `;
        refreshIcons();
    } catch (e) {
        console.error("Failed loading audit logs:", e);
    }
}

async function runAuditVerification() {
    try {
        const res = await fetch("/api/audit/verify");
        const data = await res.json();

        const badge = document.getElementById("audit-verify-badge");
        const msg = document.getElementById("audit-verify-msg");

        if (data.valid) {
            badge.className = "px-2 py-0.5 text-[10px] font-bold rounded-full bg-emerald-100 dark:bg-emerald-500/20 text-emerald-800 dark:text-emerald-400 border border-emerald-300 dark:border-emerald-500/30";
            badge.innerText = `100% Valid (${data.entries_checked} Entries Checked)`;
            msg.innerHTML = `<span class="text-emerald-600 dark:text-emerald-400 font-bold">${data.message}</span>`;
            alert(`Cryptographic Verification PASSED: ${data.message}`);
        } else {
            badge.className = "px-2 py-0.5 text-[10px] font-bold rounded-full bg-rose-100 dark:bg-rose-500/20 text-rose-800 dark:text-rose-400 border border-rose-300 dark:border-rose-500/30";
            badge.innerText = "Tampering Detected";
            msg.innerHTML = `<span class="text-rose-600 dark:text-rose-400 font-bold">${data.message}</span>`;
            alert(`Cryptographic Verification FAILED: ${data.message}`);
        }
    } catch (e) {
        alert(`Verification Error: ${e.message}`);
    }
}
