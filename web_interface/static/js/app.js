/**
 * PseuDRAGON - Main Application JavaScript
 *
 * Handles UI interactions, API calls, and state management
 */

class PseuDRAGONApp {
    constructor() {
        // Application state management.
        // Application state.
        this.state = {
            isRunning: false,
            currentStep: "IDLE",
            selectedChoices: {},
            eventSource: null,
            reports: {
                stage1: null,
                stage2: null,
            },
            // Session ID for multi-client support.
            // Keep each client session separate.
            sessionId: null,
            // Session monitoring state.
            // Session monitoring state.
            monitoredSessions: [],
            selectedMonitorSession: null,
            sessionAutoRefresh: false,
            sessionAutoRefreshInterval: null,
            sessionLogOffset: 0,
        };

        // DOM element references.
        // DOM references.
        this.elements = {
            logsContent: document.getElementById("logsContent"),
            statusBadge: document.getElementById("statusBadge"),
            statusIndicator: document.getElementById("statusIndicator"),
            tableSelect: document.getElementById("tableSelect"),
            purposeGoalInput: document.getElementById("purposeGoalInput"),
            preferredMethodInput: document.getElementById("preferredMethodInput"),
            startBtn: document.getElementById("startBtn"),
            stopBtn: document.getElementById("stopBtn"),
            reopenPolicyBtn: document.getElementById("reopenPolicyBtn"),
            stage3Modal: document.getElementById("stage3Modal"),
            stage3Content: document.getElementById("stage3Content"),
            tabsContainer: document.getElementById("tabsContainer"),
            codeTabsContainer: document.getElementById("codeTabsContainer"),
            codeViewers: document.getElementById("codeViewers"),
            reportTabsContainer: document.getElementById("reportTabsContainer"),
            reportViewers: document.getElementById("reportViewers"),
            emptyState: document.getElementById("emptyState"),
            codeContent: document.getElementById("codeContent"),
            reportsContent: document.getElementById("reportsContent"),
            emptyStateReports: document.getElementById("emptyStateReports"),
            submitPoliciesBtn: document.getElementById("submitPoliciesBtn"),
            reviewBtn: document.getElementById("reviewBtn"),
            closeModalBtn: document.getElementById("closeModalBtn"),
            runtimeModeBadge: document.getElementById("runtimeModeBadge"),
            rankingMix: document.getElementById("rankingMix"),
            authorityHierarchy: document.getElementById("authorityHierarchy"),
            evidenceIndex: document.getElementById("evidenceIndex"),
            hitlRuleGate: document.getElementById("hitlRuleGate"),
            runtimeConfigNote: document.getElementById("runtimeConfigNote"),
        };

        // Debug: Check if critical elements exist.

        if (!this.elements.stage3Content) {
            console.error("stage3Content element not found!");
        }

        this.init();
    }

    init() {
        /**
         * Initialize the application
         */
        // Validate critical elements.

        this.validateElements();

        this.loadRuntimeConfig();
        this.loadTables();
        this.setupEventListeners();
        this.setupReviewAccessibility();

        // Hide reopen policy button and stop button initially.
        // Hide review and stop controls until a pipeline starts.
        if (this.elements.reopenPolicyBtn) {
            this.elements.reopenPolicyBtn.classList.add("hidden");
        }
        if (this.elements.stopBtn) {
            this.elements.stopBtn.classList.add("hidden");
        }

        this.log("PseuDRAGON initialized. Ready to start!", "info");
    }

    setupReviewAccessibility() {
        const modal = this.elements.stage3Modal;
        if (!modal) return;
        let wasOpen = false;
        let previousFocus = null;
        const background = document.querySelector(".app-container-ide");
        const header = document.querySelector(".app-header");
        new MutationObserver(() => {
            const open = modal.classList.contains("active");
            if (open === wasOpen) return;
            wasOpen = open;
            if (background) background.inert = open;
            if (header) header.inert = open;
            if (open) {
                previousFocus = document.activeElement;
                (
                    document.getElementById("reviewFieldSearch") || this.elements.closeModalBtn
                )?.focus();
            } else if (previousFocus?.isConnected) previousFocus.focus();
        }).observe(modal, { attributes: true, attributeFilter: ["class"] });
        modal.addEventListener("keydown", (event) => {
            if (!modal.classList.contains("active")) return;
            if (event.key === "Escape") {
                this.elements.closeModalBtn.click();
                return;
            }
            if (event.key !== "Tab") return;
            const controls = Array.from(
                modal.querySelectorAll('button, input, select, textarea, summary, [tabindex="0"]')
            ).filter((el) => !el.disabled && el.getClientRects().length);
            const first = controls[0],
                last = controls[controls.length - 1];
            if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last?.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first?.focus();
            }
        });
    }

    validateElements() {
        /**
         * Validate that all critical DOM elements exist
         */
        const criticalElements = [
            "stage3Modal",
            "stage3Content",
            "submitPoliciesBtn",
            "logsContent",
            "statusBadge",
        ];

        const missing = [];
        for (const key of criticalElements) {
            if (!this.elements[key]) {
                missing.push(key);
                console.error(`Missing element: ${key}`);
            }
        }

        if (missing.length > 0) {
            console.error("Missing critical elements:", missing);
            alert(`Error: Missing UI elements: ${missing.join(", ")}. Please refresh the page.`);
        }
    }

    setupEventListeners() {
        /**
         * Setup event listeners for UI interactions
         */
        if (this.elements.startBtn) {
            this.elements.startBtn.addEventListener("click", () => this.startPipeline());
        }

        if (this.elements.stopBtn) {
            this.elements.stopBtn.addEventListener("click", () => this.stopPipeline());
        }

        if (this.elements.reopenPolicyBtn) {
            this.elements.reopenPolicyBtn.addEventListener("click", () => this.reopenPolicyModal());
        }

        if (this.elements.reviewBtn) {
            this.elements.reviewBtn.addEventListener("click", () => {
                this.elements.stage3Modal.classList.add("active");
            });
        }

        document.querySelectorAll(".tab-nav-btn").forEach((btn) => {
            btn.addEventListener("click", (e) => {
                const tab = e.currentTarget.dataset.tab;
                this.switchMainTab(tab);

                // Load data when switching to certain tabs.
                if (tab === "heuristics") {
                    this.loadHeuristics();
                } else if (tab === "expert-feedback") {
                    this.loadExpertFeedback();
                } else if (tab === "audit") {
                    this.loadAuditLog();
                } else if (tab === "sessions") {
                    this.loadSessions();
                }
            });
        });

        // Session monitoring event listeners.
        // Bind session monitoring controls.
        const refreshSessionsBtn = document.getElementById("refreshSessionsBtn");
        if (refreshSessionsBtn) {
            refreshSessionsBtn.addEventListener("click", () => this.loadSessions());
        }

        const autoRefreshToggle = document.getElementById("autoRefreshToggle");
        if (autoRefreshToggle) {
            autoRefreshToggle.addEventListener("click", () => this.toggleSessionAutoRefresh());
        }

        // Add Table modal event listeners.
        // Bind the table creation dialog.
        const addTableBtn = document.getElementById("addTableBtn");
        if (addTableBtn) {
            addTableBtn.addEventListener("click", () => this.openAddTableModal());
        }

        const closeAddTableModalBtn = document.getElementById("closeAddTableModalBtn");
        if (closeAddTableModalBtn) {
            closeAddTableModalBtn.addEventListener("click", () => this.closeAddTableModal());
        }

        const cancelAddTableBtn = document.getElementById("cancelAddTableBtn");
        if (cancelAddTableBtn) {
            cancelAddTableBtn.addEventListener("click", () => this.closeAddTableModal());
        }

        const executeAddTableBtn = document.getElementById("executeAddTableBtn");
        if (executeAddTableBtn) {
            executeAddTableBtn.addEventListener("click", () => this.executeAddTable());
        }

        // Drop Table button event listener.
        // Bind the table deletion control.
        const dropTableBtn = document.getElementById("dropTableBtn");
        if (dropTableBtn) {
            dropTableBtn.addEventListener("click", () => this.confirmDropTable());
        }

        // Update Drop Table button state when table selection changes.

        if (this.elements.tableSelect) {
            this.elements.tableSelect.addEventListener("change", () =>
                this.updateDropTableButtonState()
            );
        }

        if (this.elements.submitPoliciesBtn) {
            this.elements.submitPoliciesBtn.addEventListener("click", () => this.submitPolicies());
        }

        if (this.elements.closeModalBtn) {
            this.elements.closeModalBtn.addEventListener("click", () => {
                this.elements.stage3Modal.classList.remove("active");
            });
        }

        // Heuristics modal event listeners.
        const addHeuristicBtn = document.getElementById("addHeuristicBtn");
        if (addHeuristicBtn) {
            addHeuristicBtn.addEventListener("click", () => {
                // Reset form for new heuristic.
                document.getElementById("heuristicId").value = "";
                document.getElementById("heuristicSource").value = "manual";
                document.getElementById("heuristicForm").reset();
                document.getElementById("heuristicPatternType").value = "";
                document.getElementById("heuristicPriority").value = "50";
                document.getElementById("heuristicEnabled").checked = true;

                // Hide source display and sample columns (for new manual heuristics)
                document.getElementById("heuristicSourceDisplay").style.display = "none";
                document.getElementById("heuristicSampleColumnsGroup").style.display = "none";

                document.getElementById("heuristicModalTitle").textContent =
                    "Add Deterministic Regex Rule";
                document.getElementById("heuristicModal").classList.add("active");
            });
        }

        const closeHeuristicModalBtn = document.getElementById("closeHeuristicModalBtn");
        if (closeHeuristicModalBtn) {
            closeHeuristicModalBtn.addEventListener("click", () => {
                document.getElementById("heuristicModal").classList.remove("active");
            });
        }

        const cancelHeuristicBtn = document.getElementById("cancelHeuristicBtn");
        if (cancelHeuristicBtn) {
            cancelHeuristicBtn.addEventListener("click", () => {
                document.getElementById("heuristicModal").classList.remove("active");
            });
        }

        const saveHeuristicBtn = document.getElementById("saveHeuristicBtn");
        if (saveHeuristicBtn) {
            saveHeuristicBtn.addEventListener("click", () => this.saveHeuristic());
        }

        const testRegexBtn = document.getElementById("testRegexBtn");
        if (testRegexBtn) {
            testRegexBtn.addEventListener("click", () => {
                const regex = document.getElementById("heuristicRegex").value;
                document.getElementById("regexTestPattern").value = regex;
                document.getElementById("regexTesterModal").classList.add("active");
            });
        }

        // Regex tester modal event listeners.
        const closeRegexTesterBtn = document.getElementById("closeRegexTesterBtn");
        if (closeRegexTesterBtn) {
            closeRegexTesterBtn.addEventListener("click", () => {
                document.getElementById("regexTesterModal").classList.remove("active");
            });
        }

        const closeRegexTesterModalBtn = document.getElementById("closeRegexTesterModalBtn");
        if (closeRegexTesterModalBtn) {
            closeRegexTesterModalBtn.addEventListener("click", () => {
                document.getElementById("regexTesterModal").classList.remove("active");
            });
        }

        const runRegexTestBtn = document.getElementById("runRegexTestBtn");
        if (runRegexTestBtn) {
            runRegexTestBtn.addEventListener("click", () => this.testRegexPattern());
        }

        // Audit log filter event listeners.
        const applyAuditFilters = document.getElementById("applyAuditFilters");
        if (applyAuditFilters) {
            applyAuditFilters.addEventListener("click", () => {
                const filters = {
                    event_type: document.getElementById("auditEventTypeFilter").value,
                    start_date: document.getElementById("auditStartDate").value,
                    end_date: document.getElementById("auditEndDate").value,
                };
                this.loadAuditLog(filters);
            });
        }

        // Expert feedback refresh button.
        const refreshExpertFeedback = document.getElementById("refreshExpertFeedback");
        if (refreshExpertFeedback) {
            refreshExpertFeedback.addEventListener("click", () => this.loadExpertFeedback());
        }

        // Table schema modal event listeners.
        // Bind the schema viewer.
        if (this.elements.tableSelect) {
            this.elements.tableSelect.addEventListener("dblclick", (e) => {
                // Get the table name from the clicked option or the selected option.
                // Resolve the table from the clicked or selected option.
                let tableName = null;

                if (e.target.tagName === "OPTION") {
                    // Double-clicked on an option element.
                    tableName = e.target.value;
                } else if (e.target.selectedOptions && e.target.selectedOptions.length > 0) {
                    // Double-clicked on the select element.
                    tableName = e.target.selectedOptions[0].value;
                }

                if (tableName && tableName !== "Loading tables...") {
                    this.showTableSchema(tableName);
                }
            });
        }

        const closeSchemaModalBtn = document.getElementById("closeSchemaModalBtn");
        if (closeSchemaModalBtn) {
            closeSchemaModalBtn.addEventListener("click", () => {
                document.getElementById("tableSchemaModal").classList.remove("active");
            });
        }

        const closeSchemaModalFooterBtn = document.getElementById("closeSchemaModalFooterBtn");
        if (closeSchemaModalFooterBtn) {
            closeSchemaModalFooterBtn.addEventListener("click", () => {
                document.getElementById("tableSchemaModal").classList.remove("active");
            });
        }

        // Submit and close controls dismiss the dialog.
    }

    async loadRuntimeConfig() {
        /**
         * Show the active retrieval route and its review requirement.
         */
        try {
            const response = await fetch("/api/runtime_config");
            const data = await response.json();

            if (!response.ok || data.status !== "success") {
                throw new Error(data.message || "Runtime profile is unavailable");
            }

            const ranking = data.ranking;
            const features = data.features;
            const sources = ranking.source_families || [];
            const mode = ranking.retrieval_mode;
            if (!["source_order", "legacy", "lexical"].includes(mode)) {
                throw new Error("Unknown retrieval mode");
            }

            if (this.elements.runtimeModeBadge) {
                const ragState = features.rag_enabled ? "RAG ON" : "RAG OFF";
                const rulesState = features.heuristics_enabled ? "RULES ON" : "RULES OFF";
                this.elements.runtimeModeBadge.textContent = `${ragState} · ${rulesState}`;
                this.elements.runtimeModeBadge.classList.toggle(
                    "runtime-state-disabled",
                    !features.rag_enabled || !features.heuristics_enabled
                );
            }

            if (this.elements.rankingMix) {
                if (mode === "source_order") {
                    this.elements.rankingMix.textContent = ranking.summary;
                } else if (mode === "lexical") {
                    this.elements.rankingMix.textContent = `Lexical retrieval · source gap ${Number(ranking.source_gap).toFixed(2)}`;
                } else {
                    const alpha = Number(ranking.alpha_semantic_similarity);
                    const authorityMix = Number(ranking.authority_mixing_weight);
                    this.elements.rankingMix.textContent = `Legacy · α=${alpha.toFixed(1)}: similarity ${Math.round(alpha * 100)}% + source preference ${Math.round(authorityMix * 100)}%`;
                }
            }

            if (this.elements.authorityHierarchy) {
                const compactLabels = {
                    approved_expert_feedback: "Feedback",
                    institutional_policy: "Institutional policy",
                    national_law: "National law",
                    international_regulation: "International guidance",
                };
                this.elements.authorityHierarchy.textContent = sources
                    .map((source) =>
                        mode === "legacy"
                            ? `${compactLabels[source.key] || source.label} ${Number(source.weight).toFixed(1)}`
                            : compactLabels[source.key] || source.label
                    )
                    .join(mode === "lexical" ? " · " : " → ");
                const label = this.elements.authorityHierarchy.previousElementSibling;
                if (label) {
                    label.textContent =
                        mode === "source_order"
                            ? "Search order"
                            : mode === "lexical"
                              ? "Evidence sources"
                              : "Source preference (legacy)";
                }
            }

            const cascade = document.querySelector("#runtimeProfile .runtime-cascade");
            if (cascade) {
                cascade.textContent = data.workflow.join(" → ");
            }

            if (this.elements.evidenceIndex) {
                const pdfCount = sources.reduce(
                    (total, source) =>
                        total +
                        (Number.isInteger(source.document_count) ? source.document_count : 0),
                    0
                );
                this.elements.evidenceIndex.textContent = `${pdfCount} PDFs · chunks ${data.retrieval.chunk_size}/${data.retrieval.chunk_overlap} · Top-${data.retrieval.top_k}`;
            }

            if (this.elements.hitlRuleGate) {
                this.elements.hitlRuleGate.textContent =
                    "Confirmed names → learned rules; conflicting approvals require a fresh judgment";
            }

            if (this.elements.runtimeConfigNote) {
                const rawRecordText = features.raw_records_accessed
                    ? "Raw records enabled"
                    : "Schema metadata only; no raw records";
                const auditText = features.llm_audit_enabled
                    ? "prompt/response audit ON"
                    : "prompt/response audit OFF";
                this.elements.runtimeConfigNote.textContent = `Stage 1: ${data.models.stage_1} · temperature ${Number(data.models.temperature).toFixed(1)} · endpoint ${data.models.endpoint_scope} · ${rawRecordText} · ${auditText} · ${ranking.review_note}`;
            }
        } catch (error) {
            if (this.elements.runtimeModeBadge) {
                this.elements.runtimeModeBadge.textContent = "CONFIG ERROR";
                this.elements.runtimeModeBadge.classList.add("runtime-state-disabled");
            }
            if (this.elements.rankingMix) {
                this.elements.rankingMix.textContent = "Unavailable";
            }
            if (this.elements.authorityHierarchy) {
                this.elements.authorityHierarchy.textContent = "Check server configuration";
            }
            if (this.elements.evidenceIndex) {
                this.elements.evidenceIndex.textContent = "Unavailable";
            }
            if (this.elements.hitlRuleGate) {
                this.elements.hitlRuleGate.textContent = "Unavailable";
            }
            if (this.elements.runtimeConfigNote) {
                this.elements.runtimeConfigNote.textContent = error.message;
            }
            console.error("Failed to load runtime profile:", error);
        }
    }

    async loadTables() {
        try {
            const response = await fetch("/get_tables");
            const data = await response.json();

            if (data.error) {
                this.log(`[ERROR] Error loading tables: ${data.error}`, "error");
                return;
            }

            this.elements.tableSelect.innerHTML = "";

            if (data.tables && data.tables.length > 0) {
                data.tables.forEach((table) => {
                    const option = document.createElement("option");
                    option.value = table;
                    option.textContent = table;
                    // Default to unselected - user must manually select tables.
                    // Require an explicit table selection.
                    option.selected = false;
                    this.elements.tableSelect.appendChild(option);
                });
                this.log(`[OK] Loaded ${data.tables.length} table(s)`, "success");
            } else {
                this.log("[WARN] No tables found in database", "warning");
            }
        } catch (error) {
            this.log(`[ERROR] Failed to load tables: ${error.message}`, "error");
        }
    }

    async startPipeline() {
        const selectedTables = Array.from(this.elements.tableSelect.selectedOptions).map(
            (opt) => opt.value
        );
        const purposeGoal = this.elements.purposeGoalInput.value.trim();
        const preferredMethod = this.elements.preferredMethodInput.value.trim();

        if (selectedTables.length === 0) {
            alert("Please select at least one table");
            return;
        }

        this.state.isRunning = true;
        this.updateStatus("RUNNING");
        this.elements.startBtn.disabled = true;
        this.elements.startBtn.classList.add("hidden");
        this.elements.stopBtn.classList.remove("hidden");
        this.clearTerminal();

        this.log("[Start] Starting PseuDRAGON pipeline...", "info");
        this.log(`[Stats] Selected tables: ${selectedTables.join(", ")}`, "info");
        if (purposeGoal) {
            this.log(`[Target] Pseudonymization purpose: ${purposeGoal}`, "info");
        }
        if (preferredMethod) {
            this.log(`[Tool] Preferred method: ${preferredMethod}`, "info");
        }

        try {
            this.log("[Wait] Sending start request to server...", "info");
            const response = await fetch("/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    tables: selectedTables,
                    purpose_goal: purposeGoal,
                    preferred_method: preferredMethod,
                    session_id: this.state.sessionId, // Include existing session ID if any.
                }),
            });

            this.log(`[Server] Server response status: ${response.status}`, "info");

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server returned ${response.status}: ${errorText}`);
            }

            const data = await response.json();
            this.log(`[Data] Server response data: ${JSON.stringify(data)}`, "info");

            if (data.status === "started") {
                // Store session ID from server response.
                // Retain the server-assigned session identifier.
                if (data.session_id) {
                    this.state.sessionId = data.session_id;
                    this.log(
                        `[Session] Session ID: ${this.state.sessionId.substring(0, 8)}...`,
                        "info"
                    );
                }
                this.log("[OK] Pipeline started on server. Connecting to event stream...", "info");
                this.connectSSE();
            } else {
                throw new Error("Failed to start pipeline: " + (data.status || "Unknown error"));
            }
        } catch (error) {
            this.log(`[ERROR] Error starting pipeline: ${error.message}`, "error");
            this.updateStatus("ERROR");
            this.elements.startBtn.disabled = false;
            this.state.isRunning = false;
        }
    }

    connectSSE() {
        if (this.state.eventSource) {
            this.state.eventSource.close();
        }

        this.log("[Connect] Establishing SSE connection...", "info");
        // Include session ID in SSE connection URL.
        // Associate the event stream with this session.
        const sseUrl = this.state.sessionId
            ? `/stream?session_id=${this.state.sessionId}`
            : "/stream";
        this.state.eventSource = new EventSource(sseUrl);

        this.state.eventSource.onopen = () => {
            this.log("[Active] SSE Connection opened.", "info");
        };

        this.state.eventSource.onmessage = (event) => {
            // this.log('📨 Received event: ' + event.data.substring(0, 50) + '...', 'stream'); // Too verbose.
            this.log(event.data, "stream");
        };

        this.state.eventSource.addEventListener("stage2_complete", () => {
            this.log("\n[OK] Stage 1 & 2 Complete!", "success");
            this.log("[User] Opening Human-in-the-Loop review...", "info");
            this.loadStage3Data();
        });

        this.state.eventSource.onerror = (err) => {
            console.error("SSE Error:", err);
            // Only log if we are supposed to be running.
            if (this.state.isRunning) {
                this.log("[WARN] SSE Connection error. Retrying or check console.", "warning");
            }
        };
    }

    async loadStage3Data() {
        /**
         * Load Stage 3 data for human-in-the-loop review
         */
        try {
            // Include session ID in request URL.
            // Associate the request with this session.
            const url = this.state.sessionId
                ? `/get_stage3_data?session_id=${this.state.sessionId}`
                : "/get_stage3_data";
            const response = await fetch(url);
            const data = await response.json();

            if (data.error) {
                this.log(`[ERROR] Error: ${data.error}`, "error");
                console.error("loadStage3Data: Error in response:", data.error);
                return;
            }

            // Store report filenames.
            // Retain the report filenames.
            if (data.reports) {
                this.state.reports = data.reports;
                await this.loadReports();
            }

            // Render modal with table data.
            // Render the review dialog from table metadata.
            const tableData = data.tables || data;
            this.renderStage3Modal(tableData);

            // Show modal.
            // Show the dialog.
            if (this.elements.stage3Modal) {
                this.elements.stage3Modal.classList.add("active");
            } else {
                console.error("stage3Modal element not found");
                this.log("[ERROR] Error: Cannot show Stage 3 modal - element not found", "error");
            }

            // Show Review button and Reopen Policy button.
            if (this.elements.reviewBtn) {
                this.elements.reviewBtn.style.display = "flex";
            }
            if (this.elements.reopenPolicyBtn) {
                this.elements.reopenPolicyBtn.classList.remove("hidden");
            }
        } catch (error) {
            console.error("Stage 3 load error:", error);
            this.log(`[ERROR] Failed to load Stage 3 data: ${error.message}`, "error");
        }
    }

    reopenPolicyModal() {
        /**
         * Reopen the Stage 3 policy selection modal
         */
        this.log("[Update] Reopening policy selection...", "info");

        if (this.elements.stage3Modal) {
            this.elements.stage3Modal.classList.add("active");
        } else {
            console.error("stage3Modal element not found");
            this.log("[ERROR] Error: Cannot show Stage 3 modal - element not found", "error");
        }
    }

    stopPipeline() {
        /**
         * Stop the running pipeline
         */
        if (!this.state.isRunning) {
            alert("No pipeline is currently running");
            return;
        }

        if (!confirm("Are you sure you want to stop the pipeline?")) {
            return;
        }

        this.log("⛔ Stopping pipeline...", "warning");

        // Call backend to stop pipeline with session ID.
        // Stop only this session’s pipeline.
        fetch("/stop", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                session_id: this.state.sessionId,
            }),
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.status === "stopping") {
                    this.log("[OK] Stop request sent to backend", "success");
                }
            })
            .catch((error) => {
                this.log(`[WARN] Error stopping pipeline: ${error}`, "error");
            });

        // Close event source if open.
        if (this.state.eventSource) {
            this.state.eventSource.close();
            this.state.eventSource = null;
        }

        // Reset state.
        this.state.isRunning = false;
        this.updateStatus("IDLE");
        this.elements.startBtn.disabled = false;
        this.elements.startBtn.classList.remove("hidden");
        this.elements.stopBtn.classList.add("hidden");

        // Hide reopen policy button as well.
        this.elements.reopenPolicyBtn.classList.add("hidden");

        this.log("⛔ Pipeline stopped by user", "warning");
    }

    renderStage3Modal(data) {
        /**
         * Render Stage 3 modal with policy options
         */
        if (!this.elements.stage3Content) {
            console.error("stage3Content element is null or undefined");
            this.log("[ERROR] Error: Cannot render Stage 3 modal - element not found", "error");
            return;
        }

        this.elements.stage3Content.innerHTML = "";
        this.state.selectedChoices = {};
        this.state.tableData = data; // Store table data for PII status changes.

        const validationSection = document.createElement("details");
        validationSection.className = "validation-section review-checks";
        validationSection.innerHTML = `
            <summary>Additional review checks</summary>
            <div class="validation-header">
                <p>Inspect the proposed policies before approval.</p>
                <button class="btn btn-validate" onclick="app.validatePolicies()">
                    Validate Policies
                </button>
            </div>
            <div id="validationResults" class="validation-results"></div>
        `;
        this.elements.stage3Content.appendChild(validationSection);
        const toolbar = document.createElement("div");
        toolbar.className = "review-toolbar";
        toolbar.innerHTML = `<label>Find a field<input type="search" id="reviewFieldSearch" placeholder="Table, field, or description"></label>
            <label>Show<select id="reviewFieldFilter"><option value="all">All fields</option><option value="pii">Sensitive fields</option><option value="mask">MASK selected</option></select></label>
            <span id="reviewFieldCount" role="status"></span>`;
        toolbar.addEventListener("input", () => this.filterReviewFields());
        this.elements.stage3Content.append(toolbar);

        for (const [tableName, columns] of Object.entries(data)) {
            const tableSection = this.createTableSection(tableName, columns);
            this.elements.stage3Content.appendChild(tableSection);
        }
        this.filterReviewFields();
    }

    filterReviewFields() {
        this.elements.stage3Content.scrollTop = 0;
        const query = (document.getElementById("reviewFieldSearch")?.value || "").toLowerCase();
        const filter = document.getElementById("reviewFieldFilter")?.value || "all";
        let visible = 0;
        const cards = this.elements.stage3Content.querySelectorAll(".column-card");
        cards.forEach((card) => {
            const info = this.state.tableData[card.dataset.table][card.dataset.column];
            const key = `method_${card.dataset.table}_${card.dataset.column}`;
            const method = info.methods[this.state.selectedChoices[key]];
            const matches =
                `${card.dataset.table} ${card.dataset.column} ${info.column_comment || ""}`
                    .toLowerCase()
                    .includes(query);
            card.hidden = !(
                matches &&
                (filter === "all" ||
                    (filter === "pii" && info.is_pii) ||
                    (filter === "mask" && method?.method === "MASK"))
            );
            if (!card.hidden) visible += 1;
        });
        this.elements.stage3Content.querySelectorAll(".table-section").forEach((section) => {
            section.hidden = !Array.from(section.querySelectorAll(".column-card")).some(
                (card) => !card.hidden
            );
        });
        const count = document.getElementById("reviewFieldCount");
        if (count) count.textContent = `${visible} of ${cards.length} fields`;
    }

    createTableSection(tableName, columns) {
        /**
         * Create a table section for Stage 3 modal
         */
        const section = document.createElement("div");
        section.className = "table-section";

        const header = document.createElement("div");
        header.className = "table-header";
        header.innerHTML = `<span>📊</span> <span>Table: ${this.escapeHtml(tableName)}</span>`;
        section.appendChild(header);

        for (const [columnName, info] of Object.entries(columns)) {
            const columnCard = this.createColumnCard(tableName, columnName, info);
            section.appendChild(columnCard);
        }

        return section;
    }

    createColumnCard(tableName, columnName, info) {
        /**
         * Create a column card with PII information and method options
         */
        const uniqueKey = `method_${tableName}_${columnName}`;
        this.state.selectedChoices[uniqueKey] = 0;

        const card = document.createElement("div");
        card.className = "column-card";
        card.dataset.table = tableName;
        card.dataset.column = columnName;

        const header = document.createElement("div");
        header.className = "column-header";

        // Determine current PII status.
        const isPii = info.is_pii;
        const currentStatus = isPii ? "PII" : "Non-PII";
        const badgeClass = isPii ? "pii-badge" : "pii-badge non-pii-badge";

        header.innerHTML = `
            <div class="column-name">
                <span>🔐</span>
                <span>${this.escapeHtml(columnName)}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <div class="${badgeClass}" data-status="${currentStatus}">
                    <span>⚠️</span>
                    <span class="pii-status-text">${info.pii_type}</span>
                </div>
                <button class="btn-change-pii-status"
                        title="Change PII Status"
                        data-table="${tableName}"
                        data-column="${columnName}"
                        data-current-status="${currentStatus}">
                    🔄 Change Status
                </button>
            </div>
        `;
        card.appendChild(header);

        // Add event listener to Change Status button (instead of inline onclick)
        const changeBtn = header.querySelector(".btn-change-pii-status");
        if (changeBtn) {
            changeBtn.addEventListener("click", (event) => {
                event.stopPropagation(); // Prevent event bubbling.
                event.preventDefault(); // Prevent default button behavior.
                const table = changeBtn.dataset.table;
                const column = changeBtn.dataset.column;
                const status = changeBtn.dataset.currentStatus;
                this.changePIIStatus(table, column, status);
            });
        } else {
            console.error("[createColumnCard] Change Status button not found in header!");
        }

        const meta = document.createElement("div");
        meta.className = "column-meta";
        // Check if identification evidence is valid and meaningful.
        // Display nonempty identification evidence.
        const showIdentification =
            info.id_evidence &&
            info.id_evidence !== "Unknown" &&
            info.id_evidence !== "None" &&
            info.id_evidence.trim() !== "";

        meta.innerHTML = `
            ${
                info.column_comment
                    ? `<div class="meta-item">
                <span class="meta-label">💬 Description:</span>
                <span class="meta-value">${this.escapeHtml(info.column_comment)}</span>
            </div>`
                    : ""
            }
            ${
                showIdentification
                    ? `<div class="meta-item">
                <span class="meta-label">📋 Identification:</span>
                <span class="meta-value">${this.escapeHtml(info.id_evidence)}</span>
            </div>`
                    : ""
            }
        `;
        card.appendChild(meta);

        const methodsContainer = document.createElement("div");
        methodsContainer.className = "methods-container";

        const methodsTitle = document.createElement("div");
        methodsTitle.className = "methods-title";
        methodsTitle.innerHTML = "<span>🛠️</span> <span>Select Pseudonymization Method:</span>";
        methodsContainer.appendChild(methodsTitle);

        info.methods.forEach((method, index) => {
            const option = this.createMethodOption(uniqueKey, method, index);
            methodsContainer.appendChild(option);
        });

        card.appendChild(methodsContainer);

        return card;
    }

    formatLegalEvidence(evidence) {
        if (!evidence) return "";
        return this.escapeHtml(evidence).replace(/\n/g, "<br>");
    }

    createMethodOption(uniqueKey, method, index) {
        const option = document.createElement("div");
        const selected = (this.state.selectedChoices[uniqueKey] ?? 0) === index;
        option.className = `method-option ${selected ? "selected" : ""}`;
        option.onclick = (event) => {
            if (!event.target.closest("textarea, select, details, button, .mask-editor, a"))
                this.selectMethod(uniqueKey, index);
        };
        const codeInputId = `code_${uniqueKey}_${index}`;
        const descInputId = `desc_${uniqueKey}_${index}`;
        const radioId = `radio_${uniqueKey}_${index}`;
        const isMask = String(method.method).toUpperCase() === "MASK";
        option.innerHTML = `
            <input type="radio" id="${radioId}" name="${uniqueKey}" value="${index}" class="method-radio" ${selected ? "checked" : ""}>
            <div class="method-content">
                <div class="method-header"><label class="method-name" for="${radioId}">${this.escapeHtml(method.method)}</label></div>
                <div class="method-details">
                    <div class="method-description-editable">
                        <label class="desc-label" for="${descInputId}">Treatment description</label>
                        <textarea id="${descInputId}" class="desc-edit-input" rows="2" data-original="${this.escapeHtml(method.description || "")}">${this.escapeHtml(method.description || "")}</textarea>
                    </div>
                    ${method.legal_evidence ? `<details class="method-legal-source"><summary>Supporting evidence</summary><div class="legal-source-content">${this.formatLegalEvidence(method.legal_evidence)}</div></details>` : ""}
                    ${
                        method.code_snippet || isMask
                            ? `<details class="advanced-code"><summary>Advanced: source code & contract</summary>
                        <div class="method-code"><label class="code-label" for="${codeInputId}">Transformation source code</label>
                            <textarea id="${codeInputId}" class="code-edit-input" rows="3" data-original="${this.escapeHtml(method.code_snippet || "")}">${this.escapeHtml(method.code_snippet || "")}</textarea>
                        </div>
                        <details class="method-contract"><summary>Declare an unsupported custom edit</summary>
                            <label>Method<select id="contract_method_${uniqueKey}_${index}">
                            ${["KEEP", "DELETE", "HASH", "MASK", "TOKENIZE", "GENERALIZE", "ROUND", "ENCRYPT"].map((name) => `<option value="${name}" ${name === String(method.method).toUpperCase() ? "selected" : ""}>${name}</option>`).join("")}</select></label>
                            <label>Parameters (JSON object)<textarea id="contract_parameters_${uniqueKey}_${index}" rows="2">${this.escapeHtml(JSON.stringify(method.parameters || {}))}</textarea></label>
                            <label><input type="checkbox" id="contract_confirm_${uniqueKey}_${index}">I confirm that the method and parameters describe my edited code. Custom declarations are recorded without automatic verification.</label>
                        </details>
                    </details>`
                            : ""
                    }
                </div>
            </div>`;
        option
            .querySelector(".method-radio")
            .addEventListener("change", () => this.selectMethod(uniqueKey, index));
        if (isMask)
            option
                .querySelector(".method-details")
                .prepend(this.createMaskEditor(uniqueKey, method, index));
        return option;
    }

    createMaskEditor(uniqueKey, method, index) {
        const editor = document.createElement("div");
        editor.className = "mask-editor";
        editor.dataset.dirty = "false";
        const id = `mask_${uniqueKey}_${index}`;
        const field = this.findReviewField(uniqueKey);
        const isAccount = ["DPS_ACNO", "WD_ACNO"].includes(field?.column);
        editor.innerHTML = `<div class="mask-controls">
            <label for="${id}_mode">Masking pattern<select id="${id}_mode" data-mask="mode">
                <option value="">Choose a pattern</option><option value="keep_last">Show last characters</option>
                <option value="keep_first">Show first characters</option><option value="mask_last">Hide last characters</option>
                <option value="mask_first">Hide first characters</option><option value="keep_edges">Show beginning and end</option><option value="mask_all">Hide everything</option>
            </select></label>
            <label data-mask-field="count" for="${id}_count">Character count<input id="${id}_count" data-mask="count" type="number" min="1" max="64" value="4"></label>
            <label data-mask-field="prefix_count" for="${id}_prefix">Visible at start<input id="${id}_prefix" data-mask="prefix_count" type="number" min="0" max="64" value="1"></label>
            <label data-mask-field="suffix_count" for="${id}_suffix">Visible at end<input id="${id}_suffix" data-mask="suffix_count" type="number" min="0" max="64" value="4"></label>
            <label for="${id}_char">Mask character<select id="${id}_char" data-mask="mask_char"><option value="*">*</option><option value="#">#</option><option value="•">•</option></select></label>
        </div>
        <div class="mask-preview" aria-live="polite"><span>Example · synthetic value</span><div><code>1234567890</code><span aria-hidden="true"> → </span><code data-mask-output>Choose a pattern</code></div><small>Missing values stay missing. Short values are fully masked if the visible part would reveal the whole value.</small></div>
        ${isAccount ? '<label class="mask-shared"><input type="checkbox" data-mask="shared" checked> Apply the same pattern to DPS_ACNO and WD_ACNO account fields in this review</label>' : ""}
        <div class="mask-actions"><button type="button" class="btn btn-secondary" data-mask-apply>Apply pattern</button><span class="mask-status" role="status"></span></div>
        <p class="mask-error" role="alert" hidden></p>`;
        const config = method.masking_rule;
        if (config) {
            for (const name of ["mode", "count", "prefix_count", "suffix_count", "mask_char"]) {
                if (config[name] !== undefined) {
                    const control = editor.querySelector(`[data-mask="${name}"]`);
                    if (
                        name === "mask_char" &&
                        !Array.from(control.options).some((x) => x.value === config[name])
                    )
                        control.add(new Option(config[name], config[name]));
                    control.value = config[name];
                }
            }
        }
        editor.querySelector(".mask-status").textContent = config
            ? "Pattern applied · pending approval"
            : "Current proposal retained until you apply a pattern";
        const update = (event) => {
            if (event) {
                editor.dataset.dirty = "true";
                editor.querySelector(".mask-status").textContent = "Unapplied changes";
            }
            this.previewMaskEditor(editor);
        };
        editor.addEventListener("input", update);
        editor.querySelector("[data-mask-apply]").onclick = () =>
            this.applyMaskPattern(uniqueKey, index, editor);
        update();
        return editor;
    }

    findReviewField(uniqueKey) {
        for (const [table, columns] of Object.entries(this.state.tableData || {})) {
            for (const column of Object.keys(columns))
                if (`method_${table}_${column}` === uniqueKey) return { table, column };
        }
        return null;
    }

    readMaskOptions(editor) {
        const value = (name) => editor.querySelector(`[data-mask="${name}"]`).value;
        return {
            mode: value("mode"),
            count: Number(value("count")),
            prefix_count: Number(value("prefix_count")),
            suffix_count: Number(value("suffix_count")),
            mask_char: value("mask_char"),
        };
    }

    previewMaskEditor(editor) {
        const settings = this.readMaskOptions(editor);
        const edges = settings.mode === "keep_edges";
        editor.querySelector('[data-mask-field="count"]').hidden =
            edges || settings.mode === "mask_all" || !settings.mode;
        editor.querySelector('[data-mask-field="prefix_count"]').hidden = !edges;
        editor.querySelector('[data-mask-field="suffix_count"]').hidden = !edges;
        const output = editor.querySelector("[data-mask-output]");
        const error = editor.querySelector(".mask-error");
        error.hidden = true;
        const chars = Array.from("1234567890"),
            n = chars.length,
            c = settings.count,
            m = settings.mask_char;
        const minimum = settings.mode.startsWith("keep_") ? 0 : 1;
        editor.querySelector('[data-mask="count"]').min = minimum;
        if (!settings.mode) {
            output.textContent = "Choose a pattern";
            return;
        }
        const counts = edges
            ? [settings.prefix_count, settings.suffix_count]
            : settings.mode === "mask_all"
              ? []
              : [c];
        if (counts.some((x) => !Number.isInteger(x) || x < (edges ? 0 : minimum) || x > 64)) {
            output.textContent = "Enter a valid count";
            return;
        }
        let masked;
        if (settings.mode === "mask_all") masked = m.repeat(n);
        if (settings.mode === "keep_last")
            masked = n <= c || c === 0 ? m.repeat(n) : m.repeat(n - c) + chars.slice(-c).join("");
        if (settings.mode === "keep_first")
            masked = n <= c || c === 0 ? m.repeat(n) : chars.slice(0, c).join("") + m.repeat(n - c);
        if (settings.mode === "mask_last")
            masked = chars.slice(0, Math.max(0, n - c)).join("") + m.repeat(Math.min(n, c));
        if (settings.mode === "mask_first")
            masked = m.repeat(Math.min(n, c)) + chars.slice(c).join("");
        if (edges) {
            const a = settings.prefix_count,
                b = settings.suffix_count;
            masked =
                n <= a + b
                    ? m.repeat(n)
                    : chars.slice(0, a).join("") +
                      m.repeat(n - a - b) +
                      (b ? chars.slice(-b).join("") : "");
        }
        output.textContent = masked || "";
    }

    async applyMaskPattern(uniqueKey, index, editor) {
        if (this.state.isApplyingMask) return;
        this.state.isApplyingMask = true;
        const field = this.findReviewField(uniqueKey);
        const error = editor.querySelector(".mask-error"),
            button = editor.querySelector("[data-mask-apply]");
        const controls = Array.from(
            this.elements.stage3Content.querySelectorAll(
                ".mask-editor input, .mask-editor select, .mask-editor button"
            )
        );
        controls.forEach((control) => {
            control.disabled = true;
        });
        error.hidden = true;
        button.disabled = true;
        editor.dataset.pending = "true";
        try {
            const response = await fetch("/api/stage3/mask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: this.state.sessionId,
                    ...field,
                    index,
                    options: this.readMaskOptions(editor),
                    apply_matching_accounts:
                        !!editor.querySelector('[data-mask="shared"]')?.checked,
                }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.message || "Could not apply the pattern.");
            for (const item of result.updated) {
                const key = `method_${item.table}_${item.column}`;
                this.state.tableData[item.table][item.column].methods[item.index] = item.candidate;
                const old = document
                    .getElementById(`code_${key}_${item.index}`)
                    ?.closest(".method-option");
                if (old) old.replaceWith(this.createMethodOption(key, item.candidate, item.index));
                this.selectMethod(key, item.index);
            }
            document.getElementById(`radio_${uniqueKey}_${index}`)?.focus();
            this.log(
                `Applied one MASK pattern to ${result.updated.length} proposal(s). Final approval is still required.`,
                "info"
            );
        } catch (failure) {
            error.textContent = failure.message;
            error.hidden = false;
        } finally {
            controls.forEach((control) => {
                control.disabled = false;
            });
            button.disabled = false;
            editor.dataset.pending = "false";
            this.state.isApplyingMask = false;
        }
    }

    selectMethod(uniqueKey, index) {
        /**
         * Handle method selection
         */
        this.state.selectedChoices[uniqueKey] = index;

        document.querySelectorAll(`input[name="${uniqueKey}"]`).forEach((radio, i) => {
            const option = radio.closest(".method-option");
            if (i === index) {
                radio.checked = true;
                option.classList.add("selected");
            } else {
                radio.checked = false;
                option.classList.remove("selected");
            }
        });
    }

    async submitPolicies() {
        /**
         * Submit selected policies and generate code
         */

        if (this.state.isApplyingMask) {
            this.log("Wait for the masking pattern to finish applying before approval.", "warning");
            return;
        }
        for (const [key, index] of Object.entries(this.state.selectedChoices)) {
            const editor = document
                .getElementById(`code_${key}_${index}`)
                ?.closest(".method-option")
                ?.querySelector(".mask-editor");
            if (editor && (editor.dataset.dirty === "true" || editor.dataset.pending === "true")) {
                const error = editor.querySelector(".mask-error");
                error.textContent = "Apply this masking pattern before approving the review.";
                error.hidden = false;
                const card = editor.closest(".column-card");
                if (card) {
                    card.hidden = false;
                    card.closest(".table-section").hidden = false;
                }
                editor.scrollIntoView?.({ block: "center", behavior: "smooth" });
                editor.querySelector("[data-mask-apply]").focus();
                return;
            }
        }
        this.elements.stage3Modal.classList.remove("active");
        this.log("[Update] Submitting selected policies...", "info");
        this.log("[Processing] Generating pseudonymization code...", "info");
        this.log(
            `[Stats] Submitting ${Object.keys(this.state.selectedChoices).length} selections`,
            "info"
        );

        const policiesWithCode = {
            // Include session ID in submission.
            // Associate the submission with this session.
            session_id: this.state.sessionId,
        };
        for (const [key, index] of Object.entries(this.state.selectedChoices)) {
            const codeInputId = `code_${key}_${index}`;
            const codeInput = document.getElementById(codeInputId);
            const originalCode = codeInput?.dataset.original || "";
            const editedCode = codeInput?.value || "";

            const descInputId = `desc_${key}_${index}`;
            const descInput = document.getElementById(descInputId);
            const originalDesc = descInput?.dataset.original || "";
            const editedDesc = descInput?.value || "";

            policiesWithCode[key] = {
                index: index,
                code_snippet: editedCode,
                code_modified: editedCode !== originalCode,
                description: editedDesc,
                description_modified: editedDesc !== originalDesc,
            };
            if (document.getElementById(`contract_confirm_${key}_${index}`)?.checked) {
                try {
                    const parameters = JSON.parse(
                        document.getElementById(`contract_parameters_${key}_${index}`).value
                    );
                    if (
                        parameters === null ||
                        Array.isArray(parameters) ||
                        typeof parameters !== "object"
                    ) {
                        throw new Error("Parameters must be a JSON object.");
                    }
                    policiesWithCode[key].declared_contract = {
                        method: document.getElementById(`contract_method_${key}_${index}`).value,
                        parameters: parameters,
                    };
                } catch (error) {
                    this.log(`[ERROR] ${key}: ${error.message}`, "error");
                    this.elements.stage3Modal.classList.add("active");
                    return;
                }
            }
        }

        // Include session ID for multi-client support.
        // Keep the submission scoped to this client session.
        if (this.state.sessionId) {
            policiesWithCode.session_id = this.state.sessionId;
        }

        try {
            const response = await fetch("/submit_stage3", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(policiesWithCode),
            });

            const data = await response.json();

            if (data.status === "completed") {
                this.log("\n✨ Pipeline Completed Successfully!", "success");
                this.updateStatus("COMPLETED");
                this.state.eventSource.close();

                // Update reports with regenerated integrated reports from Stage 3.

                if (data.reports) {
                    this.state.reports = data.reports;
                    this.log("[Stats] Updated report files with Stage 3 modifications", "info");
                    // Reload reports to show updated content.

                    await this.loadReports();
                }

                this.displayGeneratedCode(data.codes);
                this.switchMainTab("code");

                // Hide reopen policy button after successful code generation.
                // Hide the reopen control after code generation succeeds.
                if (this.elements.reopenPolicyBtn) {
                    this.elements.reopenPolicyBtn.classList.add("hidden");
                }

                // Show start button and hide stop button.
                // Restore the idle pipeline controls.
                this.elements.startBtn.classList.remove("hidden");
                this.elements.stopBtn.classList.add("hidden");

                // Show heuristic learning notification if patterns were learned.
                // Show the rule-learning completion notice.
                if (data.heuristic_learning && data.heuristic_learning.total_patterns > 0) {
                    const stats = data.heuristic_learning;
                    const piiCount =
                        (stats.pii_patterns.suffixes || 0) +
                        (stats.pii_patterns.prefixes || 0) +
                        (stats.pii_patterns.keywords || 0) +
                        (stats.pii_patterns.exact || 0);
                    const nonPiiCount =
                        (stats.non_pii_patterns.suffixes || 0) +
                        (stats.non_pii_patterns.prefixes || 0) +
                        (stats.non_pii_patterns.keywords || 0) +
                        (stats.non_pii_patterns.exact || 0);
                    const message =
                        `휴리스틱이 자동으로 업데이트되었습니다.\n\n` +
                        `총 ${stats.total_patterns}개의 패턴이 학습되었습니다.\n` +
                        `• PII 패턴: ${piiCount}개\n` +
                        `• Non-PII 패턴: ${nonPiiCount}개\n\n` +
                        `Heuristics 탭에서 확인할 수 있습니다.`;
                    alert(message);

                    // Reload heuristics to show updated data.

                    this.loadHeuristics();
                }
            } else {
                if (data.status === "validation_error") {
                    this.elements.stage3Modal.classList.add("active");
                    window.alert(
                        data.error || data.message || "Please review the selected treatment."
                    );
                }
                throw new Error(data.error || data.message || "Code generation failed");
            }
        } catch (error) {
            this.log(`[ERROR] Error: ${error.message}`, "error");
            this.updateStatus("ERROR");
        } finally {
            this.elements.startBtn.disabled = false;
            this.state.isRunning = false;
        }
    }

    displayGeneratedCode(codes) {
        /**
         * Display generated pseudonymization code
         */
        this.elements.codeTabsContainer.innerHTML = "";
        this.elements.codeViewers.innerHTML = "";

        // Check if codes object is empty or null.
        // Handle missing generated code.
        if (!codes || Object.keys(codes).length === 0) {
            this.elements.emptyState.style.display = "block";
            this.elements.codeContent.style.display = "none";
            this.log("[WARN] No generated code available", "warning");
            return;
        }

        this.elements.emptyState.style.display = "none";
        this.elements.codeContent.style.display = "";

        let isFirst = true;
        for (const [tableName, code] of Object.entries(codes)) {
            const tabBtn = document.createElement("button");
            tabBtn.className = `tab-btn ${isFirst ? "active" : ""}`;
            tabBtn.textContent = `pseudonymize_${tableName}.py`;
            tabBtn.onclick = () => this.switchCodeTab(tableName);
            this.elements.codeTabsContainer.appendChild(tabBtn);

            const codeContainer = document.createElement("div");
            codeContainer.id = `code-${tableName}`;
            codeContainer.className = `tab-content ${isFirst ? "active" : ""}`;

            codeContainer.innerHTML = `
                <div class="code-viewer">
                    <div class="code-header">
                        <span class="code-title">📄 pseudonymize_${tableName}.py</span>
                        <button class="btn-copy" onclick="app.copyCode('${tableName}')">
                            📋 Copy Code
                        </button>
                    </div>
                    <div class="code-content">
                        <pre><code>${this.escapeHtml(code)}</code></pre>
                    </div>
                </div>
            `;

            this.elements.codeViewers.appendChild(codeContainer);
            isFirst = false;
        }

        this.elements.codeTabsContainer.style.display = "flex";
    }

    async loadReports() {
        /**
         * Load and display generated reports for multiple tables
         */

        this.elements.emptyStateReports.style.display = "none";
        this.elements.reportsContent.style.display = "block";
        this.elements.reportTabsContainer.innerHTML = "";
        this.elements.reportViewers.innerHTML = "";

        // Find all integrated reports (one per table)
        // Collect one integrated report per table.
        const integratedReportKeys = Object.keys(this.state.reports).filter((key) =>
            key.includes("_integrated")
        );

        if (integratedReportKeys.length > 0) {
            let isFirst = true;
            for (const reportKey of integratedReportKeys) {
                // Extract table name from report key (e.g., "CUSTOMERS_integrated" -> "CUSTOMERS")
                // Read the table name from the report key.
                const tableName = reportKey.replace("_integrated", "");
                const filename = this.state.reports[reportKey];

                // Create tab button.
                // Create the report tab.
                const tabBtn = document.createElement("button");
                tabBtn.className = `tab-btn ${isFirst ? "active" : ""}`;
                tabBtn.textContent = `${tableName}`;
                tabBtn.onclick = () => this.switchReportTab(reportKey);
                this.elements.reportTabsContainer.appendChild(tabBtn);

                // Fetch report content.
                // Fetch the report content.
                const content = await this.fetchReport(filename);

                // Create report container.
                // Create the report container.
                const reportContainer = document.createElement("div");
                reportContainer.id = `report-${reportKey}`;
                reportContainer.className = `tab-content ${isFirst ? "active" : ""}`;
                reportContainer.innerHTML = `
                    <div class="report-viewer">
                        <div class="report-header">
                            <h2 class="report-title">📊 ${tableName} - Integrated Report</h2>
                            <div class="report-actions">
                                <button class="btn-report" onclick="app.downloadReport('${reportKey}')">
                                    💾 Download
                                </button>
                            </div>
                        </div>
                        <div class="report-content">
                            ${content}
                        </div>
                    </div>
                `;

                this.elements.reportViewers.appendChild(reportContainer);
                isFirst = false;
            }

            this.elements.reportTabsContainer.style.display = "flex";

            // Auto-switch to reports tab.
            // Activate the report tab.
            this.switchMainTab("reports");
            const tableCount = integratedReportKeys.length;
            this.log(
                `[Stats] ${tableCount} integrated report(s) generated. Switching to Reports tab.`,
                "success"
            );
        } else {
            this.elements.emptyStateReports.style.display = "flex";
            this.elements.reportsContent.style.display = "none";
        }
    }

    async fetchReport(filename) {
        /**
         * Fetch report content from server
         */
        try {
            const response = await fetch(`/get_report/${filename}`);
            const markdown = await response.text();
            return this.renderMarkdown(markdown);
        } catch (error) {
            return `<p class="text-danger">Failed to load report: ${error.message}</p>`;
        }
    }

    renderMarkdown(markdown) {
        /**
         * Render the report locally without a CDN dependency.
         *
         * Input is escaped before applying the supported Markdown subset so
         * model-generated report content cannot inject executable HTML.
         */
        const escapeHtml = (text) =>
            String(text)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");

        const renderInline = (text) =>
            escapeHtml(text)
                .replace(/`([^`]+)`/g, "<code>$1</code>")
                .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
                .replace(/\*([^*]+)\*/g, "<em>$1</em>");

        const splitTableRow = (line) =>
            line
                .trim()
                .replace(/^\|/, "")
                .replace(/\|$/, "")
                .split("|")
                .map((cell) => cell.trim());

        const isTableSeparator = (line) => {
            const cells = splitTableRow(line);
            return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
        };

        const lines = String(markdown || "")
            .replace(/\r\n/g, "\n")
            .split("\n");
        const output = [];
        let inCodeBlock = false;
        let codeLines = [];
        let listType = null;

        const closeList = () => {
            if (listType) {
                output.push(`</${listType}>`);
                listType = null;
            }
        };

        for (let index = 0; index < lines.length; index += 1) {
            const line = lines[index];

            if (/^```/.test(line.trim())) {
                closeList();
                if (inCodeBlock) {
                    output.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
                    codeLines = [];
                    inCodeBlock = false;
                } else {
                    inCodeBlock = true;
                }
                continue;
            }

            if (inCodeBlock) {
                codeLines.push(line);
                continue;
            }

            if (
                line.includes("|") &&
                index + 1 < lines.length &&
                isTableSeparator(lines[index + 1])
            ) {
                closeList();
                const headers = splitTableRow(line);
                output.push('<div class="markdown-table-wrap"><table><thead><tr>');
                headers.forEach((header) => output.push(`<th>${renderInline(header)}</th>`));
                output.push("</tr></thead><tbody>");
                index += 2;

                while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
                    const cells = splitTableRow(lines[index]);
                    output.push("<tr>");
                    cells.forEach((cell) => output.push(`<td>${renderInline(cell)}</td>`));
                    output.push("</tr>");
                    index += 1;
                }

                output.push("</tbody></table></div>");
                index -= 1;
                continue;
            }

            if (/^ {0,3}>/.test(line)) {
                closeList();
                const quoteLines = [];
                while (index < lines.length && /^ {0,3}>/.test(lines[index])) {
                    quoteLines.push(lines[index].replace(/^ {0,3}> ?/, ""));
                    index += 1;
                }
                output.push(
                    `<blockquote>${this.renderMarkdown(quoteLines.join("\n"))}</blockquote>`
                );
                index -= 1;
                continue;
            }

            const headingMatch = line.match(/^ {0,3}(#{1,6})\s+(.+)$/);
            if (headingMatch) {
                closeList();
                const level = headingMatch[1].length;
                output.push(`<h${level}>${renderInline(headingMatch[2])}</h${level}>`);
                continue;
            }

            if (/^\s*[-*]\s+/.test(line)) {
                if (listType !== "ul") {
                    closeList();
                    listType = "ul";
                    output.push("<ul>");
                }
                output.push(`<li>${renderInline(line.replace(/^\s*[-*]\s+/, ""))}</li>`);
                continue;
            }

            if (/^\s*\d+\.\s+/.test(line)) {
                if (listType !== "ol") {
                    closeList();
                    listType = "ol";
                    output.push("<ol>");
                }
                output.push(`<li>${renderInline(line.replace(/^\s*\d+\.\s+/, ""))}</li>`);
                continue;
            }

            closeList();
            if (!line.trim()) {
                continue;
            }
            if (/^\s*---+\s*$/.test(line)) {
                output.push("<hr>");
            } else {
                output.push(`<p>${renderInline(line)}</p>`);
            }
        }

        closeList();
        if (inCodeBlock) {
            output.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        }

        return output.join("");
    }

    switchMainTab(tabName) {
        /**
         * Switch between main tabs (Logs, Code, Reports)
         */
        document.querySelectorAll(".tab-nav-btn").forEach((btn) => {
            btn.classList.toggle("active", btn.dataset.tab === tabName);
        });

        document.querySelectorAll(".main-tab-content").forEach((content) => {
            content.classList.toggle("active", content.id === `${tabName}Tab`);
        });
    }

    switchCodeTab(tableName, event) {
        /**
         * Switch between code tabs for different tables
         */
        document.querySelectorAll("#codeTabsContainer .tab-btn").forEach((btn) => {
            btn.classList.remove("active");
        });
        document.querySelectorAll("#codeViewers .tab-content").forEach((content) => {
            content.classList.remove("active");
        });

        if (event && event.target) {
            event.target.classList.add("active");
        }
        document.getElementById(`code-${tableName}`).classList.add("active");
    }

    switchReportTab(reportKey) {
        /**
         * Switch between report tabs
         */
        // Remove active class from all tabs and contents.

        document.querySelectorAll("#reportTabsContainer .tab-btn").forEach((btn) => {
            btn.classList.remove("active");
        });
        document.querySelectorAll("#reportViewers .tab-content").forEach((content) => {
            content.classList.remove("active");
        });

        // Find and activate the correct tab button.
        // Activate the matching tab button.
        const tableName = reportKey.replace("_integrated", "");
        document.querySelectorAll("#reportTabsContainer .tab-btn").forEach((btn) => {
            if (btn.textContent === tableName) {
                btn.classList.add("active");
            }
        });

        // Activate the report content.
        // Show the selected report.
        document.getElementById(`report-${reportKey}`).classList.add("active");
    }

    copyCode(tableName) {
        /**
         * Copy generated code to clipboard
         */
        const codeElement = document.querySelector(`#code-${tableName} pre code`);
        const code = codeElement.textContent;

        navigator.clipboard
            .writeText(code)
            .then(() => {
                this.showNotification("Code copied to clipboard!", "success");
            })
            .catch(() => {
                this.showNotification("Failed to copy code", "error");
            });
    }

    downloadReport(reportKey) {
        /**
         * Download report file
         */
        const filename = this.state.reports[reportKey];
        window.open(`/download_report/${filename}`, "_blank");
    }

    viewReportAndCloseModal() {
        /**
         * Close Stage 3 modal and switch to report tab
         */
        this.elements.stage3Modal.classList.remove("active");
        this.switchMainTab("reports");
    }

    updateStatus(status) {
        /**
         * Update pipeline status badge
         */
        this.state.currentStep = status;
        this.elements.statusBadge.textContent = status;
        this.elements.statusBadge.className = "status-badge";

        switch (status) {
            case "RUNNING":
                this.elements.statusBadge.classList.add("running");
                break;
            case "COMPLETED":
                this.elements.statusBadge.classList.add("completed");
                break;
            case "ERROR":
                this.elements.statusBadge.classList.add("error");
                break;
        }
    }

    log(message, type = "info") {
        /**
         * Log message to the logs tab
         * @param {string} message - Message to log
         * @param {string} type - Log type (info/error/warning) - reserved for future use
         */
        // Check if message already has timestamp from backend (format: [HH:MM:SS])
        // Recognize server timestamps in [HH:MM:SS] format.
        const hasTimestamp = /^\[\d{2}:\d{2}:\d{2}\]/.test(message);

        let formattedMessage;
        if (hasTimestamp) {
            // Use message as-is if it already has timestamp.
            // Preserve the server timestamp.
            formattedMessage = message;
        } else {
            // Add timestamp if not present.
            // Timestamp messages that have no server timestamp.
            const timestamp = new Date().toLocaleTimeString();
            formattedMessage = `[${timestamp}] ${message}`;
        }

        const logsTab = document.querySelector("#logsTab .empty-state");
        if (logsTab) {
            logsTab.style.display = "none";
        }

        this.elements.logsContent.style.display = "block";
        this.elements.logsContent.innerHTML += formattedMessage + "\n";
        this.elements.logsContent.scrollTop = this.elements.logsContent.scrollHeight;
    }

    clearTerminal() {
        /**
         * Clear all logs
         */
        this.elements.logsContent.innerHTML = "";
        const logsTab = document.querySelector("#logsTab .empty-state");
        if (logsTab) {
            logsTab.style.display = "flex";
        }
        this.elements.logsContent.style.display = "none";
    }

    showNotification(message, type = "info") {
        /**
         * Show notification to user
         * @param {string} message - Notification message
         * @param {string} type - Notification type (info/success/error) - reserved for future use
         */
        alert(message);
    }

    async changePIIStatus(tableName, columnName, currentStatus) {
        /**
         * Handle PII status change for a column
         *
         * When changing from Non-PII to PII, this function:
         * - Prompts user for PII type and rationale
         * - Calls backend to re-run Stage 2 with LLM
         * - Updates UI with new pseudonymization methods
         */

        // Prevent duplicate modals.
        const existingModal = document.getElementById("piiTypeModal");
        if (existingModal) {
            existingModal.remove();
        }

        // Determine new status.
        const newStatus = currentStatus === "PII" ? "Non-PII" : "PII";

        // If changing to PII, prompt for PII type and rationale.
        let piiType = "PII";
        let rationale = "";

        if (newStatus === "PII") {
            // Create modal for PII type selection.
            const modalHtml = `
                <div class="modal-overlay active" id="piiTypeModal" style="display: flex;">
                    <div class="modal-content" style="max-width: 500px;">
                        <div class="modal-header">
                            <h3>🔄 Change to PII</h3>
                            <button class="btn-close" onclick="document.getElementById('piiTypeModal').remove()">×</button>
                        </div>
                        <div class="modal-body">
                            <p>You are changing <strong>${columnName}</strong> from Non-PII to PII.</p>
                            <p>Please select the PII type and provide a rationale:</p>

                            <div class="form-group">
                                <label for="piiTypeSelect">PII Type:</label>
                                <select id="piiTypeSelect" class="form-control">
                                    <option value="PII">PII (개인정보)</option>
                                </select>
                            </div>

                            <div class="form-group">
                                <label for="piiRationale">Rationale (사유):</label>
                                <textarea id="piiRationale" class="form-control" rows="3"
                                    placeholder="Why is this column PII? (e.g., Contains personal identifiers, user requested, etc.)"></textarea>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-secondary" id="cancelPIIChange">Cancel</button>
                            <button class="btn btn-primary" id="confirmPIIChange">Confirm & Generate Methods</button>
                        </div>
                    </div>
                </div>
            `;

            // Add modal to DOM.
            document.body.insertAdjacentHTML("beforeend", modalHtml);

            // Wait for user confirmation.
            const confirmed = await new Promise((resolve) => {
                // Handle cancel button.
                const cancelBtn = document.getElementById("cancelPIIChange");
                const confirmBtn = document.getElementById("confirmPIIChange");

                if (!cancelBtn || !confirmBtn) {
                    console.error("[PII Status Change] Modal buttons not found!");
                    resolve(false);
                    return;
                }

                cancelBtn.onclick = () => {
                    document.getElementById("piiTypeModal").remove();
                    resolve(false); // User cancelled.
                };

                confirmBtn.onclick = () => {
                    piiType = document.getElementById("piiTypeSelect").value;
                    rationale = document.getElementById("piiRationale").value.trim();
                    document.getElementById("piiTypeModal").remove();
                    resolve(true); // User confirmed.
                };
            });

            // If user cancelled, exit.
            if (!confirmed) {
                return;
            }
        } else {
            // Changing to Non-PII - no confirmation needed, just set default rationale.
            rationale = "User manually changed to Non-PII";
        }

        // Show loading indicator on the card.
        const card = document.querySelector(
            `.column-card[data-table="${tableName}"][data-column="${columnName}"]`
        );
        if (card) {
            const methodsContainer = card.querySelector(".methods-container");
            if (methodsContainer) {
                const loadingHtml = `
                    <div class="loading-indicator" style="text-align: center; padding: 20px;">
                        <div class="spinner" style="display: inline-block; width: 40px; height: 40px; border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                        <p style="margin-top: 10px; color: #666;">Generating methods with LLM...</p>
                        <style>
                            @keyframes spin {
                                0% { transform: rotate(0deg); }
                                100% { transform: rotate(360deg); }
                            }
                        </style>
                    </div>
                `;
                methodsContainer.innerHTML = loadingHtml;
            }
        }

        this.log(
            `[Update] Changing ${columnName} status from ${currentStatus} to ${newStatus}...`,
            "info"
        );

        try {
            // Call backend API.
            const response = await fetch("/change_pii_status", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    table: tableName,
                    column: columnName,
                    old_status: currentStatus, // Explicitly send old_status.
                    new_status: newStatus,
                    pii_type: piiType,
                    rationale: rationale,
                    session_id: this.state.sessionId, // Include session ID for multi-client support.
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || "Failed to change PII status");
            }

            const result = await response.json();

            if (result.status === "success") {
                this.log(`[OK] ${result.message}`, "success");

                // Update the UI with new methods.
                const updateData = {
                    is_pii: newStatus === "PII",
                    pii_type: result.pii_type || "Non-PII",
                    methods: result.methods || [],
                    column_comment: result.column_comment || "",
                };

                this.updateColumnCard(tableName, columnName, updateData);

                // Update state.tableData.
                if (this.state.tableData && this.state.tableData[tableName]) {
                    this.state.tableData[tableName][columnName].is_pii = newStatus === "PII";
                    this.state.tableData[tableName][columnName].pii_type =
                        result.pii_type || "Non-PII";
                    this.state.tableData[tableName][columnName].methods = result.methods || [];
                }

                // Success - no alert needed.
            } else {
                throw new Error("Unexpected response from server");
            }
        } catch (error) {
            console.error("Error changing PII status:", error);
            this.log(`[ERROR] Error: ${error.message}`, "error");

            // Restore original UI on error.
            if (card) {
                const methodsContainer = card.querySelector(".methods-container");
                if (methodsContainer) {
                    methodsContainer.innerHTML = `
                        <div class="error-message" style="text-align: center; padding: 20px; color: #d32f2f;">
                            <p>❌ Failed to change PII status</p>
                            <p style="font-size: 14px;">${error.message}</p>
                            <button class="btn btn-secondary" onclick="location.reload()">Reload Page</button>
                        </div>
                    `;
                }
            }
        }
    }

    updateColumnCard(tableName, columnName, newData) {
        /**
         * Update a column card with new data
         */
        const card = document.querySelector(
            `.column-card[data-table="${tableName}"][data-column="${columnName}"]`
        );
        if (!card) {
            console.error(`Column card not found for ${tableName}.${columnName}`);
            return;
        }

        // Update PII badge.
        const badge = card.querySelector(".pii-badge");
        const statusText = card.querySelector(".pii-status-text");

        if (badge && statusText) {
            // Clear previous state.
            badge.classList.remove("non-pii-badge");

            if (newData.is_pii) {
                // PII: Red background.
                badge.dataset.status = "PII";
                statusText.textContent = newData.pii_type || "PII";
            } else {
                // Non-PII: Green background.
                badge.classList.add("non-pii-badge");
                badge.dataset.status = "Non-PII";
                statusText.textContent = "Non-PII";
            }
        } else {
            console.error(`[updateColumnCard] Badge or statusText not found in card`);
        }

        // Update Change Status button.
        const changeBtn = card.querySelector(".btn-change-pii-status");
        if (changeBtn) {
            const currentStatusForButton = newData.is_pii ? "PII" : "Non-PII";

            // Update data attribute instead of onclick.
            changeBtn.dataset.currentStatus = currentStatusForButton;

            // Remove old onclick attribute if it exists.
            changeBtn.removeAttribute("onclick");

            // Clone button to remove all event listeners.
            const newBtn = changeBtn.cloneNode(true);
            changeBtn.parentNode.replaceChild(newBtn, changeBtn);

            // Add new event listener.
            newBtn.addEventListener("click", (event) => {
                event.stopPropagation(); // Prevent event bubbling.
                event.preventDefault(); // Prevent default button behavior.
                const table = newBtn.dataset.table;
                const column = newBtn.dataset.column;
                const status = newBtn.dataset.currentStatus;
                this.changePIIStatus(table, column, status);
            });
        } else {
            console.error(`[updateColumnCard] Change Status button not found in card`);
        }

        // Update methods container.
        const methodsContainer = card.querySelector(".methods-container");
        if (methodsContainer) {
            // Clear existing content (including loading indicator)
            methodsContainer.innerHTML = "";

            // Add title.
            const newTitle = document.createElement("div");
            newTitle.className = "methods-title";
            newTitle.innerHTML = "<span>🛠️</span> <span>Select Pseudonymization Method:</span>";
            methodsContainer.appendChild(newTitle);

            // Check if methods exist.
            if (newData.methods && newData.methods.length > 0) {
                // Add new methods.
                const uniqueKey = `method_${tableName}_${columnName}`;
                this.state.selectedChoices[uniqueKey] = 0; // Reset to first method.

                newData.methods.forEach((method, index) => {
                    const option = this.createMethodOption(uniqueKey, method, index);
                    methodsContainer.appendChild(option);
                });
            } else {
                // No methods available.
                const noMethodsMsg = document.createElement("div");
                noMethodsMsg.style.padding = "20px";
                noMethodsMsg.style.textAlign = "center";
                noMethodsMsg.style.color = "#666";
                noMethodsMsg.textContent = "No methods available";
                methodsContainer.appendChild(noMethodsMsg);
            }
        }
    }

    async validatePolicies() {
        /**
         * Validate current policies using backend validator
         *
         * Implements real-time validation
         */
        const resultsContainer = document.getElementById("validationResults");
        if (!resultsContainer) {
            console.error("Validation results container not found");
            return;
        }

        resultsContainer.innerHTML =
            '<div class="validation-loading">🔍 Validating policies...</div>';

        try {
            const policies = this.collectUserPolicies();

            const response = await fetch("/validate_policy", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    table:
                        this.state.currentTable ||
                        Object.keys(this.state.selectedChoices)[0]?.split("_")[1],
                    policies: policies,
                    session_id: this.state.sessionId, // Include session ID for multi-client support.
                }),
            });

            if (!response.ok) {
                throw new Error(`Validation failed: ${response.statusText}`);
            }

            const result = await response.json();
            this.displayValidationResults(result);
        } catch (error) {
            console.error("Validation error:", error);
            resultsContainer.innerHTML = `
                <div class="alert alert-danger">
                    ❌ Validation failed: ${error.message}
                </div>
            `;
        }
    }

    collectUserPolicies() {
        /**
         * Collect user-selected policies from UI
         */
        const policies = {};

        for (const [key, index] of Object.entries(this.state.selectedChoices)) {
            const parts = key.split("_");
            if (parts.length >= 3) {
                const columnName = parts.slice(2).join("_");

                const radioGroup = document.querySelectorAll(`input[name="${key}"]`);
                const selectedRadio = Array.from(radioGroup).find((r) => r.checked);

                if (selectedRadio) {
                    const methodOption = selectedRadio.closest(".method-option");
                    const methodElement = methodOption.querySelector(".method-name");
                    const descElement = methodOption.querySelector(".method-description");

                    policies[columnName] = {
                        action: methodElement ? methodElement.textContent.trim() : "KEEP",
                        rationale: descElement ? descElement.textContent.trim() : "",
                    };
                }
            }
        }

        return policies;
    }

    displayValidationResults(result) {
        /**
         * Display validation results in UI
         */
        const container = document.getElementById("validationResults");
        if (!container) return;

        container.innerHTML = "";

        if (!result.violations || result.violations.length === 0) {
            container.innerHTML = `
                <div class="alert alert-success">
                    ✅ <strong>No violations found!</strong> All policies are compliant.
                </div>
            `;
            return;
        }

        const summary = result.summary || {};
        const summaryDiv = document.createElement("div");
        summaryDiv.className = "validation-summary";
        summaryDiv.innerHTML = `
            <h4>Validation Summary</h4>
            <div class="summary-stats">
                <span class="stat stat-total">Total: ${summary.total || 0}</span>
                <span class="stat stat-high">High: ${summary.by_severity?.HIGH || 0}</span>
                <span class="stat stat-medium">Medium: ${summary.by_severity?.MEDIUM || 0}</span>
                <span class="stat stat-low">Low: ${summary.by_severity?.LOW || 0}</span>
            </div>
            <div class="compliance-status ${summary.is_compliant ? "compliant" : "non-compliant"}">
                ${summary.is_compliant ? "✅ Compliant" : "⚠️ Non-Compliant"}
            </div>
        `;
        container.appendChild(summaryDiv);

        const violationsDiv = document.createElement("div");
        violationsDiv.className = "violations-list";

        result.violations.forEach((v) => {
            const severity = v.severity.toLowerCase();
            const severityIcon = severity === "high" ? "🚨" : severity === "medium" ? "⚠️" : "ℹ️";

            const violationCard = document.createElement("div");
            violationCard.className = `violation-card violation-${severity}`;
            violationCard.innerHTML = `
                <div class="violation-header">
                    <span class="violation-icon">${severityIcon}</span>
                    <span class="violation-severity">${v.severity}</span>
                    <span class="violation-check-type">${v.check_type.replace(/_/g, " ")}</span>
                </div>
                <div class="violation-body">
                    <div class="violation-column">
                        <strong>Column:</strong> ${v.column}
                    </div>
                    <div class="violation-message">
                        <strong>Issue:</strong> ${v.message}
                    </div>
                    <div class="violation-suggestion">
                        <strong>Suggestion:</strong> ${v.suggestion}
                    </div>
                </div>
            `;
            violationsDiv.appendChild(violationCard);
        });

        container.appendChild(violationsDiv);
    }

    escapeHtml(text) {
        /**
         * Escape HTML special characters
         */
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    formatLegalEvidence(evidence) {
        /**
         * Format legal evidence to extract and display document names
         */
        if (!evidence) return "Unknown";

        const parts = evidence.split(",").map((part) => part.trim());
        const formatted = parts
            .map((part) => {
                const match = part.match(/^([^:]+):\s*(.+)$/);
                if (match) {
                    const docName = match[1];
                    const content = match[2];
                    return `<div class="legal-evidence-item">
                    <span class="legal-doc-name">📘 ${this.escapeHtml(docName)}</span>
                    <span class="legal-doc-content">${this.escapeHtml(content)}</span>
                </div>`;
                }
                return `<div class="legal-evidence-item">${this.escapeHtml(part)}</div>`;
            })
            .join("");

        return formatted || this.escapeHtml(evidence);
    }

    // ============================================================================
    // Heuristics Management Methods.
    // Heuristic management.
    // ============================================================================

    async loadHeuristics() {
        /**
         * Load and display heuristics
         */
        try {
            // Load heuristics and statistics in parallel.
            const [heuristicsResponse, statsResponse] = await Promise.all([
                fetch("/api/heuristics"),
                fetch("/api/heuristics/statistics"),
            ]);

            const data = await heuristicsResponse.json();
            const statsData = await statsResponse.json();

            if (data.status === "success") {
                this.displayHeuristics(data.heuristics);
            } else {
                console.error("Failed to load heuristics:", data.message);
                document.getElementById("heuristicsListContainer").innerHTML = `
                    <div class="empty-state">
                        <div class="empty-icon">❌</div>
                        <h3 class="empty-title">Failed to Load Heuristics</h3>
                        <p class="empty-description">${data.message}</p>
                    </div>
                `;
            }

            // Update statistics display.
            if (statsData.status === "success") {
                this.updateHeuristicsStats(statsData.statistics);
            }
        } catch (error) {
            console.error("Error loading heuristics:", error);
            document.getElementById("heuristicsListContainer").innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">❌</div>
                    <h3 class="empty-title">Error Loading Heuristics</h3>
                    <p class="empty-description">${error.message}</p>
                </div>
            `;
        }
    }

    updateHeuristicsStats(stats) {
        /**
         * Update heuristics statistics display
         */
        document.getElementById("heurTotalPatterns").textContent = stats.total_patterns || 0;
        document.getElementById("heurManualPatterns").textContent = stats.manual_patterns || 0;
        document.getElementById("heurAutoPatterns").textContent = stats.auto_learned_patterns || 0;

        if (stats.last_updated) {
            const date = new Date(stats.last_updated);
            document.getElementById("heurLastUpdated").textContent = date.toLocaleString();
        } else {
            document.getElementById("heurLastUpdated").textContent = "-";
        }
    }

    displayHeuristics(heuristics) {
        /**
         * Display heuristics in the UI
         */
        const container = document.getElementById("heuristicsListContainer");

        if (!heuristics || heuristics.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">🔍</div>
                    <h3 class="empty-title">No Deterministic Rules Defined</h3>
                    <p class="empty-description">Click "Add Regex Rule" to define a rule, or run the HITL workflow to learn rules from approved expert corrections.</p>
                </div>
            `;
            return;
        }

        let html =
            '<div style="padding: 20px; overflow-x: auto;"><table style="width: 100%; border-collapse: collapse; min-width: 900px;">';
        html += '<thead><tr style="border-bottom: 2px solid #e0e0e0;">';
        html += '<th style="padding: 10px; text-align: left;">Source</th>';
        html += '<th style="padding: 10px; text-align: left;">Name</th>';
        html += '<th style="padding: 10px; text-align: left;">Pattern</th>';
        html += '<th style="padding: 10px; text-align: left;">Type</th>';
        html += '<th style="padding: 10px; text-align: left;">PII Type</th>';
        html += '<th style="padding: 10px; text-align: center;">Priority</th>';
        html += '<th style="padding: 10px; text-align: center;">Status</th>';
        html += '<th style="padding: 10px; text-align: center;">Actions</th>';
        html += "</tr></thead><tbody>";

        heuristics.forEach((h) => {
            const statusBadge = h.enabled
                ? '<span style="background: #4CAF50; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">Enabled</span>'
                : '<span style="background: #9E9E9E; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">Disabled</span>';

            // Source badge.
            const sourceBadge =
                h.source === "auto_learned"
                    ? '<span style="background: #4CAF50; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px;" title="Automatically learned from expert feedback">🤖 Auto</span>'
                    : '<span style="background: #2196F3; color: white; padding: 4px 8px; border-radius: 4px; font-size: 11px;" title="Manually created by expert">👤 Manual</span>';

            // Pattern type badge.
            const patternType = h.pattern_type || "custom";
            const patternTypeBadge = `<span style="background: #f0f0f0; color: #666; padding: 3px 6px; border-radius: 3px; font-size: 11px;">${this.escapeHtml(patternType)}</span>`;

            // PII type badge with color.
            const piiTypeBadge =
                h.pii_type === "PII"
                    ? '<span style="background: #FF9800; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">PII</span>'
                    : '<span style="background: #607D8B; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px;">Non-PII</span>';

            // Sample columns tooltip (for auto-learned)
            let patternCell = `<code style="font-family: monospace; font-size: 12px; background: #f5f5f5; padding: 2px 6px; border-radius: 3px;">${this.escapeHtml(h.regex)}</code>`;
            if (h.sample_columns && h.sample_columns.length > 0) {
                const samples = h.sample_columns.slice(0, 5).join(", ");
                patternCell += `<br><small style="color: #888; font-size: 11px;" title="${this.escapeHtml(h.sample_columns.join(", "))}">Samples: ${this.escapeHtml(samples)}</small>`;
            }

            html += '<tr style="border-bottom: 1px solid #e0e0e0;">';
            html += `<td style="padding: 10px;">${sourceBadge}</td>`;
            html += `<td style="padding: 10px;"><strong>${this.escapeHtml(h.name)}</strong></td>`;
            html += `<td style="padding: 10px;">${patternCell}</td>`;
            html += `<td style="padding: 10px;">${patternTypeBadge}</td>`;
            html += `<td style="padding: 10px;">${piiTypeBadge}</td>`;
            html += `<td style="padding: 10px; text-align: center;">${h.priority}</td>`;
            html += `<td style="padding: 10px; text-align: center;">${statusBadge}</td>`;
            html += `<td style="padding: 10px; text-align: center;">
                <div style="display: flex; gap: 5px; justify-content: center; align-items: center;">
                    <button class="btn btn-secondary" onclick="app.editHeuristic('${h.id}')" style="padding: 5px 12px; width: 70px;">Edit</button>
                    <button class="btn btn-secondary" onclick="app.deleteHeuristic('${h.id}')" style="padding: 5px 12px; width: 70px; background: #ef4444; color: #ffffff;">Delete</button>
                </div>
            </td>`;
            html += "</tr>";
        });

        html += "</tbody></table></div>";
        container.innerHTML = html;
    }

    async editHeuristic(heuristicId) {
        /**
         * Open edit modal for heuristic
         */
        try {
            const response = await fetch(`/api/heuristics/${heuristicId}`);
            const data = await response.json();

            if (data.status === "success") {
                const h = data.heuristic;

                // Populate form.
                document.getElementById("heuristicId").value = h.id;
                document.getElementById("heuristicName").value = h.name;
                document.getElementById("heuristicRegex").value = h.regex;
                document.getElementById("heuristicPiiType").value = h.pii_type;
                document.getElementById("heuristicRationale").value = h.rationale;
                document.getElementById("heuristicPriority").value = h.priority;
                document.getElementById("heuristicEnabled").checked = h.enabled;
                document.getElementById("heuristicSource").value = h.source || "manual";
                document.getElementById("heuristicPatternType").value = h.pattern_type || "";

                // Show source badge for auto-learned patterns.
                const sourceDisplay = document.getElementById("heuristicSourceDisplay");
                const sourceBadge = document.getElementById("heuristicSourceBadge");
                if (h.source === "auto_learned") {
                    sourceBadge.innerHTML =
                        '<span style="background: #4CAF50; color: white; padding: 6px 12px; border-radius: 4px; font-size: 13px;">🤖 Auto-Learned from Expert Feedback</span>';
                    sourceDisplay.style.display = "block";
                } else {
                    sourceBadge.innerHTML =
                        '<span style="background: #2196F3; color: white; padding: 6px 12px; border-radius: 4px; font-size: 13px;">👤 Manually Created</span>';
                    sourceDisplay.style.display = "block";
                }

                // Show sample columns for auto-learned patterns.
                const sampleColumnsGroup = document.getElementById("heuristicSampleColumnsGroup");
                const sampleColumnsDiv = document.getElementById("heuristicSampleColumns");
                if (h.sample_columns && h.sample_columns.length > 0) {
                    sampleColumnsDiv.textContent = h.sample_columns.join(", ");
                    sampleColumnsGroup.style.display = "block";
                } else {
                    sampleColumnsGroup.style.display = "none";
                }

                // Update modal title.
                document.getElementById("heuristicModalTitle").textContent =
                    "Edit Deterministic Regex Rule";

                // Show modal.
                document.getElementById("heuristicModal").classList.add("active");
            }
        } catch (error) {
            alert("Failed to load heuristic: " + error.message);
        }
    }

    async deleteHeuristic(heuristicId) {
        /**
         * Delete heuristic after confirmation
         */
        if (!confirm("Are you sure you want to delete this heuristic?")) {
            return;
        }

        try {
            const response = await fetch(`/api/heuristics/${heuristicId}`, {
                method: "DELETE",
            });
            const data = await response.json();

            if (data.status === "success") {
                alert("Heuristic deleted successfully");
                this.loadHeuristics();
            } else {
                alert("Failed to delete heuristic: " + data.message);
            }
        } catch (error) {
            alert("Error deleting heuristic: " + error.message);
        }
    }

    async saveHeuristic() {
        /**
         * Save (add or update) heuristic
         */
        const id = document.getElementById("heuristicId").value;
        const patternType = document.getElementById("heuristicPatternType").value;

        const data = {
            name: document.getElementById("heuristicName").value,
            regex: document.getElementById("heuristicRegex").value,
            pii_type: document.getElementById("heuristicPiiType").value,
            rationale: document.getElementById("heuristicRationale").value,
            priority: parseInt(document.getElementById("heuristicPriority").value),
            enabled: document.getElementById("heuristicEnabled").checked,
        };

        // Add pattern_type if selected.
        if (patternType) {
            data.pattern_type = patternType;
        }

        // Validate.
        if (!data.name || !data.regex || !data.pii_type || !data.rationale) {
            alert("Please fill in all required fields");
            return;
        }

        try {
            let response;
            if (id) {
                // Update.
                response = await fetch(`/api/heuristics/${id}`, {
                    method: "PUT",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(data),
                });
            } else {
                // Add new (source defaults to 'manual' on server)
                response = await fetch("/api/heuristics", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(data),
                });
            }

            const result = await response.json();

            if (result.status === "success") {
                alert(id ? "Heuristic updated successfully" : "Heuristic added successfully");
                document.getElementById("heuristicModal").classList.remove("active");
                this.loadHeuristics();
            } else {
                alert("Failed to save heuristic: " + result.message);
            }
        } catch (error) {
            alert("Error saving heuristic: " + error.message);
        }
    }

    async testRegexPattern() {
        /**
         * Test regex pattern against sample strings
         */
        const regex = document.getElementById("heuristicRegex").value;
        const testStringsText = document.getElementById("regexTestStrings").value;
        const testStrings = testStringsText.split("\n").filter((s) => s.trim());

        if (!regex) {
            alert("Please enter a regex pattern");
            return;
        }

        if (testStrings.length === 0) {
            alert("Please enter at least one test string");
            return;
        }

        try {
            const response = await fetch("/api/heuristics/test", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ regex, test_strings: testStrings }),
            });

            const data = await response.json();

            if (data.status === "success") {
                this.displayRegexTestResults(data.result);
            } else {
                alert("Test failed: " + data.message);
            }
        } catch (error) {
            alert("Error testing regex: " + error.message);
        }
    }

    displayRegexTestResults(result) {
        /**
         * Display regex test results
         */
        const container = document.getElementById("regexTestResultsContent");
        const resultsDiv = document.getElementById("regexTestResults");

        if (!result.valid) {
            container.innerHTML = `<div style="color: #f44336; padding: 10px; background: #ffebee; border-radius: 4px;">
                ❌ Invalid regex: ${this.escapeHtml(result.error)}
            </div>`;
            resultsDiv.style.display = "block";
            return;
        }

        let html = '<table style="width: 100%; border-collapse: collapse;">';
        html += '<thead><tr style="border-bottom: 2px solid #e0e0e0;">';
        html += '<th style="padding: 10px; text-align: left;">Test String</th>';
        html += '<th style="padding: 10px; text-align: center;">Result</th>';
        html += "</tr></thead><tbody>";

        result.matches.forEach((match) => {
            const icon = match.matched ? "✅" : "❌";
            const color = match.matched ? "#4CAF50" : "#9E9E9E";
            html += '<tr style="border-bottom: 1px solid #e0e0e0;">';
            html += `<td style="padding: 10px; font-family: monospace;">${this.escapeHtml(match.string)}</td>`;
            html += `<td style="padding: 10px; text-align: center; color: ${color};">${icon} ${match.matched ? "Match" : "No Match"}</td>`;
            html += "</tr>";
        });

        html += "</tbody></table>";
        container.innerHTML = html;
        resultsDiv.style.display = "block";
    }

    // ============================================================================
    // Expert Feedback Methods.
    // Expert feedback.
    // ============================================================================

    async loadExpertFeedback() {
        /**
         * Load and display expert feedback
         */
        try {
            const response = await fetch("/api/expert_feedback");
            const data = await response.json();

            if (data.status === "success") {
                this.displayExpertFeedback(data);
            } else {
                console.error("Failed to load expert feedback:", data.message);
            }
        } catch (error) {
            console.error("Error loading expert feedback:", error);
        }
    }

    async manageKnowledge(action, section = null, key = null) {
        const descriptions = {
            withdraw:
                "Withdraw this stored judgment and its accumulated selections? Derived rules will be rebuilt; the record will remain in history.",
            rebuild: "Rebuild automatic rules from active judgments? Manual rules will be kept.",
            reset: "Start a new policy scope? All active judgments and rules, including manual rules, will be archived. New approvals are required.",
        };
        if (!window.confirm(descriptions[action])) return;
        const reason = window.prompt("Reason for this knowledge change:");
        if (!reason || !reason.trim()) return;
        try {
            const response = await fetch("/api/expert_feedback/manage", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    action,
                    section,
                    key,
                    reason,
                    revision: this.knowledgeRevision,
                }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.message || "Update failed");
            await this.loadExpertFeedback();
            window.alert(
                "Knowledge updated. Start a new analysis before approving pending proposals."
            );
        } catch (error) {
            window.alert(error.message);
        }
    }

    displayExpertFeedback(data) {
        /**
         * Display expert feedback data
         */
        this.knowledgeRevision = data.revision;
        const content = document.getElementById("expertFeedbackContent");
        let controls = document.getElementById("knowledgeControls");
        if (!controls) {
            controls = document.createElement("div");
            controls.id = "knowledgeControls";
            controls.style.padding = "20px";
            content.prepend(controls);
        }
        controls.replaceChildren();
        for (const [action, label] of [
            ["rebuild", "Rebuild rules"],
            ["reset", "Start new policy scope"],
        ]) {
            const button = document.createElement("button");
            button.className = "btn btn-secondary";
            button.textContent = label;
            button.onclick = () => this.manageKnowledge(action);
            controls.append(button);
        }
        const history = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = `Knowledge changes (${(data.history || []).length})`;
        history.append(summary);
        for (const event of [...(data.history || [])].reverse()) {
            const entry = document.createElement("p");
            entry.textContent = `${event.timestamp}: ${event.action} — ${event.reason}`;
            history.append(entry);
        }
        controls.append(history);
        // Update statistics.
        const stats = data.statistics || {};
        document.getElementById("efTotalSelections").textContent = stats.total_selections || 0;
        document.getElementById("efUniqueColumns").textContent = stats.unique_columns || 0;
        document.getElementById("efPiiClassifications").textContent =
            stats.pii_classifications || 0;
        document.getElementById("efLastUpdated").textContent = stats.last_updated
            ? new Date(stats.last_updated).toLocaleString()
            : "-";

        // Display method preferences.
        const techContainer = document.getElementById("methodPreferencesContainer");
        const techPrefs = data.method_preferences || [];

        if (techPrefs.length === 0) {
            techContainer.innerHTML = `
                <div class="empty-state" style="padding: 30px;">
                    <div class="empty-icon">📊</div>
                    <h3 class="empty-title">No Method Preferences</h3>
                    <p class="empty-description">
                        Reviewed selections, including confirmed defaults, will appear here after Stage 3.
                    </p>
                </div>
            `;
        } else {
            let techHtml = `<table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background: var(--bg-tertiary);">
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid var(--border-color);">Column</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid var(--border-color);">PII Type</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid var(--border-color);">Preferred Method</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid var(--border-color);">Selections</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid var(--border-color);">Method Counts</th>
                    </tr>
                </thead>
                <tbody>`;

            techPrefs.forEach((pref) => {
                const methodCountsStr = Object.entries(pref.method_counts || {})
                    .map(([method, count]) => `${method}: ${count}`)
                    .join(", ");

                techHtml += `
                    <tr style="border-bottom: 1px solid var(--border-color);">
                        <td style="padding: 12px; font-weight: 500;">${this.escapeHtml(pref.column_name)}</td>
                        <td style="padding: 12px;">
                            <span class="badge ${pref.pii_type === "PII" ? "badge-danger" : "badge-success"}">${pref.pii_type}</span>
                        </td>
                        <td style="padding: 12px; font-family: monospace; color: var(--accent-color);">${pref.preferred_method || "-"}</td>
                        <td style="padding: 12px; text-align: center;">${pref.total_selections}</td>
                        <td style="padding: 12px; font-size: 0.85rem; color: var(--text-secondary);">${methodCountsStr || "-"}</td>
                    </tr>
                `;
            });

            techHtml += "</tbody></table>";
            techContainer.innerHTML = techHtml;
        }

        // Display PII classification changes.
        const piiContainer = document.getElementById("piiClassificationContainer");
        const piiChanges = data.pii_classifications || [];

        if (piiChanges.length === 0) {
            piiContainer.innerHTML = `
                <div class="empty-state" style="padding: 30px;">
                    <div class="empty-icon">🏷️</div>
                    <h3 class="empty-title">No PII Classification Changes</h3>
                    <p class="empty-description">
                        PII classification changes will appear here after experts reclassify columns in Stage 3.
                    </p>
                </div>
            `;
        } else {
            let piiHtml = `<table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background: var(--bg-tertiary);">
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid var(--border-color);">Column</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid var(--border-color);">Current Classification</th>
                        <th style="padding: 12px; text-align: center; border-bottom: 2px solid var(--border-color);">Changes</th>
                        <th style="padding: 12px; text-align: left; border-bottom: 2px solid var(--border-color);">Rationale</th>
                    </tr>
                </thead>
                <tbody>`;

            piiChanges.forEach((change) => {
                piiHtml += `
                    <tr style="border-bottom: 1px solid var(--border-color);">
                        <td style="padding: 12px; font-weight: 500;">${this.escapeHtml(change.column_name)}</td>
                        <td style="padding: 12px;">
                            <span class="badge ${change.classification === "PII" ? "badge-danger" : "badge-success"}">${change.classification}</span>
                        </td>
                        <td style="padding: 12px; text-align: center;">${change.change_count}</td>
                        <td style="padding: 12px; font-size: 0.85rem; color: var(--text-secondary);">${this.escapeHtml(change.rationale) || "-"}</td>
                    </tr>
                `;
            });

            piiHtml += "</tbody></table>";
            piiContainer.innerHTML = piiHtml;
        }
        for (const [container, records] of [
            [techContainer, techPrefs],
            [piiContainer, piiChanges],
        ]) {
            container.querySelectorAll("tbody tr").forEach((row, index) => {
                const cell = document.createElement("td");
                const button = document.createElement("button");
                button.textContent = "Withdraw";
                button.className = "btn btn-secondary";
                button.onclick = () =>
                    this.manageKnowledge("withdraw", records[index].section, records[index].key);
                cell.append(button);
                row.append(cell);
            });
            const header = container.querySelector("thead tr");
            if (header) {
                const cell = document.createElement("th");
                cell.textContent = "Review";
                header.append(cell);
            }
        }
    }

    // ============================================================================
    // Audit Log Methods.
    // Audit logs.
    // ============================================================================

    async loadAuditLog(filters = {}) {
        /**
         * Load and display audit log
         */
        try {
            let url = "/api/audit_log?";
            // Include session ID in request.
            // Scope the request to this session.
            if (this.state.sessionId) url += `session_id=${this.state.sessionId}&`;
            if (filters.event_type) url += `event_type=${filters.event_type}&`;
            if (filters.start_date) url += `start_date=${filters.start_date}&`;
            if (filters.end_date) url += `end_date=${filters.end_date}&`;

            const response = await fetch(url);
            const data = await response.json();

            if (data.status === "success") {
                this.displayAuditLog(data.events);
            } else {
                console.error("Failed to load audit log:", data.message);
            }
        } catch (error) {
            console.error("Error loading audit log:", error);
            document.getElementById("auditLogContainer").innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">❌</div>
                    <h3 class="empty-title">Error Loading Audit Log</h3>
                    <p class="empty-description">${error.message}</p>
                </div>
            `;
        }
    }

    displayAuditLog(events) {
        /**
         * Display audit log events
         */
        const container = document.getElementById("auditLogContainer");

        if (!events || events.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📜</div>
                    <h3 class="empty-title">No Audit Logs Available</h3>
                    <p class="empty-description">Audit logs will appear after pipeline execution</p>
                </div>
            `;
            return;
        }

        let html = '<div style="padding: 20px;">';

        events.forEach((event) => {
            const timestamp = new Date(event.timestamp).toLocaleString();
            const eventType = event.event_type;
            const data = event.data;

            html += `<div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; margin-bottom: 15px; background: #f9f9f9;">`;
            html += `<div style="display: flex; justify-content: space-between; margin-bottom: 10px;">`;
            html += `<span style="font-weight: bold; color: #2196F3;">${this.escapeHtml(eventType)}</span>`;
            html += `<span style="color: #666; font-size: 14px;">${timestamp}</span>`;
            html += `</div>`;
            html += `<pre style="background: white; padding: 10px; border-radius: 4px; overflow-x: auto; margin: 0;">${this.escapeHtml(JSON.stringify(data, null, 2))}</pre>`;
            html += `</div>`;
        });

        html += "</div>";
        container.innerHTML = html;
    }

    async showTableSchema(tableName) {
        /**
         * Show table schema in a modal window
         *
         * @param {string} tableName - Name of the table to display
         */
        const modal = document.getElementById("tableSchemaModal");
        const loadingState = document.getElementById("schemaLoadingState");
        const contentState = document.getElementById("schemaContent");
        const errorState = document.getElementById("schemaErrorState");
        const schemaTableName = document.getElementById("schemaTableName");
        const schemaTableBody = document.getElementById("schemaTableBody");
        const modalTitle = document.getElementById("schemaModalTitle");
        const modalSubtitle = document.getElementById("schemaModalSubtitle");

        // Show modal with loading state.
        modal.classList.add("active");
        loadingState.style.display = "block";
        contentState.style.display = "none";
        errorState.style.display = "none";

        // Update modal title.
        modalTitle.textContent = `📊 Table Schema: ${tableName}`;
        modalSubtitle.textContent = `View detailed schema information for ${tableName}`;

        try {
            const response = await fetch(`/get_table_schema/${encodeURIComponent(tableName)}`);
            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || "Failed to load schema");
            }

            // Hide loading, show content.
            loadingState.style.display = "none";
            contentState.style.display = "block";

            // Populate table name.
            schemaTableName.textContent = data.table_name;

            // Populate schema table.
            schemaTableBody.innerHTML = "";
            if (data.schema && data.schema.length > 0) {
                data.schema.forEach((column, index) => {
                    const row = document.createElement("tr");
                    row.style.borderBottom = "1px solid #dee2e6";

                    // Column number.
                    const cellNum = document.createElement("td");
                    cellNum.style.padding = "12px";
                    cellNum.textContent = index + 1;
                    row.appendChild(cellNum);

                    // Column name.
                    const cellName = document.createElement("td");
                    cellName.style.padding = "12px";
                    cellName.style.fontWeight = "500";
                    cellName.textContent = column.column_name || column.name || "-";
                    row.appendChild(cellName);

                    // Data type.
                    const cellType = document.createElement("td");
                    cellType.style.padding = "12px";
                    cellType.style.fontFamily = "'Fira Code', monospace";
                    cellType.style.color = "#0066cc";
                    cellType.textContent = column.data_type || column.type || "-";
                    row.appendChild(cellType);

                    // Nullable.
                    const cellNullable = document.createElement("td");
                    cellNullable.style.padding = "12px";
                    const isNullable = column.is_nullable || column.nullable;
                    if (isNullable === true || isNullable === "YES" || isNullable === 1) {
                        cellNullable.innerHTML = '<span style="color: #28a745;">✓ Yes</span>';
                    } else {
                        cellNullable.innerHTML = '<span style="color: #dc3545;">✗ No</span>';
                    }
                    row.appendChild(cellNullable);

                    // Primary key.
                    const cellPK = document.createElement("td");
                    cellPK.style.padding = "12px";
                    const isPK = column.is_primary_key || column.primary_key || column.pk;
                    if (isPK === true || isPK === "YES" || isPK === 1) {
                        cellPK.innerHTML = '<span style="color: #ffc107;">🔑 Yes</span>';
                    } else {
                        cellPK.innerHTML = '<span style="color: #6c757d;">-</span>';
                    }
                    row.appendChild(cellPK);

                    // Description (Comment)
                    const cellDesc = document.createElement("td");
                    cellDesc.style.padding = "12px";
                    cellDesc.style.color = "#495057";
                    cellDesc.style.fontSize = "14px";
                    const description = column.comment || column.description || "-";
                    cellDesc.textContent = description;
                    row.appendChild(cellDesc);

                    schemaTableBody.appendChild(row);
                });
            } else {
                const row = document.createElement("tr");
                const cell = document.createElement("td");
                cell.colSpan = 6;
                cell.style.padding = "20px";
                cell.style.textAlign = "center";
                cell.style.color = "#6c757d";
                cell.textContent = "No schema information available";
                row.appendChild(cell);
                schemaTableBody.appendChild(row);
            }
        } catch (error) {
            // Hide loading, show error.
            loadingState.style.display = "none";
            errorState.style.display = "block";
            document.getElementById("schemaErrorMessage").textContent = error.message;

            this.log(`[ERROR] Error loading schema for ${tableName}: ${error.message}`, "error");
        }
    }

    // ============================================================================
    // Session Monitoring Methods.
    // Session monitoring.
    // ============================================================================

    async loadSessions() {
        /**
         * Load and display all active sessions for monitoring
         */
        try {
            const response = await fetch("/api/sessions");
            const data = await response.json();

            if (data.status === "success") {
                this.state.monitoredSessions = data.sessions;
                this.displaySessionsList(data.sessions);
                this.updateSessionCount(data.count);
            } else {
                console.error("Failed to load sessions:", data.message);
            }
        } catch (error) {
            console.error("Error loading sessions:", error);
        }
    }

    updateSessionCount(count) {
        /**
         * Update the session count badge in the tab
         */
        const badge = document.getElementById("sessionCount");
        if (badge) {
            badge.textContent = count;
            badge.style.display = count > 0 ? "inline-block" : "none";
        }
    }

    displaySessionsList(sessions) {
        /**
         * Display the list of sessions as tabs
         */
        const noSessionsState = document.getElementById("noSessionsState");
        const tabsContainer = document.getElementById("sessionTabsContainer");
        const tabsList = document.getElementById("sessionTabsList");
        const detailsViewer = document.getElementById("sessionDetailsViewer");

        // Filter out current session from monitoring list.
        // Exclude this client from the monitored sessions.
        const otherSessions = sessions.filter((s) => s.session_id !== this.state.sessionId);

        if (otherSessions.length === 0) {
            noSessionsState.style.display = "flex";
            tabsContainer.style.display = "none";
            detailsViewer.style.display = "none";
            return;
        }

        noSessionsState.style.display = "none";
        tabsContainer.style.display = "block";

        // Generate session tabs.
        // Create the session tab.
        tabsList.innerHTML = "";
        otherSessions.forEach((session, index) => {
            const tab = document.createElement("button");
            tab.className = `session-tab ${index === 0 ? "active" : ""}`;
            tab.dataset.sessionId = session.session_id;

            const statusIcon = session.is_running ? "🟢" : "⚪";
            const tablesText =
                session.tables.length > 0
                    ? session.tables.slice(0, 2).join(", ") +
                      (session.tables.length > 2 ? "..." : "")
                    : "No tables";

            tab.innerHTML = `
                <span class="session-tab-status">${statusIcon}</span>
                <span class="session-tab-id">${session.session_id_short}</span>
                <span class="session-tab-tables">${tablesText}</span>
            `;

            tab.onclick = () => this.selectMonitorSession(session.session_id);
            tabsList.appendChild(tab);
        });

        // Auto-select first session if none selected.
        // Select the first session when none is active.
        if (!this.state.selectedMonitorSession && otherSessions.length > 0) {
            this.selectMonitorSession(otherSessions[0].session_id);
        } else if (this.state.selectedMonitorSession) {
            // Refresh current session details.
            // Refresh the active session details.
            this.loadSessionLogs(this.state.selectedMonitorSession);
        }
    }

    async selectMonitorSession(sessionId) {
        /**
         * Select a session for monitoring
         */
        this.state.selectedMonitorSession = sessionId;
        this.state.sessionLogOffset = 0;

        // Update tab active state.
        // Update the active tab.
        document.querySelectorAll(".session-tab").forEach((tab) => {
            tab.classList.toggle("active", tab.dataset.sessionId === sessionId);
        });

        // Show details viewer.
        // Show the detail viewer.
        const detailsViewer = document.getElementById("sessionDetailsViewer");
        detailsViewer.style = "";

        // Load session logs.
        // Load the session log.
        await this.loadSessionLogs(sessionId);
    }

    async loadSessionLogs(sessionId, append = false) {
        /**
         * Load logs from a specific session
         */
        try {
            const limit = 200;
            const offset = append ? this.state.sessionLogOffset : 0;

            const response = await fetch(
                `/api/sessions/${sessionId}/logs?offset=${offset}&limit=${limit}`
            );
            const data = await response.json();

            if (data.status === "success") {
                this.displaySessionDetails(data, append);

                if (data.has_more) {
                    this.state.sessionLogOffset = offset + limit;
                }
            } else {
                console.error("Failed to load session logs:", data.message);
            }
        } catch (error) {
            console.error("Error loading session logs:", error);
        }
    }

    displaySessionDetails(data, append = false) {
        /**
         * Display session details and logs
         */
        // Update header info.
        // Update the session header.
        document.getElementById("selectedSessionTitle").textContent =
            `Session: ${data.session_id_short}`;

        const statusBadge = document.getElementById("selectedSessionStatus");
        statusBadge.textContent = data.is_running ? "🟢 Running" : "⚪ Idle";
        statusBadge.className = `session-status-badge ${data.is_running ? "running" : "idle"}`;

        document.getElementById("selectedSessionStep").textContent = data.current_step || "-";
        document.getElementById("selectedSessionTables").textContent =
            data.tables.length > 0 ? `📊 ${data.tables.join(", ")}` : "No tables";

        // Display logs.
        // Render the log.
        const logsViewer = document.getElementById("sessionLogsViewer");

        if (!append) {
            logsViewer.innerHTML = "";
        }

        if (data.logs && data.logs.length > 0) {
            data.logs.forEach((log) => {
                const logEntry = document.createElement("div");
                logEntry.className = "log-entry";
                logEntry.innerHTML = this.formatLogMessage(log);
                logsViewer.appendChild(logEntry);
            });

            // Auto-scroll to bottom.
            // Scroll to the latest log entry.
            logsViewer.scrollTop = logsViewer.scrollHeight;
        } else if (!append) {
            logsViewer.innerHTML = `
                <div class="empty-state" style="padding: 20px;">
                    <div class="empty-icon">📋</div>
                    <h3 class="empty-title">No Logs Yet</h3>
                    <p class="empty-description">Logs will appear here when the session starts processing.</p>
                </div>
            `;
        }
    }

    formatLogMessage(message) {
        /**
         * Format log message with proper styling
         */
        let formatted = this.escapeHtml(message);

        // Highlight different log levels.
        // Highlight each log level.
        if (formatted.includes("✓") || formatted.includes("✅")) {
            formatted = `<span style="color: #28a745;">${formatted}</span>`;
        } else if (
            formatted.includes("✗") ||
            formatted.includes("❌") ||
            formatted.includes("Error")
        ) {
            formatted = `<span style="color: #dc3545;">${formatted}</span>`;
        } else if (formatted.includes("⚠️") || formatted.includes("Warning")) {
            formatted = `<span style="color: #ffc107;">${formatted}</span>`;
        } else if (formatted.includes("→") || formatted.includes("🔍")) {
            formatted = `<span style="color: #17a2b8;">${formatted}</span>`;
        }

        return formatted;
    }

    toggleSessionAutoRefresh() {
        /**
         * Toggle auto-refresh for session monitoring
         */
        this.state.sessionAutoRefresh = !this.state.sessionAutoRefresh;

        const btn = document.getElementById("autoRefreshToggle");
        if (this.state.sessionAutoRefresh) {
            btn.innerHTML = '<span class="btn-icon">⏱️</span><span>Auto-refresh: ON</span>';
            btn.classList.add("active");

            // Start auto-refresh interval (every 2 seconds)
            // Refresh every two seconds.
            this.state.sessionAutoRefreshInterval = setInterval(() => {
                if (this.state.selectedMonitorSession) {
                    this.loadSessionLogs(this.state.selectedMonitorSession);
                }
                this.loadSessions();
            }, 2000);
        } else {
            btn.innerHTML = '<span class="btn-icon">⏱️</span><span>Auto-refresh: OFF</span>';
            btn.classList.remove("active");

            // Stop auto-refresh.
            // Stop automatic refresh.
            if (this.state.sessionAutoRefreshInterval) {
                clearInterval(this.state.sessionAutoRefreshInterval);
                this.state.sessionAutoRefreshInterval = null;
            }
        }
    }

    // ============================================================================
    // Logo Click Handler.
    // Logo navigation.
    // ============================================================================

    handleLogoClick() {
        /**
         * Handle logo click - warn if pipeline is running
         */
        if (this.state.isRunning) {
            const confirmed = confirm(
                "⚠️ Pipeline is currently running!\n\n" +
                    "Refreshing the page will disconnect from the current session, " +
                    "but the pipeline will continue running in the background.\n\n" +
                    'You can monitor it from the "Sessions" tab after refresh.\n\n' +
                    "Do you want to refresh the page?"
            );

            if (!confirmed) {
                return;
            }
        }

        // Reload the page.
        // Reload the page.
        location.reload();
    }

    // ============================================================================
    // Add Table Modal Methods.
    // Table creation dialog.
    // ============================================================================

    openAddTableModal() {
        /**
         * Open the Add Table DDL modal
         */
        const modal = document.getElementById("addTableModal");
        const ddlInput = document.getElementById("ddlInput");
        const errorMessage = document.getElementById("ddlErrorMessage");

        // Clear previous input and error.
        // Reset the input and validation errors.
        ddlInput.value = "";
        errorMessage.style.display = "none";
        errorMessage.textContent = "";

        modal.classList.add("active");
    }

    closeAddTableModal() {
        /**
         * Close the Add Table DDL modal
         */
        const modal = document.getElementById("addTableModal");
        modal.classList.remove("active");
    }

    async executeAddTable() {
        /**
         * Execute the DDL and create the table
         */
        const ddlInput = document.getElementById("ddlInput");
        const errorMessage = document.getElementById("ddlErrorMessage");
        const executeBtn = document.getElementById("executeAddTableBtn");

        const ddl = ddlInput.value.trim();

        if (!ddl) {
            errorMessage.textContent = "❌ Please enter a CREATE TABLE statement";
            errorMessage.style.display = "block";
            return;
        }

        // Disable button during execution.
        // Disable the button while the request is pending.
        executeBtn.disabled = true;
        executeBtn.innerHTML = '<span class="btn-icon">⏳</span><span>Creating...</span>';

        try {
            const response = await fetch("/execute_ddl", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ ddl: ddl }),
            });

            const data = await response.json();

            if (data.error) {
                errorMessage.textContent = `❌ ${data.error}`;
                errorMessage.style.display = "block";
                return;
            }

            // Success - close modal and reload tables.
            // Close the dialog and refresh tables after success.
            this.closeAddTableModal();
            this.log(`[OK] ${data.message}`, "success");

            // Reload tables and select the new table.
            // Refresh the list and select the created table.
            await this.loadTables();

            // Select the newly created table.
            // Select the created table.
            const tableSelect = this.elements.tableSelect;
            const newOption = Array.from(tableSelect.options).find(
                (opt) => opt.value === data.table_name
            );
            if (newOption) {
                newOption.selected = true;
            }
        } catch (error) {
            errorMessage.textContent = `❌ Error: ${error.message}`;
            errorMessage.style.display = "block";
        } finally {
            // Re-enable button.
            // Enable the button.
            executeBtn.disabled = false;
            executeBtn.innerHTML = '<span class="btn-icon">💾</span><span>Execute</span>';
        }
    }

    updateDropTableButtonState() {
        /**
         * Update the Drop Table button state based on table selection
         */
        const dropTableBtn = document.getElementById("dropTableBtn");
        if (!dropTableBtn) return;

        const selectedTables = Array.from(this.elements.tableSelect.selectedOptions).map(
            (opt) => opt.value
        );

        if (selectedTables.length === 1) {
            dropTableBtn.disabled = false;
            dropTableBtn.title = `Drop table: ${selectedTables[0]}`;
        } else if (selectedTables.length > 1) {
            dropTableBtn.disabled = true;
            dropTableBtn.title = "Select only one table to drop";
        } else {
            dropTableBtn.disabled = true;
            dropTableBtn.title = "Select a table to drop";
        }
    }

    confirmDropTable() {
        /**
         * Show confirmation dialog before dropping a table
         */
        const selectedTables = Array.from(this.elements.tableSelect.selectedOptions).map(
            (opt) => opt.value
        );

        if (selectedTables.length !== 1) {
            alert("⚠️ Please select exactly one table to drop.\n테이블 하나만 선택해주세요.");
            return;
        }

        const tableName = selectedTables[0];

        // Show confirmation dialog.
        // Request confirmation before deleting the table.
        const confirmed = confirm(
            `⚠️ WARNING: This action cannot be undone!\n` +
                `경고: 이 작업은 되돌릴 수 없습니다!\n\n` +
                `Are you sure you want to drop the table "${tableName}"?\n` +
                `정말로 "${tableName}" 테이블을 삭제하시겠습니까?\n\n` +
                `All data in this table will be permanently deleted.\n` +
                `이 테이블의 모든 데이터가 영구적으로 삭제됩니다.`
        );

        if (confirmed) {
            this.dropTable(tableName);
        }
    }

    async dropTable(tableName) {
        /**
         * Drop the specified table from the database
         */
        const dropTableBtn = document.getElementById("dropTableBtn");

        // Disable button during execution.
        // Disable the button while the request is pending.
        if (dropTableBtn) {
            dropTableBtn.disabled = true;
            dropTableBtn.innerHTML = '<span class="btn-icon">⏳</span><span>Dropping...</span>';
        }

        try {
            const response = await fetch("/drop_table", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ table_name: tableName }),
            });

            const data = await response.json();

            if (data.error) {
                alert(`❌ Error: ${data.error}`);
                this.log(`[ERROR] Failed to drop table: ${data.error}`, "error");
                return;
            }

            // Success - reload tables.
            // Refresh the table list after success.
            this.log(`[OK] ${data.message}`, "success");
            await this.loadTables();
        } catch (error) {
            alert(`❌ Error: ${error.message}`);
            this.log(`[ERROR] Error dropping table: ${error.message}`, "error");
        } finally {
            // Re-enable button and update state.
            // Enable the button and update its state.
            if (dropTableBtn) {
                dropTableBtn.innerHTML = '<span class="btn-icon">-</span><span>Drop Table</span>';
            }
            this.updateDropTableButtonState();
        }
    }
}

let app;
window.addEventListener("DOMContentLoaded", () => {
    app = new PseuDRAGONApp();
});
