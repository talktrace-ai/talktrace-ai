# 🎯 TalkTrace AI neo — Features

> What the app can do, top to bottom.

---

## 📥 Inputs

- **Transcript upload** — `.txt`, `.docx`, `.pdf`
- **Codebook upload** — `.txt`, `.docx`, `.pdf`, with live preview
- **Multi-stage format converter** — speaker mapping, bracket/timestamp stripping, side-by-side preview before download
- **Group metadata** — class ID, class size, teacher name (all optional)

## 🤖 LLM backends

- **Four providers** — OpenAI, Anthropic, Groq, Ollama (local & cloud)
- **Local-only mode** — toggle in *Options* hides all cloud providers (incl. Ollama Cloud), forcing routing to a local Ollama instance. Compliance lever for environments with strict data-protection rules.
- **Editable model registry** — add or remove models, set per-million-token pricing
- **Custom prompts** — edit system and user prompts, reset to default any time
- **Structured outputs with codebook enums** — Shortcode + Sprecher are decoder-side constrained to the codebook entries / transcript speakers (OpenAI strict json_schema, Anthropic tool_use input_schema, Groq json_schema, Ollama format=schema). Eliminates hallucinated codes; falls back to unconstrained schema if a model rejects the strict variant.
- **Live cost prediction** — lower-bound estimate updates as you type
- **Cumulative cost tracker** — total spend across all analyses, per provider, persisted between sessions
- **API keys in the OS keyring** — Keychain, Credential Manager, SecretService

## 📊 Quantitative results

- **Participation metrics** — class size, active participants, participation rate
- **Turn distribution plot** — words spoken by teacher vs. students
- **Per-speaker turn stats** — count, average length, median length
- **Over-time view** — three-segment breakdown across the lesson

## 🏷️ Qualitative results

- **Per-speaker coding** — every coded turn carries a speaker label
- **Multi-coding** — multiple codes per utterance, opt-in toggle
- **Codebook priority hierarchy** — priority line, explicit column, or codebook order
- **Code distribution plot** — frequency of each code across the conversation
- **Coded-impulse table** — speaker, turn index, code(s), utterance text
- **Over-time code distribution** — which codes emerge when in the lesson
- **Code-transition heatmap** — Markov-style matrix of which code follows which (uncoded turns skipped, multi-coding takes priority-resolved code). Surfaces dialogue dynamics like IRE patterns that frequency plots hide. Optional report section in DOCX/HTML/XLSX/CSV.
- **Most-frequent-code summary** + teacher talking rate with per-student breakdown
- **Live coding view (streaming)** — codings appear progressively, opt-in toggle

## 🤝 Inter-coder reliability

- **Cohen's κ** with bootstrap 95% confidence interval
- **Krippendorff's α** — robust on unbalanced distributions
- **Gwet's AC1 / Brennan-Prediger κ** — better behaved on skewed prevalence than Cohen's κ
- **Percent agreement** — intuitive baseline
- **Per-code F1, precision, recall** — see *which* codes drive disagreement
- **Confusion matrix** — full-screen view available
- **Live glossary tooltips** — hover any metric for a one-line definition + paper reference
- **Compares two reports** — DOCX, XLSX, HTML, HTM
- **Expert mode** — N-rater agreement (Krippendorff's α 2–N, Fleiss' κ ≥3)
- **Significance test** — bootstrap p-value (H₀: κ=0) with conventional star notation (`***`, `**`, `*`, `n.s.`)

## 🛫 Autopilot

- **One-click two-LLM workflow** — upload once, pick two models, click *Start*; codings + κ/α/confusion matrix appear without further interaction
- **Sequential coder runs** — cleaner error attribution, predictable rate limits, no UI contention
- **Identical-model guard** — start button disables when both coders are the same
- **Auto-generated reports per coder** — Coder A's report downloadable while Coder B is still running; format and sections configurable inline
- **Retry B only** — when Coder B fails after Coder A succeeded, A is preserved and only B re-runs
- **Side-by-side coding view** + collapsible per-speaker quantitative summary
- **Provider hints + tab badges** — at-a-glance status of the active configuration

## 📄 Reports

- **Four export formats** — DOCX, PDF (Win/macOS), XLSX, HTML
- **Long-format CSV / R datapack export** — stats-friendly bundle alongside DOCX/XLSX/PDF/HTML
- **Configurable sections** — quantitative, qualitative, over-time, code legend, all toggleable
- **Embedded plots and tables** — ready to share, no post-processing
- **Reproducibility fingerprint** — short hash of codebook + prompts + model + transcript, embedded in every report
- **Auto-generated methods paragraph** — copy-to-clipboard text for the methods section of papers (tool, model, codebook size, sample scope, fingerprint, date), bilingual, also embedded in the report legend

## 💾 Sessions

- **Auto-save to history** after every successful analysis
- **Manual history browser** — load, delete, save now
- **Session import/export** as `.pkl`
- **History reload is free** — no new LLM calls when restoring a saved session

## 🎨 Interface

- **Light & dark themes** — Soft Nordic (light) and Deep Forest (dark), toggleable in sidebar
- **Bilingual UI** — English & German, switchable any time
- **Onboarding tooltips** — hover help on every key control
- **Data-protection acknowledgment gate** — first-launch dialog requires active confirmation of where transcript data will be sent before any LLM call goes out
- **Quickstart checklist** — live ✓/✗ panel showing what's ready
- **Demo button** — load a sample analysis without API keys
- **Gold-standard self-test** — one-click *Test the app* runs a known fixture and shows expected vs. actual; trust-builder before users analyse their own data
- **Tab notification badges** — at-a-glance status of where action is needed
- **Auto tab-switch** — jumps to Results when analysis completes
- **Speaker filters** — code only the teacher, only the students, or both
- **Analysis without a teacher** — student-only group discussions fully supported
- **Coder chooser** — after an autopilot run, pick Coder A or Coder B to populate the Results pipeline
- **Coder swap** — banner on the Results tab switches between coders in one click, no re-run
- **Info / License tab** — developer info, GitHub/ORCID links, CC BY-NC 4.0 notice

## 🚀 Setup & launchers

- **One-click launchers** for Windows (`start.bat`), macOS, Linux (`start.sh`)
- **Auto venv + dependency install** on first run
- **Native desktop window** (Cocoa / WebKit / GTK) or headless browser mode
- **Hot-reload dev mode** (`dev.bat` / `dev.sh`)
- **Distro-aware setup** — offers to install missing packages on Debian/Fedora/Arch

---

## 📋 Planned

The list is ordered top-to-bottom in the suggested implementation sequence. Within each priority band, simpler items come first.

Priorities: 🟠 high — small effort, high payoff, do next · 🟡 medium — meaningful effort, on the roadmap · 🟢 low — long-term, not urgent.

### 🟠 High — quick wins, do next

1. **GDPR-compliant provider integration** — add LLM backends that process data inside the EU/EEA under GDPR-compliant terms (Aleph Alpha, Mistral EU, IONOS AI Model Hub, Azure OpenAI with EU data residency, etc.) so schools and research projects with German/EU data-protection requirements have a path that doesn't depend on running models locally. Needs research on which providers actually sign DPA/AVV contracts for academic use, and on how the situation looks in UK/US/CA/AU. Pairs with #2.
2. **DPA / AVV document template** — generate a pre-filled data-processing agreement template that researchers can hand to their institution's data-protection officer (controller name, processor name = the LLM provider, categories of data, transfer mechanism, etc.). The user said they have a DPA generator they want to plug in here. Pairs with #1.

### 🟡 Medium — meaningful effort, on the roadmap

3. **PII anonymisation** — automatic masking of student / school / teacher names *before* anything is sent to an LLM. Legal precondition for many school deployments.
4. **Analysis cancellation** — clean stop of a running LLM job mid-stream, with state cleanup and partial-result handling.
5. **Onboarding tutorial** — slideshow walking through the most important functions on first launch, coupled to the chosen mode (teacher vs. researcher).
6. **Two-mode UI (teacher / researcher)** — first-launch prompt, switchable in *Options*. Teacher mode shows only the tabs needed for self-analysis; researcher mode keeps everything active.
7. **Feedback tab for teachers** — coding metrics interpreted by an LLM into a plain-language narrative on behaviour, possible improvements, and the indicators behind them.
8. **Multi-lesson comparison / trend** — stack several lessons of the same teacher into a trend view. Turns the tool into a reflection device.

### 🟢 Low — long-term, not urgent

9. **REFI-QDA export** — interoperability with MAXQDA, NVivo, and atlas.ti for hybrid (LLM + manual) coding workflows.
10. **Goal-setting with tracking** — set targets ("teacher talk under 60%") and check progress across lessons. Builds on multi-lesson comparison.
11. **Multi-transcript projects** — folder-based studies aggregating stats across many transcripts under one condition.
12. **Codebook optimiser via LLM divergence** — high disagreement signals an underspecified codebook; surface diverging codes and tips, later return an optimised codebook automatically.
13. **PyPI release** (`pip install talktrace-ai-neo`) — package metadata, console entry point, settings dialog for API keys (the same prep .exe needs).
14. **Standalone distribution (.exe / Windows Store)** — PyInstaller-built signed installer first, Windows Store packaging later for the seriousness boost.

---

<sub>🤖 Co-developed with [Claude Code](https://claude.com/claude-code)</sub>
