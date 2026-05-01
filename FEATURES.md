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
- **Editable model registry** — add or remove models, set per-million-token pricing
- **Custom prompts** — edit system and user prompts, reset to default any time
- **Live cost prediction** — lower-bound estimate updates as you type
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
- **Most-frequent-code summary** + teacher talking rate with per-student breakdown
- **Live coding view (streaming)** — codings appear progressively, opt-in toggle

## 🤝 Inter-coder reliability

- **Cohen's κ** with bootstrap 95% confidence interval
- **Krippendorff's α** — robust on unbalanced distributions
- **Percent agreement** — intuitive baseline
- **Per-code F1, precision, recall** — see *which* codes drive disagreement
- **Confusion matrix** — full-screen view available
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
- **Configurable sections** — quantitative, qualitative, over-time, code legend, all toggleable
- **Embedded plots and tables** — ready to share, no post-processing

## 💾 Sessions

- **Auto-save to history** after every successful analysis
- **Manual history browser** — load, delete, save now
- **Session import/export** as `.pkl`
- **History reload is free** — no new LLM calls when restoring a saved session

## 🎨 Interface

- **Light & dark themes** — Soft Nordic (light) and Deep Forest (dark), toggleable in sidebar
- **Bilingual UI** — English & German, switchable any time
- **Onboarding tooltips** — hover help on every key control
- **Quickstart checklist** — live ✓/✗ panel showing what's ready
- **Demo button** — load a sample analysis without API keys
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

1. **Local-only mode** — toggle in *Options* that disables all cloud providers (Ollama only). Compliance lever for schools with strict data-protection rules.
2. **Live glossary** — hover any metric (κ, α, F1, …) for a one-line definition and paper reference. Builds confidence in the numbers and groundwork for the Feedback tab.
3. **Cumulative cost tracker** — total spend across all analyses, per provider, per project. Currently the cost estimate only covers the current run.
4. **Reproducibility fingerprint** — single hash combining codebook + prompt + model + transcript hash, embedded in every report. Anyone reproducing the analysis can verify config alignment at a glance.
5. **Long-format CSV / R datapack export** — stats-friendly export alongside DOCX/XLSX/PDF/HTML for direct use in R, SPSS, Stata.
6. **Gwet's AC1 / Brennan-Prediger κ** — additional inter-coder metrics that handle skewed prevalence better than Cohen's κ. Methodologically increasingly expected.
7. **Gold-standard self-test** — one-click *Test the app* button that runs a known fixture and shows expected vs. actual. Builds trust before users analyse their own data.

### 🟡 Medium — meaningful effort, on the roadmap

8. **Structured LLM outputs** — enforce a JSON schema (OpenAI Structured Outputs / Anthropic Tool Use) so the model can only emit codes that exist in the codebook. Eliminates hallucinated codes and parser failures as a class. Foundation for everything LLM-touching that comes after.
9. **Auto-generated methods section** — paragraph for the methods section of papers ("Coded with X, model Y, prompt version Z, dated …, κ=…"). Builds on the reproducibility fingerprint.
10. **PII anonymisation** — automatic masking of student / school / teacher names *before* anything is sent to an LLM. Legal precondition for many school deployments.
11. **Analysis cancellation** — clean stop of a running LLM job mid-stream, with state cleanup and partial-result handling.
12. **Sequence analysis / code transitions** — Markov matrix or Sankey diagram showing which code follows which. Reveals dialogue dynamics that frequency plots hide.
13. **Onboarding tutorial** — slideshow walking through the most important functions on first launch, coupled to the chosen mode (teacher vs. researcher).
14. **Two-mode UI (teacher / researcher)** — first-launch prompt, switchable in *Options*. Teacher mode shows only the tabs needed for self-analysis; researcher mode keeps everything active.
15. **Feedback tab for teachers** — coding metrics interpreted by an LLM into a plain-language narrative on behaviour, possible improvements, and the indicators behind them.
16. **Multi-lesson comparison / trend** — stack several lessons of the same teacher into a trend view. Turns the tool into a reflection device.

### 🟢 Low — long-term, not urgent

17. **REFI-QDA export** — interoperability with MAXQDA, NVivo, and atlas.ti for hybrid (LLM + manual) coding workflows.
18. **Goal-setting with tracking** — set targets ("teacher talk under 60%") and check progress across lessons. Builds on multi-lesson comparison.
19. **Multi-transcript projects** — folder-based studies aggregating stats across many transcripts under one condition.
20. **Codebook optimiser via LLM divergence** — high disagreement signals an underspecified codebook; surface diverging codes and tips, later return an optimised codebook automatically.
21. **PyPI release** (`pip install talktrace-ai-neo`) — package metadata, console entry point, settings dialog for API keys (the same prep .exe needs).
22. **Standalone distribution (.exe / Windows Store)** — PyInstaller-built signed installer first, Windows Store packaging later for the seriousness boost.

---

<sub>🤖 Co-developed with [Claude Code](https://claude.com/claude-code)</sub>
