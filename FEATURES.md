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

- **Dark mode** — Obsidian-inspired theme, toggleable in sidebar
- **Bilingual UI** — English & German, switchable any time
- **Onboarding tooltips** — hover help on every key control
- **Quickstart checklist** — live ✓/✗ panel showing what's ready
- **Demo button** — load a sample analysis without API keys
- **Auto tab-switch** — jumps to Results when analysis completes
- **Speaker filters** — code only the teacher, only the students, or both
- **Analysis without a teacher** — student-only group discussions fully supported

## 🚀 Setup & launchers

- **One-click launchers** for Windows (`start.bat`), macOS, Linux (`start.sh`)
- **Auto venv + dependency install** on first run
- **Native desktop window** (Cocoa / WebKit / GTK) or headless browser mode
- **Hot-reload dev mode** (`dev.bat` / `dev.sh`)
- **Distro-aware setup** — offers to install missing packages on Debian/Fedora/Arch

---

<sub>🤖 Co-developed with [Claude Code](https://claude.com/claude-code)</sub>
