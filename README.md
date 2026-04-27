# TalkTrace AI neo

<p align="left">
    <picture>
        <source media="(prefers-color-scheme: light)" srcset="images/bright.svg">
        <source media="(prefers-color-scheme: dark)" srcset="images/dark.svg">
        <img src="images/logo_white.png" alt="TalkTrace AI neo" width="256">
    </picture>
</p>

## About

**TalkTrace-AI-neo** is an actively developed fork of [TalkTrace-AI](<!-- TODO: upstream repository URL -->), a FLOSS (Free/Libre Open Source Software), platform-independent web application for analysing verbal interaction in classroom and small-group settings. Built on [Shiny for Python](https://shiny.posit.co/py/), it leverages Large Language Models (LLMs) to produce both **quantitative** metrics (participation, conversation shares) and **qualitative** coding (speech acts) of transcribed dialogues, and exports them as structured reports.

The `neo` fork extends the original tool toward dialogue analysis **without a teacher present** (e.g. small-group student discussions), adds quality-of-life features such as dark mode and Ollama Cloud support, and introduces utilities for inter-coder reliability assessment. See [What's New in `neo`](#whats-new-in-neo) for the full list of changes.

**Supported LLM backends:**

- [OpenAI](https://platform.openai.com/) — paid API
- [Groq](https://groq.com/) — paid API
- [Anthropic](https://www.anthropic.com/api) — paid API
- [Ollama](https://ollama.com/) — local (free) or Ollama Cloud (free tier available, paid for premium models)

## Quickstart

The repository ships with launch helpers that create a virtual environment, install dependencies, and start the app.

### Prerequisites

**Python ≥ 3.12 is required** (development and testing target: 3.13). Check your installed version with `python --version` (Windows) or `python3 --version` (macOS/Linux); if it is below 3.12, install or upgrade as described below.

- **Windows** — download from the [official Python website](https://www.python.org/downloads/windows/). During installation, ensure the option *"Add python.exe to PATH"* is enabled, otherwise `start.bat` will not locate the interpreter when bootstrapping the virtual environment.
- **macOS** — the Python interpreter shipped with macOS is typically outdated (Sequoia, for instance, ships with 3.9). Install a current version from [python.org](https://www.python.org/downloads/macos/) or via [Homebrew](https://brew.sh/) (`brew install python@3.13`).
- **Linux** — Python 3.13 is not yet present in the default repositories of many distributions. On Debian/Ubuntu, the [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) provides current builds; for fully version-managed setups, [`pyenv`](https://github.com/pyenv/pyenv) is recommended.

### Windows

Double-click `start.bat`, or run from a terminal:

```bat
start.bat
```

### macOS

```bash
chmod +x start.sh
./start.sh
```

No additional system dependencies are required — the native window uses the Cocoa/WebKit backend that ships with macOS.

### Linux

```bash
chmod +x start.sh
./start.sh
```

For a native desktop window, install the WebKit/GTK bindings (Debian/Ubuntu):

```bash
sudo apt install gir1.2-webkit2-4.1 python3-gi
```

Without those packages, the app automatically falls back to opening in your default browser.

**Linux limitations:**

- PDF report export is not available (the export pipeline relies on Microsoft Word). Export to DOCX instead.
- Without a system keyring (GNOME Keyring / KWallet via SecretService), API keys are kept only for the running session. The app installs `keyrings.alt` as a file-based fallback; alternatively, start a keyring daemon (e.g. `gnome-keyring-daemon`) for persistent storage.

### Launcher flags

| Flag (Unix) | Flag (Windows) | Effect |
|---|---|---|
| `--reinstall` | `/reinstall` | Recreate the virtual environment from scratch |
| `--nowindow` | `/nowindow` | Start headless; access the app at <http://localhost:8000> |

### Ollama Cloud

The `*-cloud` models require **both** a local [Ollama](https://ollama.com/) installation **and** an Ollama Cloud subscription. Cloud models confirmed to work as of April 2026 are `Gemma4:31b-cloud`, `kimi-2.6:cloud`, and `glm-5.1:cloud` (all usable on the free tier).

## Interface

The workflow is organised into two main tabs — **Analysis** and **Results** — plus an **Options** tab for configuration. The sidebar provides shortcuts for LLM selection, session save/restore, a dark-mode toggle, and a language switch (EN/DE).

### Analysis tab

The Document Input panel accepts the following inputs:

- **Transcript** *(required)* — must follow the [noScribe](https://github.com/kaixxx/noScribe) format. A built-in converter transforms transcripts produced by other tools (e.g. [aTrain](https://github.com/JuergenFleiss/aTrain)) into the expected schema.
- **Codebook** *(required for qualitative analysis)* — see the [example codebook](images/Example%20Codebook.docx). Codes are applied to **all speakers** (teacher and students), so codebooks may equally target student speech acts.
- **Teacher name** *(optional)* — providing the teacher's identifier as it appears in the transcript enables teacher-specific metrics. If omitted, qualitative analysis still runs over all speakers.
- **Group identifier and metadata** — used for report labelling.

The analysis is started via the **Analyze** button in the sidebar. On completion, the app switches automatically to the Results tab.

> **Note on token prediction.** The cost estimate displayed in the sidebar is a *lower bound* only. It is computed from transcript and codebook length, the provider's input-token cost, and an assumed output ≈ 4 × input ratio. Reasoning models in particular may produce substantially longer outputs. Actual usage should be verified via the provider's own metrics.

### Results tab

Results are split into a quantitative and a qualitative section.

**Quantitative results** are computed deterministically (pattern matching, basic arithmetic) and report participation metrics together with visualisations of conversation shares (absolute and relative).

<p align="center">
  <img src="images/Results-1.png" width="500">
</p>

**Qualitative results** are produced by the selected LLM on the basis of the uploaded codebook. Each coded utterance carries a `Sprecher` label (`Lehrperson`, `S01`, `S02`, …), and statistics are reported per speaker, so that teacher contributions and individual student contributions can be inspected separately. Code distributions are summarised above the textual display; sections without matching data show a *No data* placeholder.

<p align="center">
  <img src="images/Results-2.png" width="500">
</p>

### Options tab

<p align="center">
  <img src="images/Options.png" width="500">
</p>

- **API configuration** — manage API keys for OpenAI, Groq, Anthropic, and Ollama. For Ollama, the app detects a configured cloud key and otherwise falls back to a local Ollama instance on `localhost`.
- **Models for LLM Selection** — edit the list of selectable models; changes propagate to the sidebar in real time.
- **Custom Prompts** — modify the system and user prompts used for qualitative coding to fit specific analytical requirements; defaults can be restored at any time.
- **Additional Options** — adjust the default values for teacher name, group ID, and class size.

The configuration is stored locally in the app folder and can be partially reset via the corresponding button.

<p align="center">
  <img src="images/settings-2.png" width="500">
</p>

## What's New in `neo`

The following extensions and changes distinguish `neo` from the upstream TalkTrace-AI:

- **Analysis without a teacher.** Qualitative coding now runs even when no teacher is identified in the transcript, enabling the study of small-group student discussions.
- **Per-speaker qualitative coding.** The LLM codes utterances of *all* speakers (teacher and students). Each output entry carries a `Sprecher` field (e.g. `Lehrperson`, `S01`, `S02`), and the Results tab reports per-speaker statistics.
- **Inter-coder agreement.** Two analysis reports of the same transcript produced with different LLMs can be uploaded to compute [Cohen's κ](https://en.wikipedia.org/wiki/Cohen%27s_kappa) for the qualitative coding.

  <p align="center">
    <img src="images/Kappa.png" width="500">
  </p>

- **Ollama Cloud support.** A new API client integrates Ollama's hosted models alongside OpenAI, Groq, and Anthropic. Local Ollama remains supported as a fully offline backend.
- **Dark mode.** Obsidian-inspired theme, toggleable from the sidebar.

  <p align="center">
    <img src="images/Interface_darkmode.png" width="500">
  </p>

- **Windows launcher (`start.bat`).** Bootstraps a local `.venv`, installs dependencies, and starts the app (flags: `/reinstall`, `/nowindow`).
- **Updated prompts.** System and user prompts have been adapted to multi-speaker coding and the optional-teacher case.

## Privacy Note

TalkTrace-AI does not store transcripts or analysis results on any external server controlled by the developers. All data required for preparing and displaying an analysis are held in local memory in the browser during interaction with the tool.

Because LLM models are not hosted locally (with the exception of a local Ollama instance), the application backend communicates with external LLM providers during the qualitative coding step. When qualitative coding is enabled, the relevant parts of the transcript and the codebook are transmitted to the selected provider via its API. Any server-side storage or logging of these data therefore depends on the data-protection policies and technical settings of the chosen LLM service.

Raw LLM output and session data can be stored locally for later reuse via the export/import session controls, and processed outputs can be downloaded as result reports. API keys are stored in the operating system's encrypted credential vault — Keychain on macOS, Credential Manager on Windows, and SecretService-compatible backends (GNOME Keyring, KWallet) on Linux.

This architecture supports institutions that prefer to keep teaching and research data under their own control and aligns with recommendations that AI-supported learning analytics should be designed to minimise unnecessary data retention on external services.

## Credits

TalkTrace-AI-neo is a fork of TalkTrace-AI and is under ongoing development by [Simon Filler](https://orcid.org/0009-0008-8736-8831) at [TU Dortmund University](https://idif.sowi.tu-dortmund.de/institut/).

TalkTrace-AI was developed by [Jami Schorling](https://orcid.org/0009-0005-9007-2896) and [Dennis Hauk](https://orcid.org/0000-0002-5779-2876) at the [Chair for Research on Teaching and Learning in Civic Education](https://www.sozphil.uni-leipzig.de/institut-fuer-politikwissenschaft/arbeitsbereiche/professur-fuer-fachdidaktik-gemeinschaftskunde/team/prof-dr-dennis-hauk), Leipzig University, Germany.

## Contributing

Contributions are welcome. Please submit a pull request or open an issue on GitHub for enhancements or bug fixes.

## License

This project is licensed under the [**CC BY-NC 4.0**](https://creativecommons.org/licenses/by-nc/4.0/) license. See the [LICENSE](LICENSE) file for details. *Let's socialize software for the open-source democratic stack!*
