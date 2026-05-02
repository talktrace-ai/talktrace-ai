# TalkTrace AI neo

<p align="left">
    <picture>
        <source media="(prefers-color-scheme: light)" srcset="images/light.png">
        <source media="(prefers-color-scheme: dark)" srcset="images/light.png">
        <img src="images/light.png" alt="TalkTrace AI neo" width="1280">
    </picture>
</p>

LLM-assisted analysis of classroom and small-group transcripts. Quantitative metrics, qualitative coding, inter-coder reliability — packaged as a desktop app.

> **Highlights** — Autopilot two-LLM workflow · N-rater κ / α / Fleiss with bootstrap p-values · Code-transition heatmap and over-time views · Auto-generated methods paragraph + reproducibility fingerprint · DOCX / PDF / XLSX / HTML / CSV exports · Light/Dark themes · EN/DE UI

📋 **Full feature list →** [FEATURES.md](FEATURES.md)

---

## About

**TalkTrace-AI-neo** is an actively developed fork of [TalkTrace-AI](https://github.com/talktrace-ai/talktrace-ai) — a FLOSS, platform-independent web app for analysing verbal interaction in classroom and small-group settings. Built on [Shiny for Python](https://shiny.posit.co/py/), it leverages LLMs to produce both **quantitative** metrics (participation, conversation shares) and **qualitative** coding (speech acts), and exports them as structured reports.

The `neo` fork extends the original toward dialogue analysis **without a teacher present** (e.g. small-group student discussions), adds an Autopilot for two-LLM workflows, structured outputs, extended inter-coder metrics, custom themes, and a bilingual UI.

**Backends:** [OpenAI](https://platform.openai.com/) · [Anthropic](https://www.anthropic.com/api) · [Groq](https://groq.com/) · [Ollama](https://ollama.com/) (local & cloud)

---

## Quickstart

**Python ≥ 3.12 required** (development target: 3.13). Then pick your OS:

<details>
<summary><strong>🪟 Windows</strong></summary>

Double-click `start.bat`, or from a terminal:

```bat
start.bat
```

<details>
<summary>Python install</summary>

Download from [python.org](https://www.python.org/downloads/windows/) and ensure *"Add python.exe to PATH"* is enabled — otherwise `start.bat` cannot locate the interpreter.

</details>

</details>

<details>
<summary><strong>🍎 macOS</strong></summary>

```bash
chmod +x start.sh
./start.sh
```

No additional system dependencies — the native window uses the Cocoa/WebKit backend that ships with macOS.

<details>
<summary>Python install</summary>

The Python shipped with macOS is typically outdated (Sequoia ships with 3.9). Install a current version from [python.org](https://www.python.org/downloads/macos/) or via [Homebrew](https://brew.sh/) (`brew install python@3.13`).

</details>

</details>

<details>
<summary><strong>🐧 Linux</strong></summary>

```bash
chmod +x start.sh
./start.sh
```

`start.sh` detects missing `python3-venv` / `python3-pip` and offers to install them via `apt` / `dnf` / `pacman` (one `sudo` prompt).

For a native desktop window (otherwise opens in your default browser):

```bash
sudo apt install gir1.2-webkit2-4.1 python3-gi    # Debian/Ubuntu
```

**Limitations:**
- PDF export unavailable (relies on Microsoft Word) — use DOCX instead.
- Without a system keyring (GNOME Keyring / KWallet via SecretService), API keys live only for the running session. Either start a keyring daemon, or rely on the bundled `keyrings.alt` file fallback.

<details>
<summary>Python install</summary>

Python 3.13 isn't yet in default repos of many distributions. On Debian/Ubuntu, the [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) provides current builds; for fully version-managed setups, [`pyenv`](https://github.com/pyenv/pyenv) is recommended.

</details>

<details>
<summary>Line-ending issue</summary>

If you cloned on Windows and copied to Linux, the `.sh` files may have CRLF endings and silently fail. Convert once:

```bash
sed -i 's/\r$//' start.sh dev.sh   # or: dos2unix start.sh dev.sh
```

</details>

</details>

<details>
<summary><strong>⚙️ Launcher flags & development mode</strong></summary>

| Flag (Unix) | Flag (Windows) | Effect |
|---|---|---|
| `--reinstall` | `/reinstall` | Recreate the virtual environment from scratch |
| `--nowindow` | `/nowindow` | Start headless — open at <http://localhost:8000> |
| `--setup-only` | — | Provision the venv + dependencies, then exit |

For active development, use `dev.bat` (Windows) or `./dev.sh` (Linux/macOS) — runs the app under `shiny run --reload`, auto-restarting on `.py` saves. Opens in your default browser, so DevTools and tab refresh remain available. On a fresh checkout, `dev.sh` performs the same one-shot setup as `start.sh`. Stop with `Ctrl+C`.

</details>

<details>
<summary><strong>☁️ Ollama Cloud</strong></summary>

The `*-cloud` models require **both** a local [Ollama](https://ollama.com/) installation **and** an Ollama Cloud subscription. Cloud models confirmed to work as of April 2026: `Gemma4:31b-cloud`, `kimi-2.6:cloud`, `glm-5.1:cloud` (all on the free tier; slow at peak times).

</details>

---

## Interface

Five tabs — **Analysis** · **Results** · **Testing** · **Autopilot** · **Options** — plus a sidebar with model picker, session save/restore, dark-mode toggle, EN/DE switch, live cost estimate, and a quickstart checklist. Tab notification badges signal where action is needed.

<details>
<summary><strong>📥 Analysis tab</strong></summary>

Document Input panel:
- **Transcript** *(required)* — must follow the [noScribe](https://github.com/kaixxx/noScribe) format. The interactive multi-stage converter handles transcripts from other tools (e.g. [aTrain](https://github.com/JuergenFleiss/aTrain)) — speaker-label detection (noScribe `SPEAKER_XX` *and* inline notation like `Frau Müller:`, `L1:`, `Schüler 3:`), timestamp stripping, bracket-annotation review, per-speaker mapping, side-by-side preview before download.
- **Codebook** *(required for qualitative analysis)* — see the [example codebook](images/Example%20Codebook.docx). Codes apply to **all speakers** (teacher and students).
- **Teacher name** *(optional)* — if present in the transcript, enables teacher-specific metrics. Teacher auto-detection helps fill this in.
- **Group identifier and metadata** — used for report labelling.

Click *Analyze* in the sidebar; the app switches to Results on completion.

> **Cost estimate.** The sidebar figure is a *lower bound* — transcript+codebook length × input price × ~4 for output. Reasoning models may exceed this. A cumulative cost tracker (in *Options*) sums spend across all your analyses.

</details>

<details>
<summary><strong>📊 Results tab</strong></summary>

Split into quantitative and qualitative sections.

**Quantitative** (deterministic): participation metrics, conversation shares (absolute + relative), per-speaker turn stats (count / mean / median), three-segment over-time view.

<p align="center"><img src="images/Results-1.png" width="500"></p>

**Qualitative** (LLM-coded): per-speaker coding (every turn carries a `Sprecher` label), code distribution plot, coded-impulse table, over-time code distribution, **Markov-style code-transition heatmap** (which code follows which — IRE patterns made visible), and an **auto-generated methods paragraph** for paper manuscripts (copy-to-clipboard, EN/DE).

<p align="center"><img src="images/Results-2.png" width="500"></p>

</details>

<details>
<summary><strong>🤝 Testing tab</strong></summary>

Inter-coder reliability between two or more codings of the same transcript:

- **Cohen's κ** with bootstrap 95 % CI
- **Krippendorff's α** (nominal, 2–N coders) — robust on unbalanced distributions
- **Gwet's AC1 / Brennan-Prediger κ** — better on skewed prevalence
- **Percent agreement** — intuitive baseline
- **Per-code F1 / precision / recall** — see *which* codes drive disagreement
- **Confusion matrix** with full-screen view
- **Live glossary** — hover any metric for a one-line definition + paper reference

**Expert mode** (top toggle): N-rater agreement (Krippendorff α 2–N, Fleiss κ ≥3) with bootstrap p-value (H₀: κ=0) and conventional star notation (`***` p<0.001, `**` p<0.01, `*` p<0.05, `n.s.`).

<p align="center"><img src="images/Kappa.png" width="500"></p>

</details>

<details>
<summary><strong>🛫 Autopilot tab</strong></summary>

One-click two-LLM workflow for inter-coder reliability — collapses the manual *"run twice, switch model in between, export both, upload to Testing"* loop into a single button:

1. Upload transcript and codebook once.
2. Fill in group ID, group size, teacher name (validated against the transcript).
3. Pick multi-coding mode and a speaker filter (teacher / students / both).
4. Choose two different provider+model pairs for Coder A and Coder B (start button disables when both are identical, since the agreement would be trivial).
5. Click *Start Autopilot* — sequential coding (cleaner errors, predictable rate limits, no UI contention), both runs persisted as session pickles, results pushed straight into the Testing tab.

**Auto-generated reports.** Optional toggle builds a full report per coder in the background — Coder A's report is downloadable while Coder B is still running. Format and section selection configurable inline.

**Retry B only.** When Coder B fails after Coder A succeeds, A is preserved and only B re-runs.

**Coder chooser + swap.** After an autopilot run, switching to the Results tab opens a chooser dialog (*Coder A — model_a* / *Coder B — model_b*); a banner on the Results tab swaps to the other coder in one click without re-running.

</details>

<details>
<summary><strong>⚙️ Options tab</strong></summary>

<p align="center"><img src="images/Options.png" width="500"></p>

- **API configuration** — keys for OpenAI, Anthropic, Groq, Ollama. Keys live in the OS keyring (Keychain / Credential Manager / SecretService).
- **Local-only mode** — toggle that hides all cloud providers (Ollama only). Compliance lever for environments with strict data-protection rules.
- **Models for LLM Selection** — edit the registry (add/remove models, set per-million-token prices); changes propagate to the sidebar in real time.
- **Custom Prompts** — modify the system + user prompts used for qualitative coding; defaults restorable any time.
- **Cost tracker** — cumulative spend across all analyses, per provider.
- **Test the app** (gold-standard self-test) — runs a known fixture and shows expected vs. actual to build trust before you analyse real data.
- **Additional Options** — defaults for teacher name, group ID, class size, advanced toggles like streaming.

<p align="center"><img src="images/settings-2.png" width="500"></p>

</details>

---

## What's new in `neo`

The following extensions distinguish `neo` from upstream TalkTrace-AI. Click any item for details.

<details>
<summary><strong>Autopilot — one-click two-LLM coding</strong></summary>

Inter-coder reliability between two LLMs traditionally requires running the same transcript twice, switching the active model in between, exporting each report, and uploading both files into the Testing tab. The Autopilot tab collapses this into a single button: upload once, pick two models, click *Start*. Both runs are persisted as session pickles (with `_coderA` / `_coderB` suffixes), and the resulting DataFrames flow straight into the Testing tab — Cohen's κ, Krippendorff's α, percent agreement, the confusion matrix, and the per-code agreement table appear without any further interaction.

The two runs are explicitly **sequential** (not parallel) for clean error attribution and predictable rate limits. *Retry B only* covers the case where Coder B fails after Coder A succeeded. Optional **Auto-generated reports** build per-coder reports in the background — Coder A's is downloadable while Coder B is still running. The **Coder chooser** + Results-tab **coder swap** banner make it trivial to inspect either coder's analysis after the run.

</details>

<details>
<summary><strong>Extended inter-coder reliability — α, Fleiss, Gwet, F1, expert N-rater mode</strong></summary>

The Testing tab reports a broad set of statistics alongside Cohen's κ, addressing known limitations of κ in single-lesson datasets:

- **Percent agreement** — intuitive baseline that complements κ.
- **Krippendorff's α** (nominal, two coders) — robust to unbalanced code distributions and the κ-paradox.
- **Gwet's AC1** + **Brennan-Prediger κ** — better behaved on skewed prevalence.
- **Bootstrap 95 % CI for κ** — 1000 resamples, fixed seed (percentile method), displayed as `κ = 0.62 [0.48, 0.74]`.
- **Per-code agreement table** — F1, precision, recall, counts — surfaces *which* codes drive disagreement.
- **Live glossary tooltips** — hover any metric for a one-line definition + paper reference.

**Expert mode.** A toggle at the top unlocks an extended dialog for more than two coders: pick a metric (Cohen's κ for exactly 2, Krippendorff's α for 2–N, Fleiss' κ for ≥3), upload N reports, get the value, a 95 % bootstrap CI, and a two-sided p-value (H₀: κ=0) with conventional star notation. Because providers and models are now plentiful, generating three or more independent codings is cheap, and N-rater agreement becomes practically achievable.

</details>

<details>
<summary><strong>Code-transition heatmap</strong></summary>

Markov-style matrix of which code follows which. In a Mercer-style classroom dataset, that's the difference between *"the teacher asks lots of explanation questions"* and *"explanation questions reliably trigger elaborated student answers, which the teacher then confirms with feedback"* — same code counts, very different conversational dynamics. Uncoded turns are skipped; multi-coding cells fall back to the priority-resolved code. Available as an opt-in section in DOCX / HTML / XLSX / CSV exports.

</details>

<details>
<summary><strong>Reproducibility fingerprint + auto-generated methods paragraph</strong></summary>

Every report carries a short hash of (codebook + system prompt + user prompt + model + transcript). Anyone reproducing the analysis can verify config alignment at a glance.

The **auto-generated methods paragraph** — copy-to-clipboard text for paper manuscripts in EN/DE — embeds the fingerprint, model name, codebook size, sample scope (number of pupils, participants, impulses, coded turns), and the date. Saves the typical 10-minute methods-section write-up.

</details>

<details>
<summary><strong>Structured outputs with codebook enums</strong></summary>

Where the provider supports it, the LLM is constrained at decoding time to emit only Shortcodes from your codebook and Sprecher labels from your transcript:

- **OpenAI** — strict `json_schema` mode
- **Anthropic** — `tool_use` with input_schema
- **Groq** — `json_schema` response format
- **Ollama** — `format=schema`

Eliminates hallucinated codes; falls back to unconstrained schema if a model rejects the strict variant.

</details>

<details>
<summary><strong>Multi-coding + codebook priority hierarchy</strong></summary>

A sidebar toggle (default off) lets the LLM assign multiple codes per utterance — useful for longer turns covering several themes. With the toggle off, classic single-code-per-turn behaviour is preserved.

A three-tier priority resolver drives both multi-coding ordering and single-coding tie-breaking:
1. Explicit priority line in the codebook, e.g. `Priorisierung: A1 > B2 > C3`.
2. Explicit `Priorität` / `Priority` column with numeric values.
3. Position of the entry in the codebook — earlier entries have higher priority.

Existing codebooks without any of these still work — they fall back to position-based priority automatically.

</details>

<details>
<summary><strong>Multi-format reports</strong></summary>

DOCX, PDF (Win/macOS), XLSX, HTML, plus a **long-format CSV / R datapack** for direct use in R, SPSS, Stata. Configurable sections — quantitative, over-time quantitative, qualitative, over-time qualitative, code transitions, code legend — all toggleable. Embedded plots and tables; the methods paragraph and fingerprint travel with the legend so reports remain self-describing.

</details>

<details>
<summary><strong>Streaming, theming, bilingual UI</strong></summary>

- **Streaming coding view** (opt-in) — codings appear progressively as the LLM produces them. Doesn't speed up analysis but makes long runs feel less opaque (especially for reasoning models). OpenAI / Anthropic / Groq / Ollama.
- **Soft Nordic** (light) and **Deep Forest** (dark) themes — same CSS-variable structure, layout-stable switch. Warm-grey surfaces, low-saturation sage accent, soft borders instead of box shadows; dark mode uses hierarchical lightness for elevation rather than shadow.
- **EN / DE** UI, switchable any time.

<p align="center"><img src="images/Interface_darkmode.png" width="500"></p>

</details>

<details>
<summary><strong>Local-only mode + data-protection acknowledgment</strong></summary>

Toggle in *Options* that hides all cloud providers (incl. Ollama Cloud), forcing routing to a local Ollama instance — a one-click compliance lever for schools with strict data-protection rules. A first-launch acknowledgment dialog ensures users actively confirm where their transcript data will be sent before any LLM call goes out.

</details>

<details>
<summary><strong>Interactive transcript-format converter</strong></summary>

Multi-stage wizard that analyses the uploaded transcript before conversion — detects speaker labels in noScribe (`SPEAKER_XX`) *and* inline notation (`Frau Müller:`, `L1:`, `Schüler 3:`), strips a wide range of timestamp formats (`[hh:mm:ss]`, `(hh:mm)`, line-leading times, ranges like `00:32:31:13 --> 00:33:02:21`), surfaces every bracket annotation (`[]`, `()`, `{}`, `<>`, `//...//`, `*...*`) and standalone marker (`-->`, `===`, `***`, `###`) for explicit keep-or-remove decisions, heuristic per-speaker mapping (teacher / `S01..SN` / ignore), and a final preview before download. Reliable for transcripts well beyond noScribe.

</details>

<details>
<summary><strong>Analysis without a teacher · per-speaker coding</strong></summary>

Qualitative coding runs even when no teacher is identified, enabling small-group student-only studies. Every coded turn carries a `Sprecher` label (`Lehrperson`, `S01`, `S02`, …), and statistics are reported per speaker so teacher and individual student contributions can be inspected separately.

</details>

<details>
<summary><strong>Quality-of-life</strong></summary>

- **Cumulative cost tracker** — total spend per provider across all analyses (not just the current run).
- **Live glossary tooltips** on every metric.
- **Tab notification badges** — at-a-glance status of where action is needed.
- **Quickstart checklist** in the sidebar — live ✓/✗ panel showing what's still missing for analysis.
- **Demo button** — load a sample analysis without API keys.
- **Gold-standard self-test** — one-click *Test the app* runs a known fixture and shows expected vs. actual.
- **Info / License tab** — developer info, GitHub/ORCID links, CC BY-NC 4.0 notice.

</details>

---

## Privacy

<details>
<summary><strong>How data is handled</strong></summary>

TalkTrace AI does **not** store transcripts or analysis results on any external server controlled by the developers. All data required for an analysis are held in browser/local memory while you interact with the tool.

Because LLM models are not hosted by the app (with the exception of a local Ollama instance), the backend communicates with external LLM providers during the qualitative coding step. The relevant transcript and codebook excerpts are transmitted via the provider's API — any server-side storage or logging then depends on that provider's policies and your account settings.

Sessions can be saved/restored locally as `.pkl` files; reports can be downloaded. **API keys** live in the OS encrypted credential vault — Keychain (macOS), Credential Manager (Windows), SecretService (GNOME Keyring, KWallet) on Linux.

For stricter scenarios, **local-only mode** (Ollama only) is available today; **GDPR-compliant EU-provider integration** with an AVV/DPA template is on the roadmap (see [FEATURES.md](FEATURES.md#-planned)).

</details>

---

## Credits

TalkTrace-AI-neo is a fork of TalkTrace-AI, in active development by [Simon Filler](https://orcid.org/0009-0008-8736-8831) at [TU Dortmund University](https://idif.sowi.tu-dortmund.de/institut/).

TalkTrace-AI was developed by [Jami Schorling](https://orcid.org/0009-0005-9007-2896) and [Dennis Hauk](https://orcid.org/0000-0002-5779-2876) at the [Chair for Research on Teaching and Learning in Civic Education](https://www.sozphil.uni-leipzig.de/institut-fuer-politikwissenschaft/arbeitsbereiche/professur-fuer-fachdidaktik-gemeinschaftskunde/team/prof-dr-dennis-hauk), Leipzig University.

## Contributing

Contributions are welcome — open an issue or PR.

## License

[**CC BY-NC 4.0**](LICENSE) — *Let's socialize software for the open-source democratic stack!*
