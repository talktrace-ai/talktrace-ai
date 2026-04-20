## About

## What's New / Änderungen

**English**
- **Dark mode**: Obsidian-inspired dark theme, toggled from the sidebar.
- **Ollama Cloud support**: new API client for Ollama alongside OpenAI, Groq, and Anthropic. Requires a local Ollama installation; `*-cloud` models additionally require an Ollama cloud subscription. Without a cloud key the app falls back to a local Ollama instance on `localhost`.
- **Qualitative coding of student utterances**: the LLM now codes speech acts of students (SuS) as well, not only the teacher. Output JSON carries a `Sprecher` field (e.g. `Lehrperson`, `S01`, `S02`) and the Results tab shows per-speaker statistics.
- **Analysis without a teacher**: specifying a teacher name is now optional — qualitative analysis runs even if no teacher is present in the transcript.
- **PowerShell launcher `run.ps1`**: bootstraps a local `.venv`, installs dependencies, starts the app and opens the browser. Flags: `-Reinstall` (rebuild venv), `-NoBrowser` (skip auto-open).
- **Updated prompts**: system and user prompts were adjusted to the new capabilities (multi-speaker coding, optional teacher).

**Deutsch**
- **Dunkelmodus**: Obsidian-inspiriertes Dark-Theme, in der Sidebar umschaltbar.
- **Ollama-Cloud-Unterstützung**: zusätzlicher API-Client für Ollama neben OpenAI, Groq und Anthropic. Voraussetzung ist eine lokale Ollama-Installation; `*-cloud`-Modelle benötigen zusätzlich ein Ollama-Cloud-Abo. Ohne Cloud-Key nutzt die App automatisch das lokale Ollama auf `localhost`.
- **Qualitative Analyse der SuS-Sprechakte**: Das LLM kodiert nun auch Schüler:innen-Äußerungen, nicht mehr nur die Lehrkraft. Das Ausgabe-JSON enthält ein `Sprecher`-Feld (z. B. `Lehrperson`, `S01`, `S02`), die Ergebnisseite zeigt Statistiken pro Sprecher:in.
- **Analyse auch ohne Lehrkraft**: Die Angabe eines Lehrkraftnamens ist jetzt optional — die qualitative Analyse funktioniert auch dann, wenn keine Lehrkraft im Transkript vorkommt.
- **PowerShell-Startskript `run.ps1`**: legt automatisch ein `.venv` an, installiert Abhängigkeiten, startet die App und öffnet den Browser. Flags: `-Reinstall` (venv neu bauen), `-NoBrowser` (Browser nicht automatisch öffnen).
- **Angepasste Prompts**: System- und User-Prompts wurden an die neuen Möglichkeiten (Mehrsprecher-Kodierung, optionale Lehrkraft) angepasst.

## Installation

Clone the repository and install the Python dependencies listed in [requirements.txt](requirements.txt) (includes `ollama` and `anthropic` for the corresponding API clients).

**Windows (recommended):** run the bundled PowerShell launcher from the project root:
```
./run.ps1
```
It creates a local `.venv`, installs dependencies, starts the Shiny app on `http://127.0.0.1:8000` and opens the browser. Use `-Reinstall` to force-rebuild the venv or `-NoBrowser` to skip the automatic browser launch.

**Ollama Cloud:** the `*-cloud` models require a local [Ollama](https://ollama.com/) installation *and* an Ollama cloud subscription. If no cloud API key is configured, TalkTrace-AI automatically falls back to a local Ollama instance on `localhost`.

## Usage

Once the application is running, it will automatically open the interface in your webbrowser at http://localhost:8000.

## Interface
The process of TalkTrace-AI is organized into 2 steps/tabs: Analysis and Results. The app-sidebar gives you quick options control for the analysis, e.g. enabling/changing LLM analysis, store/restore Session, etc.

#### Analysis

Under the Analysis tab, you can provide general information like the group and identifiers of the group. Specifying the **name of the teaching person in the transcript** is recommended so that TalkTrace-AI can correctly identify the teacher and calculate teacher-specific metrics, but it is no longer mandatory — qualitative analysis also runs on transcripts without a teacher.

To run the analysis, at least a transcript is required, which may be uploaded via the Document Input panel. Transcripts need to follow the scheme of [noScribe](https://github.com/kaixxx/noScribe) for the parsing to work, i.e. 
```"S01: Utterance"
"S02: Utterance"
"S01: Utterance"
"S04: Utterance"
```
 ... and so on.

If both quantitative and qualitative analysis is needed, a codebook is required as well (see the [example file](images/Example%20Codebook.docx)). Qualitative codes are applied to **all speakers** — teacher *and* student utterances — so the codebook may target speech acts of students (SuS) as well.
After upload, the analysis is started via the Analyze button in the sidebar. When results are ready, TalkTrace-AI automatically switches to the results tab
**Note:** Token prediction in the sidebar provides only a very rough estimate of the minimal expected costs. It is based on the length of the provided transcript/codebook, the LLM input token costs and and estimate of 4 times the output tokens. Since LLMs may provide significantly longer answers (especially reasoning models), only a lower bound can be predicted. Actual token usage may be checked via the LLM providers metrics.  

#### Results
The Results section is organized into quantitative and qualitative analysis. Only the latter is performed by a LLM, quantitative results are calculated using pattern matching and basic mathematical operations.

Quantitative Results provides basic metrics and a visualization on the class participation and the distribution of conversation shares (both relative an absolute measures).

Qualitative Results provide the coding of the LLM based on the uploaded codebook. Each coded utterance carries a `Sprecher` label (e.g. `Lehrperson`, `S01`, `S02`), and results are broken down per speaker so that teacher contributions and individual student contributions can be inspected separately. Basic metrics and a visualization of the distribution of codes are highlighted above the textual display; sections without matching data show a "No data" placeholder.

### Options
The Options tab allows for configuration of app settings.
If an LLM is used for qualitative analysis, TalkTrace-AI needs an API-key to communicate with the LLM-backend, which can be added, changed or deleted in the _API configuration_ settings. Selection of the LLM-Client is possible as well, with **OpenAI, Groq, Anthropic and Ollama** as choices. For Ollama the app detects whether a cloud API key is configured and otherwise falls back to a local Ollama instance on `localhost`.

The sidebar additionally exposes a **dark-mode toggle** (Obsidian-inspired theme) and a **language switch (EN/DE)** via the globe icon.

The preconfigured list of LLM Models can be edited in the section on _Models for LLM Selection_, which will update the available choices in the sidebar in realtime. This makes it possible to add new models or to exercise control over the used LLMs.

Custom System and User Prompts for the LLM can be configured in the _Custom Prompts section_, to meet specific analysis requirements. In case of doubt, prompts can be reset to the app default.

In the _Additional Options Panel_ allows to change the default values for teacher name, group ID and class size.

The configuration is stored locally on the app folder and can be partially reset via the according reset button.      

## Privacy Note
TalkTrace-AI does not store transcripts or analysis results on any external server. All data needed for preparing and displaying an analysis are held in local memory in the browser during interaction with the tool. Since LLM-models are not hosted locally, the application backend communicates with external large language models during the qualitative coding step. When qualitative coding is enabled, the relevant parts of the transcript and the codebook are transmitted to the selected LLM provider via the configured API. Any server-side storage or logging of these data therefore depends on the data protection policies and technical settings of the chosen LLM service. Raw LLM output and session data can be stored locally for later reuse via the export and import session controls, and processed outputs can be downloaded as result reports. API keys are stored securely in the operating system’s encrypted password vault. This architecture supports institutions that prefer to keep teaching and research data under their own control and aligns with recommendations that AI-supported analytics should be designed to minimise unnecessary data retention on external services. 

## Credits
TalkTrace-AI is being developed by Jami Schorling (https://orcid.org/0009-0005-9007-2896) and Dennis Hauk (https://orcid.org/0000-0002-5779-2876) at the [Chair for Research on Teaching and Learning in Civic Education at Leipzig University](https://www.sozphil.uni-leipzig.de/institut-fuer-politikwissenschaft/arbeitsbereiche/professur-fuer-fachdidaktik-gemeinschaftskunde/team/prof-dr-dennis-hauk) in Germany. 

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes on github.

## License
This project is licensed under the CC BY-NC 4.0 License. See the LICENSE file for more details. Let's socialize software for the open-source democratic stack!


