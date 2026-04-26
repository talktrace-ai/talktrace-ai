## About

TalkTrace-AI is a FLOSS, platform independent webapp for evaluating the performance of teaching students during class room simulation, leveraging the power of Large Language Models (LLMs). It will provide both quantitative and qualitative reports of the verbal classroom performance and allows for customization of the analysis parameters. It was built Shiny for Python web application. It provides an interactive web interface for users to engage with data and visualizations. An API-Key for either OpenAI or groq is required to perform qualitative analysis.

## Installation
Install the TalkTrace package on your python 3 via

`pip install talktrace`
In some scenarios, you may need to run

`python3 -m pip install talktrace`

## Usage
To run the web application, from terminal simply run

`talktrace`

or

`python3 -m talktrace`

Once the application is running, it will automatically open the interface in your webbrowser at http://localhost:8000.

## Quickstart per OS

The repository ships with launch helpers that create a virtual environment, install dependencies, and start the app.

### Windows
Double-click `start.bat`, or run from a terminal:
```
start.bat
```

### macOS
```
chmod +x start.sh
./start.sh
```
No additional system dependencies are required — the native window uses the Cocoa/WebKit backend that ships with macOS.

### Linux
```
chmod +x start.sh
./start.sh
```
For a native desktop window, install the WebKit/GTK bindings (Debian/Ubuntu):
```
sudo apt install gir1.2-webkit2-4.1 python3-gi
```
Without those packages, the app automatically falls back to opening in your default browser.

**Linux limitations:**
- PDF report export is not available on Linux (relies on Microsoft Word). Export to DOCX instead.
- Without a system keyring (GNOME Keyring / KWallet via SecretService), API keys are kept only for the running session. The app installs `keyrings.alt` as a file-based fallback, but you can also start the keyring daemon (`gnome-keyring-daemon` or similar) for persistent storage.

### Common flags
- `--reinstall` — recreate the virtual environment from scratch
- `--nowindow` — start headless (no native window); use a browser to visit http://localhost:8000

## Credits
TalkTrace-AI is being developed by Jami Schorling (https://orcid.org/0009-0005-9007-2896) and Dennis Hauk (https://orcid.org/0000-0002-5779-2876) at the [Chair for Research on Teaching and Learning in Civic Education at Leipzig University](https://www.sozphil.uni-leipzig.de/institut-fuer-politikwissenschaft/arbeitsbereiche/professur-fuer-fachdidaktik-gemeinschaftskunde/team/prof-dr-dennis-hauk) in Germany. 

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes on github.

## License
This project is licensed under the CC BY-NC 4.0 License. See the LICENSE file for more details. Let's socialize software for the open-source democratic stack!


