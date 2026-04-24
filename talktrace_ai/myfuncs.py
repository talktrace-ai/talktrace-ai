from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import pandas as pd
import tempfile
import json
import re
import hashlib
from collections import OrderedDict
from groq import BadRequestError, AuthenticationError, RateLimitError, InternalServerError, APIError
from openai import OpenAI
from openai.types.chat import ChatCompletion
import anthropic as anthropic_sdk
from ollama import chat as ollama_chat, Client as OllamaClient, ResponseError as OllamaResponseError
import tempfile
from pyparsing import line
from .localization.translation import TRANSLATIONS
from .config.config_manager import ConfigManager


# Config-Singleton: ConfigManager wurde zuvor bei jedem translate()-Aufruf neu
# instanziert und las die JSON-Datei vom Disk. Einmal cachen spart I/O.
_config_singleton = None

def _get_config():
    global _config_singleton
    if _config_singleton is None:
        _config_singleton = ConfigManager()
    return _config_singleton


# Helper function to get translated text
def translate(section, key):
    return TRANSLATIONS[_get_config().get_localization()["current_language"]][section][key]


# Response-Cache: bei identischem (provider, model, system, user, transcript,
# codebook) liefern wir direkt die gespeicherte Antwort zurück. Spart komplette
# API-Calls z.B. beim Re-Run nach UI-Wechseln.
_RESPONSE_CACHE_MAX = 32
_response_cache: "OrderedDict[str, str]" = OrderedDict()


def _cache_key(provider, model, system_prompt, user_prompt, transcript, codebook, extra=""):
    h = hashlib.md5()
    for part in (provider, model, system_prompt, user_prompt, str(transcript),
                 _format_codebook(codebook), extra):
        h.update(part.encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_get(key):
    if key in _response_cache:
        _response_cache.move_to_end(key)
        return _response_cache[key]
    return None


def _cache_put(key, value):
    # Fehlerantworten nicht cachen.
    if not value or '"error":' in value:
        return
    # Leere Codierungen nicht cachen — sonst bleiben fehlerhafte
    # API-Calls hängen und jeder Retry liefert dasselbe leere Ergebnis.
    if '"analysis": []' in value or '"analysis":[]' in value:
        return
    _response_cache[key] = value
    _response_cache.move_to_end(key)
    while len(_response_cache) > _RESPONSE_CACHE_MAX:
        _response_cache.popitem(last=False)


# Client-Cache: je API-Key wird nur ein SDK-Client instanziert.
_client_cache = {}


def get_groq_client(api_key):
    from groq import Groq
    key = ("groq", api_key)
    if key not in _client_cache:
        _client_cache[key] = Groq(api_key=api_key)
    return _client_cache[key]


def get_openai_client(api_key):
    key = ("openai", api_key)
    if key not in _client_cache:
        _client_cache[key] = OpenAI(api_key=api_key)
    return _client_cache[key]


def get_anthropic_client(api_key):
    key = ("anthropic", api_key)
    if key not in _client_cache:
        _client_cache[key] = anthropic_sdk.Anthropic(api_key=api_key)
    return _client_cache[key]


# Transkript-Chunking: bei sehr langen Transkripten an Sprecherwechseln teilen.
# Default-Schwelle ist bewusst hoch, damit normale Analysen unverändert bleiben.
CHUNK_CHAR_THRESHOLD = 120_000


def _chunk_transcript_by_turns(transcript, max_chars=60_000):
    """Teilt ein Transkript an Sprecherzeilen-Grenzen in Chunks <= max_chars."""
    if not transcript or len(transcript) <= max_chars:
        return [transcript]
    turn_re = re.compile(r"(?=^\s*(?:[A-Za-zÄÖÜäöüß_]+|S\d{2})\s*:)", re.MULTILINE)
    indices = [m.start() for m in turn_re.finditer(transcript)]
    if not indices or indices[0] != 0:
        indices = [0] + indices
    indices.append(len(transcript))

    chunks, buf_start, buf_end = [], indices[0], indices[0]
    for nxt in indices[1:]:
        if nxt - buf_start > max_chars and buf_end > buf_start:
            chunks.append(transcript[buf_start:buf_end])
            buf_start = buf_end
        buf_end = nxt
    if buf_end > buf_start:
        chunks.append(transcript[buf_start:buf_end])
    return chunks or [transcript]



def docx_to_json(docx_file_path):
    doc = Document(docx_file_path)

    if not doc.tables:
        text = []

        for para in doc.paragraphs:
            cleaned = para.text.strip()
            if cleaned:  # Skip empty lines
                text.append(cleaned)

        return "\n".join(text)

    table_data = []
    table = doc.tables[0]
    headers = [table.cell(0, i).text.strip() for i in range(len(table.columns))]

    for ri, row in enumerate(table.rows):
        # Row 0 is the header row; it would otherwise emit a useless
        # {"Code": "Code", "Bezeichnung": "Bezeichnung", ...} item.
        if ri == 0:
            continue
        cell_texts = [cell.text.strip() for cell in row.cells]
        row_data = {headers[i]: cell_texts[i] for i in range(len(headers))}
        table_data.append(row_data)

    return table_data


def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        txt = file.read()
    return txt


def import_file(file_dict):
    if file_dict['type'] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx_to_json(file_dict['datapath'])
    elif file_dict['type'] == "text/plain":
        return read_txt(file_dict['datapath'])
    else:
        return None


def count_pupils(transcript):
  # Regex für Sprecher
  sprecher_pattern = r'\b(' + re.escape(translate("analysis", "name_teacher_var")) + r'|S\d{2})\b(?=:)'

  # Alle Sprecher finden
  sprecher_liste = re.findall(sprecher_pattern, transcript)

  # Einzigartige Sprecher
  einzigartige_sprecher = set(sprecher_liste)

  # Sprecher ohne Lehrer (alle S\d{2} gelten als Schüler:innen)
  sprecher_ohne_lehrer = {s for s in einzigartige_sprecher if s != translate("analysis", "name_teacher_var")}

  # Every distinct S\d{2} speaker counts as a student, regardless of whether
  # a teacher label is present. This supports pure student-dialog transcripts.
  return len(sprecher_ohne_lehrer)


def dialog_stats_per_speaker(transcript, lehrperson):
    """Per-speaker stats WITHOUT aggregating students.

    Returns a DataFrame with one row per distinct speaker label
    (teacher + each S## student), columns: Sprecher, Anzahl_Beitraege,
    Gesamt_Woerter, Durchschnitt_Woerter, Median_Woerter.
    """
    text_split = re.sub(r"//(.*?)//", r"\n\1\n", transcript, flags=re.DOTALL)
    beitrag_pattern = re.compile(rf"\b({lehrperson}|S\d{{2}})\b:\s*(.*)")
    beitraege = beitrag_pattern.findall(text_split)
    df = pd.DataFrame(beitraege, columns=["Sprecher", "Beitrag"])
    df['Wortanzahl'] = df['Beitrag'].str.split().apply(len)
    df_summary = df.groupby('Sprecher').agg(
        Anzahl_Beitraege=('Beitrag', 'count'),
        Gesamt_Woerter=('Wortanzahl', 'sum'),
        Durchschnitt_Woerter=('Wortanzahl', 'mean'),
        Median_Woerter=('Wortanzahl', 'median')
    ).reset_index()
    return df_summary


def dialog_stats(transcript, lehrperson):
 # beitrag_pattern = r'(?:^|(?<=\s)|(?<=//))\b(' + lehrperson + r'|S\d{2})\b:\s*(.*)'
 # beitrag_pattern = r'\b(' + lehrperson + r'|S\d{2})\b:\s*(.*)'
 # beitrag_pattern = r'(?://\s*)?(?:\b(' + lehrperson + r'|S\d{2})\b):\s*(.*)'
 # beitrag_pattern = r'(?:^|//)?\b(' + lehrperson + r'|S\d{2})\b:\s*(.*)'
 # beitrag_pattern = rf'(?:^|//\s*|\s+){lehrperson}|S\d{2})\b:\s*(.*)'

# Extrahieren der Beiträge
  # 1. Einschübe aufsplitten
  text_split = re.sub(r"//(.*?)//", r"\n\1\n", transcript, flags=re.DOTALL)
  # 2. Regex für Beiträge
  beitrag_pattern = re.compile(rf"\b({lehrperson}|S\d{{2}})\b:\s*(.*)")
  beitraege = beitrag_pattern.findall(text_split)

  # DataFrame bauen
  df = pd.DataFrame(beitraege, columns=["Sprecher", "Beitrag"])

  # Wortanzahl je Beitrag berechnen
  df['Wortanzahl'] = df['Beitrag'].str.split().apply(len)

  # Zusammenfassung
  df_summary = df.groupby('Sprecher').agg(
      Anzahl_Beitraege=('Beitrag', 'count'),
      Gesamt_Woerter=('Wortanzahl', 'sum'),
      Durchschnitt_Woerter=('Wortanzahl', 'mean'),
      Median_Woerter=('Wortanzahl', 'median')
  ).reset_index()

    # 🟣 Schüler vs. Lehrer trennen
  df_lehrer = df_summary[df_summary['Sprecher'] == lehrperson]
  df_schueler = df_summary[df_summary['Sprecher'] != lehrperson]

  # Schüler zusammenfassen:
  schueler_summary = pd.DataFrame({
      'Sprecher': ['Schüler:innen'],
      'Anzahl_Beitraege': [df_schueler['Anzahl_Beitraege'].sum()],
      'Gesamt_Woerter': [df_schueler['Gesamt_Woerter'].sum()],
      'Durchschnitt_Woerter': [df_schueler['Gesamt_Woerter'].sum() / df_schueler['Anzahl_Beitraege'].sum() if df_schueler['Anzahl_Beitraege'].sum() > 0 else 0],
      'Median_Woerter': [df_schueler['Median_Woerter'].median()]
  })

  # Neu zusammenfügen:
  df_summary_neu = pd.concat([df_lehrer, schueler_summary], ignore_index=True)
  return df_summary_neu





def llm_analysis_groq(system_prompt, user_prompt, model, transcript, codebook, client):
    cache_key = _cache_key("groq", model, system_prompt, user_prompt, transcript, codebook)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        # Create chat completion object with JSON response format
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt.replace("{transcript}", str(transcript)).replace("{codebook}", _format_codebook(codebook)),
                }
            ],
            model=model,
            response_format={"type": "json_object"},
            max_tokens=12000,
        )

        analysis_json_string = chat_completion.choices[0].message.content
        _cache_put(cache_key, analysis_json_string)
        return analysis_json_string
    
    except BadRequestError as e:
        print(f"[ERROR]BadRequestError (400): {str(e)}")
        return json.dumps({"error": "Bad request - Failed to generate JSON."})

    except AuthenticationError as e:
        print(f"[ERROR]AuthenticationError (403): {str(e)}")
        return json.dumps({"error": "Authentication failed - Check API key or access rights."})

    except RateLimitError as e:
        print(f"[ERROR]RateLimitError (429): {str(e)}")
        return json.dumps({"error": "Rate limit exceeded - Too many requests."})

    except InternalServerError as e:
        print(f"[ERROR]InternalServerError (500+): {str(e)}")
        return json.dumps({"error": "Server error - Please try again later."})

    except APIError as e:
        print(f"[ERROR]APIError: {str(e)}")
        return json.dumps({"error": f"API error: {str(e)}"})

    except Exception as e:
        print(f"[ERROR]Unexpected error: {str(e)}")
        return json.dumps({"error": f"Unexpected error: {str(e)}"})
    

def llm_analysis_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    transcript,
    codebook,
    client: OpenAI
) -> str:
    cache_key = _cache_key("openai", model, system_prompt, user_prompt, transcript, codebook)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:

        # Define schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "#": {"type": "integer", "description": "Nummerierung"},
                            "Sprecher": {"type": "string", "description": "Sprecher-Kennung (z.B. 'Lehrperson', 'LEHRER', 'S01', ...)"},
                            "Shortcode": {"type": "string", "description": "Der Shortcode"},
                            "Impuls": {"type": "string", "description": "Die Äußerung"}
                        },
                        "required": ["#", "Sprecher", "Shortcode", "Impuls"],
                        "additionalProperties": False
                    },
                    "description": "Liste von Analyseobjekten"
                }
            },
            "required": ["analysis"],
            "additionalProperties": False
        }


        # Make the API call with structured output
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.replace("{transcript}", str(transcript)).replace("{codebook}", _format_codebook(codebook))}
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "analysis",
                    "schema": schema,
                    "strict": True
                }
            }
        )

        _cache_put(cache_key, response.output_text)
        return response.output_text

    except Exception as e:
        print(f"[ERROR] OpenAI API error: {e}")
        return json.dumps({"error": str(e)})


def llm_analysis_anthropic(system_prompt, user_prompt, model, transcript, codebook, client, progress_cb=None):
    """Anthropic implementation using forced tool_use for structured output.

    Forced tool_use is Anthropic's recommended pattern for structured output —
    equivalent to OpenAI's `response_format=json_schema`. The model is required
    to call our `submit_analysis` tool, and we extract the typed input. This
    bypasses the prose/markdown/refusal issues smarter models (Sonnet 4.5,
    Opus 4.6) exhibit when asked to "just output JSON" via prompt instructions.

    Nutzt Prompt-Caching: System-Prompt + Codebook-Block werden mit
    cache_control markiert, sodass bei wiederholten Analysen mit gleichem
    Codebook nur das (wechselnde) Transkript neu berechnet wird.
    """
    cache_key = _cache_key("anthropic", model, system_prompt, user_prompt, transcript, codebook)
    cached = _cache_get(cache_key)
    if cached is not None:
        print(f"[ANTHROPIC DEBUG] cache=HIT key={cache_key[:8]} — returning cached result")
        return cached

    stop_reason = None
    try:
        # Split: alles AUSSER dem Transkript gehört in den cache-fähigen Block
        # (Codebook + Intro + Schluss-Instruktion). Das Transkript ist der einzige
        # volatile Teil. So profitiert jede Wiederholungsanalyse vom Prompt-Cache.
        codebook_str = _format_codebook(codebook)
        with_codebook = user_prompt.replace("{codebook}", codebook_str)
        if "{transcript}" in with_codebook:
            before, after = with_codebook.split("{transcript}", 1)
            intro_text = before
            trailing_text = after
        else:
            intro_text = with_codebook
            trailing_text = ""

        # Tool-Instruktion VORNE: das Modell sieht zuerst, dass per Tool geantwortet
        # wird, nicht per JSON im Text — das entschärft den Konflikt mit der
        # "als JSON ausgeben"-Zeile im user_prompt-Template.
        tool_instruction = (
            "WICHTIG: Antworte ausschließlich durch Aufruf des Tools 'submit_analysis'. "
            "Übergib darin das vollständige Codierungs-Array. Codiere JEDE Äußerung im Transkript, "
            "die zu einem Code aus dem Codebuch passt — sowohl Äußerungen der Lehrperson als "
            "auch der Schüler:innen. Ordne im Zweifelsfall den bestpassenden Code zu; sei nicht "
            "überkritisch. Ein leeres Array ist fast immer falsch.\n\n"
        )

        stable_block = tool_instruction + intro_text + trailing_text
        volatile_block = str(transcript)

        user_content = [
            {"type": "text", "text": stable_block, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": volatile_block},
        ]

        print(
            f"[ANTHROPIC DEBUG] cache=miss sizes: transcript={len(volatile_block)} "
            f"codebook={len(codebook_str)} system={len(system_prompt)} "
            f"stable={len(stable_block)} model={model}"
        )

        # Output-Cap pro Modell. Deutlich reduziert gegenüber früher (32k/16k),
        # da tool_use-Payloads praktisch nie so groß werden und hohe max_tokens
        # bei einigen Modellen die Latenz erhöhen.
        if "opus" in model.lower() or "sonnet" in model.lower():
            max_tok = 16000
        else:
            max_tok = 8000

        # Define the structured-output tool. The model MUST call this tool.
        # Description emphasises native array/integer types — Sonnet 4.x tends
        # to stringify nested arrays under forced tool_use otherwise.
        analysis_tool = {
            "name": "submit_analysis",
            "description": (
                "Submit the qualitative coding analysis of the classroom transcript. "
                "The 'analysis' argument MUST be a native JSON array of objects — "
                "NOT a JSON-encoded string. Each '#' field MUST be a native integer — "
                "not a string. Include one object per coded utterance from any speaker "
                "(teacher and students)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "analysis": {
                        "type": "array",
                        "description": "Native JSON array (NOT a string) of coded utterances. One object per codable utterance in the transcript.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "#": {"type": "integer", "description": "Sequential index starting at 1."},
                                "Sprecher": {"type": "string", "description": "Speaker label (e.g. 'Lehrperson', 'LEHRER', 'S01', 'S02')."},
                                "Shortcode": {"type": "string", "description": "The matching shortcode from the codebook."},
                                "Impuls": {"type": "string", "description": "The verbatim utterance text."},
                            },
                            "required": ["#", "Sprecher", "Shortcode", "Impuls"],
                        },
                    },
                },
                "required": ["analysis"],
            },
        }

        # Streaming + forced tool_use. tool_choice forces the model to call our tool.
        final_msg = None
        system_blocks = [
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
        ]
        with client.messages.stream(
            model=model,
            max_tokens=max_tok,
            system=system_blocks,
            tools=[analysis_tool],
            tool_choice={"type": "tool", "name": "submit_analysis"},
            messages=[
                {"role": "user", "content": user_content},
            ],
        ) as stream:
            # Drain the stream so the SDK collects the full message.
            chunks_seen = 0
            for _ in stream:
                chunks_seen += 1
                if progress_cb and chunks_seen % 20 == 0:
                    try:
                        progress_cb(chunks_seen)
                    except Exception:
                        pass
            final_msg = stream.get_final_message()

        stop_reason = getattr(final_msg, "stop_reason", None)

        # Extract the tool_use input from the response content blocks.
        tool_input = None
        block_types = []
        for block in final_msg.content:
            btype = getattr(block, "type", "")
            block_types.append(btype)
            if btype == "tool_use" and getattr(block, "name", "") == "submit_analysis":
                tool_input = getattr(block, "input", None)
                break

        # Typ des analysis-Feldes inspizieren — manche SDK-Versionen liefern
        # bei Streaming mit forced tool_use den partial_json als String statt
        # als geparstes Dict/List zurück. Ohne Recovery landet das im UI als
        # "0 coded items", weil app.py non-list-analysis auf [] zurücksetzt.
        analysis_field = None
        if isinstance(tool_input, dict):
            analysis_field = tool_input.get("analysis")
        analysis_type = type(analysis_field).__name__ if analysis_field is not None else "None"
        analysis_len = len(analysis_field) if hasattr(analysis_field, "__len__") else -1
        usage = getattr(final_msg, "usage", None)
        print(
            f"[ANTHROPIC DEBUG] model={model} stop_reason={stop_reason} blocks={block_types} "
            f"analysis_type={analysis_type} analysis_len={analysis_len} usage={usage}"
        )

        # Recovery: analysis ist ein String — versuche, ihn als JSON-Array zu parsen.
        # Claude 4.x Sonnet/Opus stringifiziert bei forced tool_use mit nested
        # array-schemas gelegentlich den Array-Wert. Wir fangen das progressiv ab.
        if isinstance(analysis_field, str):
            sample = analysis_field[:200].replace("\n", " ")
            print(f"[ANTHROPIC DEBUG] analysis is STRING, first 200 chars: {sample!r}")
            parsed_items = None
            # Versuch 1: strict JSON parse
            try:
                cand = json.loads(analysis_field)
                if isinstance(cand, list):
                    parsed_items = cand
                    print(f"[ANTHROPIC DEBUG] strict parse OK — {len(cand)} items")
            except (json.JSONDecodeError, ValueError) as e:
                # Versuch 2: lenient parse (erlaubt rohe Control-Chars in Strings).
                try:
                    cand = json.loads(analysis_field, strict=False)
                    if isinstance(cand, list):
                        parsed_items = cand
                        print(f"[ANTHROPIC DEBUG] lenient (strict=False) parse OK — {len(cand)} items")
                except (json.JSONDecodeError, ValueError) as e2:
                    # Kontext um die Fehlerstelle loggen (hilft bei Diagnose).
                    pos = getattr(e2, "pos", None) or getattr(e, "pos", 0)
                    around = analysis_field[max(0, pos - 40):pos + 40]
                    print(f"[ANTHROPIC DEBUG] parse errors — strict: {e} / lenient: {e2}")
                    print(f"[ANTHROPIC DEBUG] context around pos {pos}: {around!r}")
                    # Versuch 3: Items einzeln via Brace-Walking extrahieren.
                    prog = _extract_items_progressive(analysis_field)
                    # Versuch 4: Feldbasierte Extraktion — robust gegen
                    # unescapte ASCII-Quotes in Impuls-Texten (Brace-Walking
                    # geht nach dem ersten kaputten Item aus dem Tritt).
                    schema = _extract_items_by_schema(analysis_field)
                    print(f"[ANTHROPIC DEBUG] progressive={len(prog)} schema={len(schema)}")
                    # Das Verfahren mit mehr Items wählen.
                    parsed_items = schema if len(schema) >= len(prog) else prog

            if parsed_items is not None and len(parsed_items) > 0:
                tool_input["analysis"] = parsed_items

        # Empty-Dump wie vorher, nachdem evtl. Recovery gelaufen ist.
        if isinstance(tool_input, dict):
            final_analysis = tool_input.get("analysis")
            final_n = len(final_analysis) if isinstance(final_analysis, list) else -1
            if final_n == 0:
                dump = json.dumps(tool_input, ensure_ascii=False)[:500]
                print(f"[ANTHROPIC DEBUG] empty tool_input first 500: {dump}")

        if isinstance(tool_input, dict) and isinstance(tool_input.get("analysis"), list):
            result = json.dumps(tool_input, ensure_ascii=False)
            _cache_put(cache_key, result)
            return result

        # Fallback: model didn't use the tool (rare under forced tool_choice).
        # Pull any text blocks and try to parse them as JSON.
        text_fallback = "".join(
            getattr(b, "text", "") for b in final_msg.content if getattr(b, "type", "") == "text"
        )
        print(f"[ANTHROPIC DEBUG] no tool_use block; text fallback first 500: {text_fallback[:500]!r}")
        extracted = _extract_json(text_fallback) if text_fallback else None
        if extracted:
            return extracted

        return json.dumps({
            "error": f"Anthropic did not return a tool_use call (stop_reason={stop_reason}, blocks={block_types})."
        })

    except anthropic_sdk.AuthenticationError as e:
        print(f"[ERROR]AuthenticationError: {str(e)}")
        return json.dumps({"error": "Authentication failed - Check API key or access rights."})

    except anthropic_sdk.RateLimitError as e:
        print(f"[ERROR]RateLimitError: {str(e)}")
        return json.dumps({"error": "Rate limit exceeded - Too many requests."})

    except anthropic_sdk.BadRequestError as e:
        print(f"[ERROR]BadRequestError: {str(e)}")
        return json.dumps({"error": f"Bad request: {str(e)}"})

    except anthropic_sdk.APIError as e:
        print(f"[ERROR]APIError: {str(e)}")
        return json.dumps({"error": f"API error: {str(e)}"})

    except Exception as e:
        print(f"[ERROR]Unexpected error: {str(e)}")
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def _extract_json(text):
    """Try to extract a valid JSON object or array from text that may contain markdown fences, prose, or extra content."""
    if not text:
        return None
    # Fast path: strip markdown code fences and try parsing as-is.
    stripped = re.sub(r'```(?:json)?\s*', '', text).strip()
    stripped = re.sub(r'```\s*$', '', stripped).strip()
    try:
        json.loads(stripped)
        return stripped
    except (json.JSONDecodeError, ValueError):
        pass

    # Brace-balanced scan: find each candidate '{' or '[' and try to parse
    # the substring from there to its matching close. This is resilient to
    # prose prefixes, markdown fences, trailing commentary, and multi-JSON
    # artifacts like "{```json\n{...}" that a naive regex mishandles.
    def _balanced_parse(s, open_ch, close_ch):
        for start in range(len(s)):
            if s[start] != open_ch:
                continue
            depth = 0
            in_str = False
            escape = False
            for i in range(start, len(s)):
                c = s[i]
                if in_str:
                    if escape:
                        escape = False
                    elif c == '\\':
                        escape = True
                    elif c == '"':
                        in_str = False
                    continue
                if c == '"':
                    in_str = True
                elif c == open_ch:
                    depth += 1
                elif c == close_ch:
                    depth -= 1
                    if depth == 0:
                        candidate = s[start:i + 1]
                        try:
                            parsed = json.loads(candidate)
                            # Prefer candidates that actually look like our schema.
                            if isinstance(parsed, dict) and "analysis" in parsed:
                                return candidate
                            if isinstance(parsed, list):
                                return candidate
                            # Keep looking for a better candidate; fall through.
                            return candidate
                        except (json.JSONDecodeError, ValueError):
                            break  # unbalanced / malformed; try next start
        return None

    result = _balanced_parse(text, '{', '}')
    if result:
        return result
    result = _balanced_parse(text, '[', ']')
    if result:
        # Wrap bare array in {"analysis": [...]}
        try:
            arr = json.loads(result)
            if isinstance(arr, list):
                return json.dumps({"analysis": arr})
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _format_codebook(codebook):
    """Format codebook for LLM consumption. Converts list of dicts to readable JSON instead of Python repr."""
    if isinstance(codebook, list):
        return json.dumps(codebook, ensure_ascii=False, indent=2)
    return str(codebook)


def _extract_items_by_schema(text):
    """Field-aware recovery. Matches the fixed {#, Sprecher, Shortcode, Impuls}
    schema via regex and extracts each item. Robust against unescaped ASCII
    quotes inside Impuls text (a common Claude 4.x stringification artifact) —
    those break generic JSON parsers but not field-pattern matching.
    """
    items = []
    # Match item prefix up to the Impuls value opening quote. Everything before
    # Impuls is short/simple enough that inner quotes are unlikely.
    prefix_re = re.compile(
        r'\{\s*'
        r'"#"\s*:\s*(\d+)\s*,\s*'
        r'"Sprecher"\s*:\s*"([^"]*)"\s*,\s*'
        r'"Shortcode"\s*:\s*"([^"]*)"\s*,\s*'
        r'"Impuls"\s*:\s*"',
        re.DOTALL,
    )
    # Terminator: a quote followed by optional whitespace + closing brace,
    # followed by comma, closing bracket, or end-of-string. This is the only
    # reliable way to find the real end of Impuls when it may contain ".
    term_re = re.compile(r'"\s*\}\s*(?=,|\]|$)')

    pos = 0
    while pos < len(text):
        m = prefix_re.search(text, pos)
        if not m:
            break
        num_str, sprecher, shortcode = m.group(1), m.group(2), m.group(3)
        impuls_start = m.end()
        t = term_re.search(text, impuls_start)
        if not t:
            # last item without trailing comma — try relaxed terminator.
            t2 = re.search(r'"\s*\}\s*$', text[impuls_start:])
            if t2:
                impuls_end = impuls_start + t2.start()
                next_pos = len(text)
            else:
                break
        else:
            impuls_end = t.start()
            next_pos = t.end()
        impuls = text[impuls_start:impuls_end]
        try:
            items.append({
                "#": int(num_str),
                "Sprecher": sprecher,
                "Shortcode": shortcode,
                "Impuls": impuls,
            })
        except ValueError:
            pass
        pos = next_pos
    return items


def _extract_items_progressive(text):
    """Walk a JSON-array-like string and extract each top-level {...} block
    as a parsed dict. Skips items that fail to parse (e.g. unescaped quote
    inside an Impuls string) — robust recovery when the whole-array parse
    fails partway through. Tries strict JSON first, then strict=False.
    """
    items = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    i = 0
    while i < len(text):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
        elif c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start:i + 1]
                parsed = None
                for kwargs in ({}, {"strict": False}):
                    try:
                        parsed = json.loads(candidate, **kwargs)
                        break
                    except (json.JSONDecodeError, ValueError):
                        continue
                if isinstance(parsed, dict):
                    items.append(parsed)
                start = -1
        i += 1
    return items


def _repair_truncated_analysis(text):
    """Salvage a truncated JSON response of the form {"analysis": [...]}.

    When the model hits its output cap mid-array, the tail looks like:
        ..., {"#": 7, "Sprecher": "S01", "Shortcode": "EF", "Imp
    We walk back to the last complete '}' that closes an item, drop
    the dangling partial, and close the array + object. Returns the
    repaired JSON string, or None if repair isn't possible.
    """
    if not text:
        return None
    # Find the opening of the analysis array.
    arr_start = text.find('"analysis"')
    if arr_start < 0:
        return None
    bracket_start = text.find('[', arr_start)
    if bracket_start < 0:
        return None
    # Find the outer '{' that opens the object containing "analysis".
    # Walk backwards from arr_start to skip any markdown fence / prose prefix.
    obj_start = text.rfind('{', 0, arr_start)
    if obj_start < 0:
        return None
    # Walk the array, tracking brace depth, to find the last complete item.
    depth = 0
    last_complete = -1  # index just after the most recent fully-closed item
    in_str = False
    escape = False
    for i in range(bracket_start + 1, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                last_complete = i + 1
        elif c == ']' and depth == 0:
            # Array closed cleanly; let the normal parser handle it.
            return None
    if last_complete < 0:
        # Nothing complete — return {"analysis": []}.
        return json.dumps({"analysis": []})
    # Build: text[obj_start..last_complete] + ']' + '}' (closes array + outer object).
    # Slicing from obj_start strips any leading prose/markdown fence.
    repaired = text[obj_start:last_complete] + "]}"
    try:
        json.loads(repaired)
        return repaired
    except (json.JSONDecodeError, ValueError):
        return None


def llm_analysis_ollama(system_prompt, user_prompt, model, transcript, codebook, api_key=None):
    try:
        # Strong structural guidance so the model returns the exact shape we expect.
        # kimi-k2.5:cloud (and other Ollama models) have no strict schema support
        # like OpenAI, so we spell out the wrapper explicitly to avoid bare arrays
        # or empty responses.
        structure_hint = (
            "\n\nWICHTIG: Antworte AUSSCHLIESSLICH mit einem JSON-Objekt im folgenden Format "
            "(keine Markdown-Fences, kein zusätzlicher Text):\n"
            '{\n  "analysis": [\n    {"#": 1, "Sprecher": "...", "Shortcode": "...", "Impuls": "..."},\n'
            '    {"#": 2, "Sprecher": "...", "Shortcode": "...", "Impuls": "..."}\n  ]\n}\n'
            "Füge für JEDE codierbare Äußerung ALLER Sprecher:innen im Transkript einen Eintrag hinzu "
            "(Lehrperson UND Schüler:innen). Das Feld 'Sprecher' enthält die Kennung aus dem Transkript "
            "(z.B. 'Lehrperson', 'LEHRER', 'S01', 'S02', ...). "
            "Gib dir Mühe, tatsächlich Codes zuzuweisen — ein leeres Array ist fast immer falsch, "
            "weil reale Unterrichtsgespräche praktisch immer codierbare Beiträge enthalten. "
            "Ordne im Zweifelsfall den bestpassenden Code zu."
        )
        # Ankerbeispiele referenzieren oft ein anderes Nummerierungsschema
        # (z.B. T1-T21) als das aktuelle Transkript (S01, Lehrperson, ...).
        # Strikte JSON-Modelle wie kimi-k2:1t-cloud lesen die Ankerbeispiele
        # dann als Filter und geben ein leeres Analyse-Array zurück. Wir
        # entfernen die Spalte für die Ollama-Pipeline.
        codebook_for_ollama = codebook
        if isinstance(codebook, list):
            codebook_for_ollama = [
                {k: v for k, v in entry.items() if k != "Ankerbeispiel"}
                if isinstance(entry, dict) else entry
                for entry in codebook
            ]
        rendered_user = user_prompt.replace("{transcript}", str(transcript)).replace("{codebook}", _format_codebook(codebook_for_ollama))
        messages = [
            {"role": "system", "content": system_prompt + structure_hint},
            {"role": "user", "content": rendered_user + structure_hint},
        ]
        # Stream the response. Streaming keeps the TCP connection active,
        # so Cloudflare's 100s idle-timeout (the source of 524 errors for
        # slow models like kimi-k2.5:cloud coding every turn) no longer
        # fires. We accumulate the chunks into the final JSON string.
        # Generous output budget so the model doesn't truncate mid-JSON when
        # coding every turn of a long transcript.
        options = {"num_predict": 65536, "num_ctx": 131072, "temperature": 0.2}

        # NOTE: do NOT pass `format="json"` here. Ollama's constrained-JSON
        # decoding mode causes trillion-param cloud models (e.g. kimi-k2:1t-cloud)
        # to collapse into the trivial `{"analysis": []}` output for any codebook
        # beyond the trivial case. We instead rely on the structure_hint in the
        # prompt plus the downstream _repair_truncated_analysis / _extract_json
        # path to extract JSON from free-form text. OpenAI still gets strict
        # schema enforcement via response_format=json_schema in its own function.
        # Local mode: always use local Ollama server at http://localhost:11434
        client = OllamaClient(host="http://localhost:11434")
        stream = client.chat(model=model, messages=messages, stream=True, options=options)

        content_parts = []
        done_reason = None
        for chunk in stream:
            # chunk is an ollama ChatResponse; .message.content holds the delta
            try:
                delta = chunk.message.content
            except AttributeError:
                delta = None
            if delta:
                content_parts.append(delta)
            # Capture why the stream ended (stop, length, etc.)
            try:
                if getattr(chunk, "done", False):
                    done_reason = getattr(chunk, "done_reason", None)
            except Exception:
                pass
        content = "".join(content_parts)

        if not content:
            return json.dumps({"error": "Ollama returned an empty response. Try a different model or retry."})

        # Strip markdown fences before parsing (cloud models often wrap JSON in ```json...```)
        cleaned = re.sub(r'```(?:json)?\s*', '', content).strip()
        cleaned = re.sub(r'```\s*$', '', cleaned).strip()

        # Try cleaned content first
        try:
            json.loads(cleaned)
            return cleaned
        except (json.JSONDecodeError, ValueError):
            pass

        # Try original content
        try:
            json.loads(content)
            return content
        except (json.JSONDecodeError, ValueError):
            pass

        # Try repairing a truncated JSON object (most common when done_reason
        # is "length" — the model hit num_predict mid-string). We walk back
        # to the last complete item of the "analysis" array and close braces.
        repaired = _repair_truncated_analysis(cleaned or content)
        if repaired:
            return repaired

        # Last resort: extract JSON object/array from the text
        extracted = _extract_json(content)
        if extracted:
            return extracted

        # Surface a bit of context so the user can see WHY parsing failed.
        preview = content[:160].replace("\n", " ")
        suffix = f" (done_reason={done_reason})" if done_reason else ""
        return json.dumps({"error": f"Failed to parse JSON from Ollama{suffix}. First 160 chars: {preview}"})

    except OllamaResponseError as e:
        msg = str(e)
        # Cloudflare 524: upstream (Ollama cloud) took too long. Provide a
        # concise, user-facing message instead of the full HTML error page.
        if "524" in msg or "timeout occurred" in msg.lower():
            return json.dumps({"error": "Ollama cloud timeout (524). The model took too long to respond. Try a smaller transcript, a different model, or retry in a few minutes."})
        return json.dumps({"error": f"Ollama error: {msg}"})

    except ConnectionError as e:
        return json.dumps({"error": "Cannot connect to Ollama. Make sure Ollama is running (local) or your API key is valid (cloud)."})

    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


def count_teacher_impulses(df, teacher_name):
  matches = df.loc[df['Sprecher'] == teacher_name, 'Anzahl_Beitraege']
  if matches.empty:
      return 0
  return matches.values[0]

def remove_table_borders(table):
    tbl = table._tbl  # Access the XML element
    tblPr = tbl.tblPr

    tblBorders = tblPr.xpath('./w:tblBorders')
    if tblBorders:
        tblPr.remove(tblBorders[0])  # Remove existing borders

    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        edge_el = OxmlElement(f'w:{edge}')
        edge_el.set(qn('w:val'), 'nil')  # 'nil' removes the line
        borders.append(edge_el)

    tblPr.append(borders)

def set_row_borders(row, top=False, bottom=False, left=False, right=False, size=12, color="000000", space="0"):
    for cell in row:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        tcBorders = tcPr.find(qn('w:tcBorders'))
        if tcBorders is None:
            tcBorders = OxmlElement('w:tcBorders')
            tcPr.append(tcBorders)

        def set_border(side):
            side_el = tcBorders.find(qn(f'w:{side}'))
            if side_el is None:
                side_el = OxmlElement(f'w:{side}')
                tcBorders.append(side_el)
            side_el.set(qn('w:val'), 'single')
            side_el.set(qn('w:sz'), str(size))       # border thickness
            side_el.set(qn('w:color'), color)        # hex color
            side_el.set(qn('w:space'), space)

        if top:
            set_border('top')
        if bottom:
            set_border('bottom')
        if left:
            set_border('left')
        if right:
            set_border('right')


def generate_report2(
    output_path: str, 
    group_name: str,
    num_pupils: int,
    num_participants: int,
    participation_rate: float,
    teacher_data: dict,
    student_data: dict,
    plot_distribution,  # matplotlib Figure
    num_impulses: int,
    caption: str = "",
    llm_analysis: bool = False,
    plot_impulse_coding = None, # matplotlib Figure
    impulse_table = None,
    model_name: str = "",
):

    # Neues Dokument
    doc = Document()

    # Schriften formatieren
    styles = doc.styles
    styles['Heading1'].element.rPr.rFonts.set(qn("w:asciiTheme"), "Aptos")
    styles['Heading1'].font.name = 'Aptos'
    styles['Heading1'].font.size = Pt(16)
    styles['Heading1'].font.bold = True
    styles['Heading1'].font.color.rgb = RGBColor(0, 0, 0)  # Schwarz
    styles['Heading1'].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    styles['Heading1'].paragraph_format.space_after = Pt(0)
    styles['Heading1'].paragraph_format.space_before = Pt(0)
    styles['Heading1'].paragraph_format.line_spacing = 1

    styles['Heading2'].font.name = 'Aptos'
    styles['Heading2'].font.size = Pt(12)
    styles['Heading2'].font.bold = True
    styles['Heading2'].font.color.rgb = RGBColor(0, 0, 0)  # Schwarz
    styles['Heading2'].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    styles['Heading2'].paragraph_format.space_after = Pt(0)
    styles['Heading2'].paragraph_format.space_before = Pt(0)
    styles['Heading2'].paragraph_format.line_spacing = 1

    styles['Normal'].font.name = 'Aptos'
    styles['Normal'].font.size = Pt(12)
    styles['Normal'].paragraph_format.space_after = Pt(0)
    styles['Normal'].paragraph_format.space_before = Pt(0)
    styles['Normal'].paragraph_format.line_spacing = 1

    # Ränder
    doc.sections[0].left_margin = Inches(0.5)
    doc.sections[0].right_margin = Inches(0.5)
    doc.sections[0].top_margin = Inches(0.5)
    doc.sections[0].bottom_margin = Inches(0.5)

    # === Titel ===
    doc.add_heading(f"{translate("report", "header")} {group_name}", level=1)

    # === Abschnitt: Quantitative Verteilung ===
    doc.add_heading(translate("report", "section_1"), level=2)
    doc.add_paragraph("").paragraph_format.line_spacing = 0.3
    doc.add_paragraph(f"{translate("report", "class_size")}: {num_pupils}\t\t{translate("report", "pupil_count")}: {num_participants} ({translate("report", "participation_rate")}: {participation_rate:.1f}%)")
    doc.add_paragraph("").paragraph_format.line_spacing = 0.5

    # === Tabelle: Gesprächsbeiträge ===
    par1 = doc.add_paragraph()
    par1.add_run(f"{translate("report", "table")}: ")
    par1.add_run(translate("report", "interaction_turns_teacher_pupils")).italic = True
    par1.paragraph_format.line_spacing = 1.2

    table = doc.add_table(rows=4, cols=6)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = translate("report", "participants")
    hdr_cells[1].text = translate("report", "teacher")
#    hdr_cells[2].text = ""
 #   hdr_cells[3].text = ""
    hdr_cells[4].text = translate("report", "pupils")
  #  hdr_cells[5].text = ""

    row1 = table.rows[1].cells
    row1[0].text = ""
    row1[1].text = translate("report", "quantity")
    row1[2].text = translate("report", "length_words")
    row1[3].text = ""
    row1[4].text = translate("report", "quantity")
    row1[5].text = translate("report", "length_words")

    row2 = table.rows[2].cells
    row2[0].text = ""
    row2[1].text = "N"
    row2[2].text = "M(SD)"
    row2[3].text = ""
    row2[4].text = "N"
    row2[5].text = "M(SD)"
    
    row3 = table.rows[3].cells
    row3[0].text = translate("report", "interaction_turns")
    row3[1].text = str(teacher_data["num"])
    row3[2].text = f"{str(teacher_data["words"])} ({str(teacher_data["mean_sd"])})"
    row3[3].text = ""
    row3[4].text = str(student_data["num"])
    row3[5].text = f"{str(student_data["words"])} ({str(student_data["mean_sd"])})"
    
    # Tabelle formatieren
    for cell in row2:
        cell.paragraphs[0].runs[0].italic = True   
    hdr_cells[1].merge(hdr_cells[2])
    hdr_cells[4].merge(hdr_cells[5])
    remove_table_borders(table)
    set_row_borders(hdr_cells, top=True, bottom=True)
    set_row_borders(row2[1:3], bottom=True)
    set_row_borders(row2[4:6], bottom=True)
    set_row_borders(row3, bottom=True)

    for row in table.rows:
        for cell in row.cells[2:6]:
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # === Abbildung: Gesprächsverteilung als Plot ===
    doc.add_paragraph("")
    doc.add_paragraph(f"{translate("report", "figure")}: ")
    doc.add_paragraph().add_run(translate("report", "distribution_of_turns")).italic = True
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plot_dis = plot_distribution
        plot_dis.figure.tight_layout()
        plot_dis.figure.set_size_inches(5.5, 2.9)
        plot_dis.figure.savefig(tmpfile.name, dpi=300, bbox_inches='tight')
        doc.add_picture(tmpfile.name)
    doc.add_paragraph("")

    # === Abschnitt: Qualitative Codierung ===
    if llm_analysis:
        doc.add_heading(translate("report", "section_2"), level=2)
        doc.add_paragraph("").paragraph_format.line_spacing = 0.3
        doc.add_paragraph(f"{translate("report", "impulses_count")}: N = {num_impulses}")
        par2 = doc.add_paragraph()
        par2.add_run(f"{translate("report", "figure")}: ")
        par2.add_run(translate("report", "teacher_impulses")).italic = True

        # === Abbildung: Qualitative Verteilung als Plot ===
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile2:
            plot_qual = plot_impulse_coding
            plot_qual.figure.tight_layout()
            plot_qual.figure.set_size_inches(5.5, 2.9)
            plot_qual.figure.savefig(tmpfile2.name, dpi=300, bbox_inches='tight')
            doc.add_picture(tmpfile2.name)
        doc.add_paragraph("")

        par3 = doc.add_paragraph()
        par3.add_run(f"{translate("report", "table")}: ")
        par3.add_run(translate("report", "teacher_impulses")).italic = True

        speaker_col = translate("report", "speaker")
        statement_col = translate("report", "teacher_statement")
        code_col = translate("report", "shortcode")
        has_speaker = speaker_col in impulse_table.columns

        ncols = 4 if has_speaker else 3
        t = doc.add_table(rows=1, cols=ncols)
        t.style = 'Table Grid'
        hdr = t.rows[0].cells
        hdr[0].text = "#"
        if has_speaker:
            hdr[1].text = speaker_col
            hdr[2].text = statement_col
            hdr[3].text = translate("report", "code")
        else:
            hdr[1].text = statement_col
            hdr[2].text = translate("report", "code")

        for i, row in impulse_table.iterrows():
            row_cells = t.add_row().cells
            row_cells[0].text = str(i + 1)
            if has_speaker:
                row_cells[1].text = str(row[speaker_col])
                row_cells[2].text = str(row[statement_col])
                row_cells[3].text = str(row[code_col])
            else:
                row_cells[1].text = str(row[statement_col])
                row_cells[2].text = str(row[code_col])

        # Tabellen-Header formatieren
        for cell in hdr:
            cell.paragraphs[0].runs[0].bold = True

        # Schriftgröße für die Tabelle anpassen
        for row in t.rows:
            for cell in row.cells:
                cell.paragraphs[0].runs[0].font.size = Pt(8)

        # Breite der Zellen anpassen
        for row in t.rows:
            if has_speaker:
                row.cells[0].width = Inches(0.3)
                row.cells[1].width = Inches(0.8)
                row.cells[2].width = Inches(6.0)
                row.cells[3].width = Inches(0.5)
            else:
                row.cells[0].width = Inches(0.3)
                row.cells[1].width = Inches(6.8)
                row.cells[2].width = Inches(0.5)

        # === Fußnote / Hinweis zu Codes ===
        doc.add_paragraph("")
        par4 = doc.add_paragraph()
        par4.add_run(f"{translate("report", "caption")}: ")
        par4.add_run(caption).italic = True

        if model_name:
            par5 = doc.add_paragraph()
            par5.add_run(f"{translate("report", "model_used")}: ")
            par5.add_run(model_name).italic = True

    # === Speichern ===
    doc.save(output_path)