from flask import Flask, jsonify, request, render_template
from difflib import get_close_matches
import os
import pandas as pd
from dotenv import load_dotenv
import openai
from flask import render_template

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('chat.html')

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")

# Lazily create an OpenAI client compatible with different installed SDK versions.
_OPENAI_CLIENT = None
_OPENAI_IS_MODERN = None

def get_openai_client():
    """Return a client object and a flag indicating whether it's the modern
    `OpenAI` client (True) or the legacy `openai` module (False).
    """
    global _OPENAI_CLIENT, _OPENAI_IS_MODERN
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT, _OPENAI_IS_MODERN

    try:
        # Try modern client style
        from openai import OpenAI as OpenAIClient
        client = OpenAIClient(api_key=OPENAI_API_KEY)
        _OPENAI_CLIENT = client
        _OPENAI_IS_MODERN = True
        return _OPENAI_CLIENT, _OPENAI_IS_MODERN
    except Exception:
        # Fallback to legacy module interface
        try:
            import openai as legacy
            if OPENAI_API_KEY:
                legacy.api_key = OPENAI_API_KEY
            _OPENAI_CLIENT = legacy
            _OPENAI_IS_MODERN = False
            return _OPENAI_CLIENT, _OPENAI_IS_MODERN
        except Exception:
            _OPENAI_CLIENT = None
            _OPENAI_IS_MODERN = None
            return None, None


def call_llm(prompt: str) -> str:
    """Call the installed OpenAI SDK with the provided prompt and return a text answer.
    Handles both the modern `OpenAI` client and the legacy `openai` module.
    """
    client, is_modern = get_openai_client()
    if client is None:
        return "Sorry, the AI backend is unavailable."

    try:
        if is_modern:
            resp = client.responses.create(model=OPENAI_MODEL, input=prompt, max_output_tokens=256)
            text = getattr(resp, "output_text", None)
            if text:
                return text.strip()
            if getattr(resp, "output", None) and len(resp.output) > 0:
                first = resp.output[0]
                if hasattr(first, "content") and first.content:
                    return first.content[0].text.strip()
            return "Sorry, I couldn't generate an answer."
        else:
            # Legacy package: use ChatCompletion if available
            if hasattr(client, "ChatCompletion"):
                resp = client.ChatCompletion.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                )
                return resp.choices[0].message.content
            # Fallback to completion or other interfaces
            if hasattr(client, "Completion"):
                resp = client.Completion.create(model=OPENAI_MODEL, prompt=prompt, max_tokens=256)
                return resp.choices[0].text
            return "Sorry, the AI backend does not support this operation."
    except Exception as e:
        return f"Sorry, the AI backend had an issue: {e}"


def ask_llm(user_q: str) -> str:
    llm_prompt = (
        "You are a groundwater assistant for India. "
        "When the user asks about Punjab, Haryana, or Rajasthan, prefer to use numbers from the provided context when available. "
        "If exact numbers for a district/state are not present in the context, explain that exact figures are unavailable and provide clear, practical guidance rather than inventing numbers. "
        "Keep explanations simple for non-technical users.\n\n"
        f"User question: {user_q}"
    )
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=llm_prompt,
            reasoning={"effort": "minimal"},
            max_output_tokens=256,
        )
        text = getattr(resp, "output_text", None)
        if text:
            return text.strip()
        if getattr(resp, "output", None) and len(resp.output) > 0:
            first = resp.output[0]
            if hasattr(first, "content") and first.content:
                return first.content[0].text.strip()
        return "Sorry, I could not generate an answer this time."
    except Exception as e:
        return f"Sorry, the AI backend had an issue: {e}"

app = Flask(__name__)


def summarize_rows(rows: pd.DataFrame) -> str:
    """Create a short, human-readable summary from CSV rows."""
    parts = []
    for _, r in rows.iterrows():
        d = r.to_dict()
        district = d.get('district') or d.get('District') or ''
        state = d.get('state') or d.get('State') or ''
        stage = d.get('stage_percent') or d.get('stage') or d.get('Stage') or ''
        status = d.get('status') or d.get('Status') or ''
        recharge = d.get('recharge') or d.get('recharge_bcm') or ''
        extraction = d.get('extraction') or d.get('extraction_bcm') or ''
        line = []
        if district:
            line.append(f"District: {district}")
        if state:
            line.append(f"State: {state}")
        if stage not in (None, ''):
            line.append(f"Stage: {stage}")
        if status:
            line.append(f"Status: {status}")
        if recharge not in (None, ''):
            line.append(f"Recharge (BCM): {recharge}")
        if extraction not in (None, ''):
            line.append(f"Extraction (BCM): {extraction}")
        parts.append(' | '.join(line))
    return '\n'.join(parts)

STATE_DATA = {
    "punjab": {
        "recharge_bcm": 18.6,
        "extraction_bcm": 26.3,
        "stage_percent": 156.0,
        "status": "Over-exploited",
        "source": "CGWB Dynamic Groundwater Resources, 2024–25 (Parliament reply)"
    },
    "rajasthan": {
        "recharge_bcm": None,
        "extraction_bcm": None,
        "stage_percent": 147.1,
        "status": "Over-exploited",
        "source": "CGWB Dynamic Groundwater Resources, 2024–25"
    },
    "haryana": {
        "recharge_bcm": None,
        "extraction_bcm": None,
        "stage_percent": 136.8,
        "status": "Over-exploited",
        "source": "CGWB Dynamic Groundwater Resources, 2024–25"
    },
}

HTML_PAGE = """
<!doctype html>
<html>
  <head>
    <title>INGRES Groundwater Demo Bot</title>
  </head>
  <body>
    <h1>INGRES Groundwater Demo Bot</h1>

    <p>
      Ask simple questions about groundwater recharge, extraction, or the status
      of Punjab, Haryana, or Rajasthan. This is a small demo using real CGWB-style data.
    </p>

    <form method="post">
      <label for="question"><strong>Type your question:</strong></label><br>
      <input
        type="text"
        id="question"
        name="question"
        style="width:450px"
        placeholder="Example: Punjab groundwater status or What is recharge?"
        autofocus
      >
      <button type="submit">Ask</button>
    </form>

    <hr>

    <p><strong>You asked:</strong> {user_question}</p>
    <p><strong>Bot answer:</strong><br>{bot_answer}</p>
  </body>
</html>
"""


# Do not read CSV at import time — load it inside request handlers to avoid
# startup failures when the file is missing or unreadable.



@app.route('/home', methods=["GET", "POST"])
def home():
    user_q = ""
    answer = ""

    if request.method == "POST":
        user_q = request.form.get("question", "")
        text = user_q.lower()

        for state in ["punjab", "rajasthan", "haryana"]:
            if state in text:
                s = STATE_DATA[state]
                answer = (
                    f"For {state.title()}, the latest CGWB assessment shows a groundwater "
                    f"development stage of about {s['stage_percent']}%, which means extraction "
                    f"is far above long-term recharge, so the state is classified as "
                    f"\"{s['status']}\".\n"
                )
                break

        if not answer:
            if "recharge" in text:
                answer = "Groundwater recharge is the amount of water added to the aquifer each year."
            elif "extraction" in text:
                answer = "Groundwater extraction is the water pumped or withdrawn from the aquifer."
            else:
                answer = ask_llm(user_q)

    return HTML_PAGE.format(user_question=user_q, bot_answer=answer.replace('\n', '<br>'))


@app.route('/')
def index():
    try:
        return render_template('chat.html')
    except Exception as e:
        print('Template render error for / :', e)
        # Return a minimal fallback page so the service remains available
        fallback = """
        <!doctype html>
        <html><head><title>INGRES AI</title></head>
        <body>
        <h1>INGRES AI</h1>
        <p>Service temporarily unavailable: template render failed.</p>
        </body></html>
        """
        return fallback, 200, {'Content-Type': 'text/html'}


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/get', methods=['POST'])
def get_data():
    """Compatibility endpoint for older UI that sends JSON {msg: text} and
    expects {'response': ...}.
    """
    try:
        payload = request.get_json(silent=True) or {}
        user_msg = (payload.get('msg') if isinstance(payload, dict) else None) or (request.json or {}).get('msg', '')
        user_msg = (user_msg or '').strip()

        # reuse the same logic as /chat: load CSV and search
        df = pd.read_csv('groundwater.csv')

        q = user_msg
        if q:
            import re
            tokens = [t for t in re.findall(r"[\w']+", q) if len(t) > 0]
            mask = pd.Series([False] * len(df))
            for tok in tokens:
                tok_mask = (
                    df['district'].astype(str).str.contains(tok, case=False, na=False) |
                    df['state'].astype(str).str.contains(tok, case=False, na=False)
                )
                mask = mask | tok_mask

            if not mask.any():
                try:
                    choices = df['district'].dropna().astype(str).unique().tolist()
                    matches = get_close_matches(q, choices, n=5, cutoff=0.6)
                    if matches:
                        mask = df['district'].astype(str).isin(matches)
                    else:
                        state_choices = df['state'].dropna().astype(str).unique().tolist()
                        matches = get_close_matches(q, state_choices, n=5, cutoff=0.6)
                        if matches:
                            mask = df['state'].astype(str).isin(matches)
                except Exception:
                    pass
        else:
            mask = pd.Series([False] * len(df))

        relevant_rows = df[mask]
        if relevant_rows.empty:
            context = "No specific district data found."
            # Build prompt and call LLM for more general answers
            prompt = f"""You are CGWB INGRES AI.
        
EXACT DATA (copy numbers exactly):
{context}

If the data above contains exact numbers for the district/state asked about, use them exactly in your answer. If exact numbers are not present, explain that exact figures are unavailable and provide clear, practical guidance and recommendations without inventing numeric values.

User: {user_msg}
"""

            try:
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.choices[0].message.content
            except Exception as e:
                print("LLM error in /get:", e)
                return jsonify({'response': 'Sorry, I could not fetch groundwater data.'})

            return jsonify({'response': answer})
        else:
            # If CSV has matching rows, return an HTML-formatted metric table for the first match
            try:
                row = relevant_rows.iloc[0]
                district = row.get('district') or row.get('District') or ''
                state = row.get('state') or row.get('State') or ''
                recharge = row.get('recharge_bcm') if 'recharge_bcm' in row.index else row.get('recharge') if 'recharge' in row.index else None
                extraction = row.get('extraction_bcm') if 'extraction_bcm' in row.index else row.get('extraction') if 'extraction' in row.index else None
                stage = row.get('stage_percent') if 'stage_percent' in row.index else row.get('stage') if 'stage' in row.index else None

                # coerce to numeric where possible
                try:
                    recharge_val = float(recharge) if recharge not in (None, '') else None
                except Exception:
                    recharge_val = None
                try:
                    extraction_val = float(extraction) if extraction not in (None, '') else None
                except Exception:
                    extraction_val = None
                try:
                    stage_val = float(stage) if stage not in (None, '') else None
                except Exception:
                    stage_val = None

                status = "Over-Exploited" if (stage_val is not None and stage_val > 100) else "Safe"
                gap = None
                if (recharge_val is not None) and (extraction_val is not None):
                    gap = extraction_val - recharge_val

                # format values for display
                recharge_disp = f"{recharge_val}" if recharge_val is not None else "N/A"
                extraction_disp = f"{extraction_val}" if extraction_val is not None else "N/A"
                stage_disp = f"{stage_val}" if stage_val is not None else "N/A"
                gap_disp = f"{gap:.1f}" if gap is not None else "N/A"

                response = f"""
**📍 {district}, {state} Groundwater Status**

<table class="metric-table">
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Recharge (BCM)</td><td>{recharge_disp}</td></tr>
<tr><td>Extraction (BCM)</td><td>{extraction_disp}</td></tr>
<tr><td>Stage (%)</td><td class="status-{status.lower().replace('-','')}">{stage_disp}%</td></tr>
<tr><td>Status</td><td class="status-{status.lower().replace('-','')}">{status}</td></tr>
</table>

**⚠️ Analysis:** Extraction exceeds recharge by **{gap_disp} BCM**. {status} zone.
"""
            except Exception:
                response = "No groundwater data found for this district. Try: Amritsar, Ludhiana, Bathinda."

            return jsonify({'response': response})
    except Exception as e:
        print("Server error in /get:", e)
        return jsonify({'response': 'Sorry, I could not fetch groundwater data.'})


@app.route('/chat', methods=['POST'])
def chat_post():
    """Primary chat endpoint used by the newer UI. Expects JSON {query: ...}
    and returns {'answer': ...}.
    """
    try:
        payload = request.get_json(silent=True) or {}
        user_q = (payload.get('query') if isinstance(payload, dict) else None) or (request.json or {}).get('query', '')
        user_q = (user_q or '').strip()

        df = pd.read_csv('groundwater.csv')

        q = user_q
        if q:
            import re
            tokens = [t for t in re.findall(r"[\w']+", q) if len(t) > 0]
            mask = pd.Series([False] * len(df))
            for tok in tokens:
                tok_mask = (
                    df['district'].astype(str).str.contains(tok, case=False, na=False) |
                    df['state'].astype(str).str.contains(tok, case=False, na=False)
                )
                mask = mask | tok_mask

            if not mask.any():
                try:
                    choices = df['district'].dropna().astype(str).unique().tolist()
                    matches = get_close_matches(q, choices, n=5, cutoff=0.6)
                    if matches:
                        mask = df['district'].astype(str).isin(matches)
                    else:
                        state_choices = df['state'].dropna().astype(str).unique().tolist()
                        matches = get_close_matches(q, state_choices, n=5, cutoff=0.6)
                        if matches:
                            mask = df['state'].astype(str).isin(matches)
                except Exception:
                    pass
        else:
            mask = pd.Series([False] * len(df))

        relevant_rows = df[mask]
        if relevant_rows.empty:
            context = "No specific district data found."
            prompt = f"""You are CGWB INGRES AI.

EXACT DATA (copy numbers exactly):
{context}

If the data above contains exact numbers for the district/state asked about, use them exactly in your answer. If exact numbers are not present, explain that exact figures are unavailable and provide clear, practical guidance and recommendations without inventing numeric values.

User: {user_q}
"""

            try:
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.choices[0].message.content
            except Exception as e:
                print("LLM error in /chat:", e)
                return jsonify({'answer': 'Sorry, I could not generate an answer right now.'}), 502

            return jsonify({'answer': answer})
        else:
            summary = summarize_rows(relevant_rows)
            return jsonify({'answer': summary})
    except Exception as e:
        print("Server error in /chat:", e)
        return jsonify({'answer': 'Server error'}), 500


# At the BOTTOM of server.py (replace if __name__ == '__main__')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
    
