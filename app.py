import streamlit as st
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import anthropic
import openai
from google import genai
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# ── Models ──────────────────────────────────────────────
CLAUDE_MODEL  = "claude-sonnet-4-6"
GPT_MODEL     = "gpt-5.4"
GEMINI_MODEL  = "gemini-3.5-flash"

PRICES = {
    "claude": {"input": 3.0,  "output": 15.0},
    "gpt":    {"input": 2.5,  "output": 10.0},
    "gemini": {"input": 0.3,  "output": 2.5},
}

# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title="Decision AI", page_icon="🧠", layout="wide")

st.title("🧠 Decision AI")
st.caption("Multi-AI structured debate → synthesized recommendation")

# ── Session state ────────────────────────────────────────
if "token_tracker" not in st.session_state:
    st.session_state.token_tracker = {
        "claude": {"input": 0, "output": 0},
        "gpt":    {"input": 0, "output": 0},
        "gemini": {"input": 0, "output": 0},
    }
if "history" not in st.session_state:
    st.session_state.history = []          # list of round dicts
if "round_count" not in st.session_state:
    st.session_state.round_count = 0
if "filename" not in st.session_state:
    st.session_state.filename = None

# ── Helpers ──────────────────────────────────────────────
def track(model, tokens):
    if tokens:
        st.session_state.token_tracker[model]["input"]  += tokens["input"]
        st.session_state.token_tracker[model]["output"] += tokens["output"]

def load_prompt(stage):
    with open(f"prompts/{stage}.txt", "r", encoding="utf-8") as f:
        return f.read()

def ask_claude(prompt):
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=4000,
            messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text, {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}
    except Exception as e:
        return f"[Claude error: {e}]", None

def ask_gpt(prompt):
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        r = client.chat.completions.create(
            model=GPT_MODEL, max_completion_tokens=4000,
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content, {"input": r.usage.prompt_tokens, "output": r.usage.completion_tokens}
    except Exception as e:
        return f"[GPT error: {e}]", None

def ask_gemini(prompt):
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        r = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return r.text, {"input": r.usage_metadata.prompt_token_count, "output": r.usage_metadata.candidates_token_count}
    except Exception as e:
        return f"[Gemini error: {e}]", None

def run_parallel(tasks):
    results = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(fn, p): name for name, fn, p in tasks}
        for f in as_completed(futures):
            name = futures[f]
            try:
                results[name] = f.result()
            except Exception as e:
                results[name] = (f"[{name} error: {e}]", None)
    return results

def summarize(response, ai_name):
    if not response or response.startswith("["):
        return response, None
    prompt = f"""Summarize the following AI response into a concise position statement of no more than 100 words.
Focus only on: the core recommendation, the key reasoning, and whether the position was modified.
Do not include formatting or headers. Respond in the same language as the content below.
{ai_name}'s response:\n{response}"""
    return ask_gemini(prompt)

def cost_table():
    tracker = st.session_state.token_tracker
    total = 0
    rows = []
    for m, u in tracker.items():
        ic = u["input"]  / 1e6 * PRICES[m]["input"]
        oc = u["output"] / 1e6 * PRICES[m]["output"]
        mc = ic + oc
        total += mc
        rows.append({"Model": m.upper(), "Input": f"{u['input']:,}", "Output": f"{u['output']:,}", "Cost": f"${mc:.4f}"})
    rows.append({"Model": "**TOTAL**", "Input": "", "Output": "", "Cost": f"**${total:.4f}**"})
    return rows

def save_report(question, background, rounds, synthesizer_name):
    os.makedirs("outputs", exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    fn = f"outputs/{ts}.md"
    with open(fn, "w", encoding="utf-8") as f:
        f.write(f"# Decision Analysis Report\n\n")
        f.write(f"**Question:** {question}\n\n")
        if background:
            f.write(f"**Background:** {background}\n\n")
        for i, r in enumerate(rounds):
            label = "Main" if i == 0 else f"Follow-up {i}"
            f.write(f"---\n\n## {label} Round\n\n")
            f.write(f"### Stage 1\n\n#### Claude\n{r['s1']['claude']}\n\n#### GPT\n{r['s1']['gpt']}\n\n#### Gemini\n{r['s1']['gemini']}\n\n")
            f.write(f"### Stage 2\n\n#### Claude\n{r['s2']['claude']}\n\n#### GPT\n{r['s2']['gpt']}\n\n#### Gemini\n{r['s2']['gemini']}\n\n")
            f.write(f"### Stage 3 — *Synthesized by {synthesizer_name}*\n\n{r['summary']}\n\n")
        f.write(f"---\n\n## Token Usage\n\n")
        f.write("| Model | Input | Output | Cost |\n|---|---|---|---|\n")
        tracker = st.session_state.token_tracker
        total = 0
        for m, u in tracker.items():
            ic = u["input"] / 1e6 * PRICES[m]["input"]
            oc = u["output"] / 1e6 * PRICES[m]["output"]
            mc = ic + oc
            total += mc
            f.write(f"| {m.upper()} | {u['input']:,} | {u['output']:,} | ${mc:.4f} |\n")
        f.write(f"| **TOTAL** | | | **${total:.4f}** |\n")
    return fn

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    synthesizer = st.radio("Final Synthesizer", ["Claude (default)", "GPT", "Gemini"])
    judge_map = {"Claude (default)": ("claude", ask_claude, f"Claude ({CLAUDE_MODEL})"),
                 "GPT":              ("gpt",    ask_gpt,    f"GPT ({GPT_MODEL})"),
                 "Gemini":           ("gemini", ask_gemini, f"Gemini ({GEMINI_MODEL})")}
    judge_key, judge_fn, judge_label = judge_map[synthesizer]

    st.divider()
    st.caption(f"**Models**\n- Claude: `{CLAUDE_MODEL}`\n- GPT: `{GPT_MODEL}`\n- Gemini: `{GEMINI_MODEL}`")

    st.divider()
    if st.button("🗑️ Reset session"):
        for k in ["token_tracker", "history", "round_count", "filename"]:
            del st.session_state[k]
        st.rerun()

    st.divider()
    st.subheader("📊 Token Usage")
    for row in cost_table():
        st.write(f"{row['Model']}: {row['Input']} in / {row['Output']} out — {row['Cost']}")

# ── Main input area ───────────────────────────────────────
is_followup = st.session_state.round_count > 0

if not is_followup:
    st.subheader("Your Decision Question")
    question    = st.text_area("Question", placeholder="E.g. Should I learn Python or JavaScript first?", height=80)
    background  = st.text_area("Background (optional)", placeholder="Age, location, constraints, preferences...", height=80)
    run_label   = "🚀 Run Analysis"
else:
    st.subheader("Follow-up Question")
    question    = st.session_state.history[0]["question"]
    background  = st.session_state.history[0]["background"]
    followup_q  = st.text_area("Follow-up", placeholder="Ask a follow-up based on the analysis above...", height=80)
    run_label   = "🔄 Run Follow-up"

run_btn = st.button(run_label, type="primary")

# ── Show previous rounds ──────────────────────────────────
for i, rnd in enumerate(st.session_state.history):
    label = "📋 Main Analysis" if i == 0 else f"🔄 Follow-up {i}"
    with st.expander(label, expanded=(i == len(st.session_state.history) - 1)):
        tabs = st.tabs(["Stage 1", "Stage 2", "Stage 3"])
        with tabs[0]:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Claude**\n\n{rnd['s1']['claude']}")
            c2.markdown(f"**GPT**\n\n{rnd['s1']['gpt']}")
            c3.markdown(f"**Gemini**\n\n{rnd['s1']['gemini']}")
        with tabs[1]:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Claude**\n\n{rnd['s2']['claude']}")
            c2.markdown(f"**GPT**\n\n{rnd['s2']['gpt']}")
            c3.markdown(f"**Gemini**\n\n{rnd['s2']['gemini']}")
        with tabs[2]:
            st.markdown(rnd["summary"])
        st.caption(f"⏱ {rnd['elapsed']:.1f}s")

        # Download button for the report
        if st.session_state.filename and os.path.exists(st.session_state.filename):
            with open(st.session_state.filename, "r", encoding="utf-8") as f:
                report_content = f.read()
            st.download_button(
                label="📥 Download Report",
                data=report_content,
                file_name=os.path.basename(st.session_state.filename),
                mime="text/markdown",
                key=f"download_{i}"
            )

# ── Run pipeline ──────────────────────────────────────────
if run_btn:
    if not is_followup:
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()
        current_q  = question.strip()
        current_bg = background.strip()
        full_q = current_q + (f"\n\n[Background]: {current_bg}" if current_bg else "")
        prior_s1  = None
        prior_sum = None
    else:
        if not followup_q.strip():
            st.warning("Please enter a follow-up question.")
            st.stop()
        current_q  = followup_q.strip()
        current_bg = background
        last = st.session_state.history[-1]
        prior_s1  = last["s1"]
        prior_sum = last["summary"]
        full_q    = current_q

    t0 = time.time()

    # Stage 1
    with st.status("⏳ Stage 1: Independent Analysis...", expanded=True) as status:
        tpl1 = load_prompt("stage1")

        if prior_s1 is None:
            p_claude = p_gpt = p_gemini = tpl1.replace("{{question}}", full_q)
        else:
            def mk_fu_prompt(name, pr, ps):
                return (f"You are continuing an analysis.\n\nOriginal Question: {question}\n"
                        f"Background (already provided, do not re-ask): {current_bg or 'Not provided'}\n\n"
                        f"Your Previous Response:\n{pr}\n\n"
                        f"Round 1 Final Synthesis (reference only, treat as working hypothesis):\n{ps}\n\n"
                        f"Follow-up Question: {current_q}\n\n"
                        f"IMPORTANT: Do not list background info as missing.\n\n"
                        ) + tpl1.replace("{{question}}", current_q)
            p_claude  = mk_fu_prompt("Claude",  prior_s1["claude"],  prior_sum)
            p_gpt     = mk_fu_prompt("GPT",     prior_s1["gpt"],     prior_sum)
            p_gemini  = mk_fu_prompt("Gemini",  prior_s1["gemini"],  prior_sum)

        r1 = run_parallel([("claude", ask_claude, p_claude),
                           ("gpt",    ask_gpt,    p_gpt),
                           ("gemini", ask_gemini, p_gemini)])
        s1 = {}
        for name in ("claude", "gpt", "gemini"):
            txt, tok = r1.get(name, ("[unavailable]", None))
            s1[name] = txt or "[unavailable]"
            track(name, tok)
        status.update(label="✅ Stage 1 complete", state="complete")

    # Stage 2
    with st.status("⏳ Stage 2: Cross-Critique...", expanded=True) as status:
        tpl2 = load_prompt("stage2")
        def mk2(cur, a_name, a_resp, b_name, b_resp):
            return tpl2.replace("{{question}}", current_q)\
                       .replace("{{current_AI_response}}", cur)\
                       .replace("{{AI_A_name}}", a_name).replace("{{AI_A_response}}", a_resp)\
                       .replace("{{AI_B_name}}", b_name).replace("{{AI_B_response}}", b_resp)
        r2 = run_parallel([
            ("claude", ask_claude, mk2(s1["claude"], "GPT", s1["gpt"], "Gemini", s1["gemini"])),
            ("gpt",    ask_gpt,    mk2(s1["gpt"],    "Claude", s1["claude"], "Gemini", s1["gemini"])),
            ("gemini", ask_gemini, mk2(s1["gemini"], "Claude", s1["claude"], "GPT", s1["gpt"])),
        ])
        s2 = {}
        for name in ("claude", "gpt", "gemini"):
            txt, tok = r2.get(name, ("[unavailable]", None))
            s2[name] = txt or "[unavailable]"
            track(name, tok)
        status.update(label="✅ Stage 2 complete", state="complete")

    # Compress Stage 2
    with st.status("⏳ Compressing Stage 2...", expanded=False) as status:
        sums = {}
        for name in ("claude", "gpt", "gemini"):
            txt, tok = summarize(s2[name], name.capitalize())
            sums[name] = txt or s2[name]
            track("gemini", tok)
        status.update(label="✅ Compression complete", state="complete")

    # Stage 3
    with st.status(f"⏳ Stage 3: Final Synthesis ({judge_label})...", expanded=True) as status:
        tpl3 = load_prompt("stage3")
        p3 = tpl3.replace("{{question}}", current_q)\
                 .replace("{{background}}", current_bg or "Not provided")\
                 .replace("{{Claude_round1}}", s1["claude"])\
                 .replace("{{ChatGPT_round1}}", s1["gpt"])\
                 .replace("{{Gemini_round1}}", s1["gemini"])\
                 .replace("{{Claude_round2}}", sums["claude"])\
                 .replace("{{ChatGPT_round2}}", sums["gpt"])\
                 .replace("{{Gemini_round2}}", sums["gemini"])
        final, tok3 = judge_fn(p3)
        final = final or "[Synthesis unavailable]"
        track(judge_key, tok3)
        status.update(label="✅ Stage 3 complete", state="complete")

    elapsed = time.time() - t0

    # Save round
    rnd = {
        "question": current_q, "background": current_bg,
        "s1": s1, "s2": s2, "summary": final, "elapsed": elapsed
    }
    if st.session_state.round_count == 0:
        rnd["question"] = question
        rnd["background"] = background
    st.session_state.history.append(rnd)
    st.session_state.round_count += 1

    # Save report
    fn = save_report(question if not is_followup else st.session_state.history[0]["question"],
                     background if not is_followup else st.session_state.history[0]["background"],
                     st.session_state.history, judge_label)
    st.session_state.filename = fn
    st.success(f"✅ Done in {elapsed:.1f}s — report saved to `{fn}`")
    st.rerun()