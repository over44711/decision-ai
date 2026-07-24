# Decision AI

A multi-AI decision analysis platform that routes the same question simultaneously to Claude, GPT, and Gemini, facilitates structured cross-critique between models, and produces a synthesized recommendation.

The core insight: **structured debate between AI models surfaces disagreements, hidden assumptions, and blind spots that no single AI can identify on its own.**

---

## How It Works

### Three-Stage Framework

**Stage 1 — Independent Analysis**
Each AI analyzes the question independently without seeing the others' responses. They are explicitly instructed not to hedge, not to use filler disclaimers, and to take a clear position.

**Stage 2 — Cross-Critique**
Each AI reads the other two responses and conducts a rational comparison — identifying what the others got right, what may be incorrect, what was missed, and what hidden assumptions were made. They must also state which opposing argument they found most compelling, and why it did or did not change their conclusion.

**Stage 3 — Synthesis**
A selected AI (default: Claude) acts as a neutral synthesizer, producing a structured final report that distinguishes consensus from genuine disagreement, evaluates reasoning quality, and rates the importance of any missing information on a 0–9 scale.

---

## Information Sufficiency Scoring

A key feature of this framework is the **Information Sufficiency Score** in the Stage 3 output.

Each piece of missing information is rated 0–9:
- **7–9**: Critical — missing this could lead to a fundamentally wrong recommendation
- **4–6**: Important — providing this would significantly improve the recommendation
- **0–3**: Nice to have — would only fine-tune the recommendation

If all missing items score 4 or below, the current recommendation can be acted upon directly. If any item scores 5 or above, that information should be gathered before making a final decision.

This solves the **information recursion problem** — the tendency for decision tools to keep asking for more and more information indefinitely.

---

## Models Used

| Role | Model |
|------|-------|
| Claude | claude-sonnet-4-6 |
| GPT | gpt-5.4 |
| Gemini | gemini-3.5-flash |

Models are defined as constants at the top of `main.py` and can be swapped in one line.

---

## Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/over44711/decision-ai.git
cd decision-ai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up API keys**

Create a `.env` file in the project root:
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

**4. Run**
```bash
python main.py
```

You will be prompted to enter a question and optional background information. The system runs all three stages automatically and saves the full report to `outputs/`.

---

## Project Structure

decision-ai/
├── main.py              # Main program
├── prompts/
│   ├── stage1.txt       # Independent analysis prompt
│   ├── stage2.txt       # Cross-critique prompt
│   └── stage3.txt       # Synthesis prompt
├── outputs/             # Auto-saved markdown reports
├── .env                 # API keys (excluded from git)
├── .gitignore
└── requirements.txt

---

## Key Design Decisions

**Why three different AI models?**
Models from different companies have different training data, fine-tuning approaches, and implicit biases. When they converge on the same answer, that convergence carries more weight. When they diverge, the disagreement itself is informative.

**Why structured prompts instead of free-form chat?**
Free-form multi-AI comparison tends toward polite agreement. The Stage 2 prompt is specifically designed to force each model to identify the strongest opposing argument and explain why it does or does not change their conclusion — preventing surface-level consensus.

**Why separate the synthesizer from the debaters?**
Having a model synthesize a debate it participated in introduces bias. The Stage 3 synthesizer receives all six responses (three from Stage 1, three from Stage 2) as a neutral reader with no prior position.

---

## Test Cases

Five decision scenarios tested across different domains:

| # | Domain | Question |
|---|--------|----------|
| 1 | Technical | Python vs JavaScript for a career switcher with 6 months |
| 2 | Career | Join an early-stage startup at a 20% pay cut with equity |
| 3 | Product | Is a medication reminder app for elderly users worth building |
| 4 | Naming | ClearMind vs Nexus for an enterprise AI decision tool |
| 5 | Ethics | Should a manager warn employees before a layoff announcement |

---

## Sample Output Structure
Decision Analysis Report
Question: ...
Background: ...
Stage 1: Independent Analysis
Claude
GPT
Gemini
Stage 2: Comparison & Critique
Claude
GPT
Gemini
Stage 3: Final Synthesis

Final Synthesized Recommendation
Key Consensus Among the Three AIs
Key Disagreements and Root Cause Classification
Most Compelling Arguments (with conditions for validity)
Information Still Needed (with 0-9 importance scores)
Information Sufficiency Assessment
Recommended Next Steps
Final Confidence Level

---

## Requirements

- Python 3.10+
- Anthropic API key
- OpenAI API key
- Google AI Studio API key