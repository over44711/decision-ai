import os
from dotenv import load_dotenv
import anthropic
import openai
from google import genai
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


CLAUDE_MODEL = "claude-sonnet-4-6"
GPT_MODEL = "gpt-5.4"
GEMINI_MODEL = "gemini-3.5-flash"


load_dotenv()

# Load prompt file / 读取提示词文件
def load_prompt(stage):
    with open(f"prompts/{stage}.txt", "r", encoding="utf-8") as f:
        return f.read()

# Call Claude / 调用Claude
def ask_claude(prompt):
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except anthropic.AuthenticationError:
        print("❌ Claude API key invalid / API密钥无效")
        return None
    except anthropic.RateLimitError:
        print("❌ Claude rate limit reached / 请求频率超限，请稍后再试")
        return None
    except anthropic.BadRequestError as e:
        print(f"❌ Claude bad request / 请求错误：{e}")
        return None
    except Exception as e:
        print(f"❌ Claude unexpected error / 未知错误：{e}")
        return None

# Call GPT / 调用GPT
def ask_gpt(prompt):
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=GPT_MODEL,
            max_completion_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except openai.AuthenticationError:
        print("❌ GPT API key invalid / API密钥无效")
        return None
    except openai.RateLimitError:
        print("❌ GPT rate limit or quota exceeded / 请求超限或余额不足")
        return None
    except openai.BadRequestError as e:
        print(f"❌ GPT bad request / 请求错误：{e}")
        return None
    except Exception as e:
        print(f"❌ GPT unexpected error / 未知错误：{e}")
        return None

# Call Gemini / 调用Gemini
def ask_gemini(prompt):
    try:
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg or "authentication" in error_msg.lower():
            print("❌ Gemini API key invalid / API密钥无效")
        elif "quota" in error_msg.lower() or "429" in error_msg:
            print("❌ Gemini quota exceeded / 余额不足或请求超限")
        elif "not found" in error_msg.lower() or "404" in error_msg:
            print(f"❌ Gemini model not found / 模型不存在：{GEMINI_MODEL}")
        else:
            print(f"❌ Gemini unexpected error / 未知错误：{e}")
        return None

# Run three AI calls in parallel / 并行调用三个AI
def run_parallel(tasks):
    results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fn, prompt): name for name, fn, prompt in tasks}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                print(f"❌ {name} parallel error / 并行调用错误：{e}")
                results[name] = None
    return results

# Summarize a response into 100 words / 压缩回答为100字摘要
def summarize(response, ai_name):
    if response is None or response.startswith("["):
        return response
    prompt = f"""Summarize the following AI response into a concise position statement of no more than 100 words.
Focus only on: the core recommendation, the key reasoning, and whether the position was modified after seeing other AIs' responses.
Do not include formatting or headers.
Respond in the same language as the content below.

{ai_name}'s response:
{response}"""
    # return ask_claude(prompt)  # using Claude for summarization
    return ask_gemini(prompt)  # using Gemini for summarization

# Stage 1: Independent analysis / 三个AI独立回答
def stage1(question):
    print("\n========== STAGE 1: Independent Analysis / 独立分析 ==========\n")
    template = load_prompt("stage1")
    prompt = template.replace("{{question}}", question)

    print("⏳ All three AIs analyzing in parallel... / 三个AI并行分析中...")
    t_start = time.time()

    tasks = [
        ("claude", ask_claude, prompt),
        ("gpt", ask_gpt, prompt),
        ("gemini", ask_gemini, prompt),
    ]
    results = run_parallel(tasks)

    claude_answer = results.get("claude") or "[Claude response unavailable / Claude响应失败]"
    gpt_answer = results.get("gpt") or "[GPT response unavailable / GPT响应失败]"
    gemini_answer = results.get("gemini") or "[Gemini response unavailable / Gemini响应失败]"

    t_end = time.time()

    print(f"✅ Stage 1 complete / Stage 1完成 ⏱ {t_end - t_start:.1f}s\n")
    print(f"--- Claude ---\n{claude_answer}\n")
    print(f"--- GPT ---\n{gpt_answer}\n")
    print(f"--- Gemini ---\n{gemini_answer}\n")

    return claude_answer, gpt_answer, gemini_answer

# Stage 2: Cross-critique / 三个AI互相评论
def stage2(question, claude_s1, gpt_s1, gemini_s1):
    print("\n========== STAGE 2: Comparison & Critique / 对比与评论 ==========\n")
    template = load_prompt("stage2")

    claude_prompt = template.replace("{{question}}", question)\
                            .replace("{{current_AI_response}}", claude_s1)\
                            .replace("{{AI_A_name}}", "GPT")\
                            .replace("{{AI_A_response}}", gpt_s1)\
                            .replace("{{AI_B_name}}", "Gemini")\
                            .replace("{{AI_B_response}}", gemini_s1)

    gpt_prompt = template.replace("{{question}}", question)\
                         .replace("{{current_AI_response}}", gpt_s1)\
                         .replace("{{AI_A_name}}", "Claude")\
                         .replace("{{AI_A_response}}", claude_s1)\
                         .replace("{{AI_B_name}}", "Gemini")\
                         .replace("{{AI_B_response}}", gemini_s1)

    gemini_prompt = template.replace("{{question}}", question)\
                            .replace("{{current_AI_response}}", gemini_s1)\
                            .replace("{{AI_A_name}}", "Claude")\
                            .replace("{{AI_A_response}}", claude_s1)\
                            .replace("{{AI_B_name}}", "GPT")\
                            .replace("{{AI_B_response}}", gpt_s1)
    
    t_start = time.time()   
    print("⏳ All three AIs critiquing in parallel... / 三个AI并行评论中...")

    tasks = [
        ("claude", ask_claude, claude_prompt),
        ("gpt", ask_gpt, gpt_prompt),
        ("gemini", ask_gemini, gemini_prompt),
    ]
    results = run_parallel(tasks)

    claude_s2 = results.get("claude") or "[Claude response unavailable / Claude响应失败]"
    gpt_s2 = results.get("gpt") or "[GPT response unavailable / GPT响应失败]"
    gemini_s2 = results.get("gemini") or "[Gemini response unavailable / Gemini响应失败]"

    t_end = time.time()
    print(f"✅ Stage 2 complete / Stage 2完成 ⏱ {t_end - t_start:.1f}s\n")
    print(f"--- Claude ---\n{claude_s2}\n")
    print(f"--- GPT ---\n{gpt_s2}\n")
    print(f"--- Gemini ---\n{gemini_s2}\n")

    return claude_s2, gpt_s2, gemini_s2

# Compress Stage 2 responses into summaries / 压缩Stage 2回答为摘要
def summarize_stage2(claude_s2, gpt_s2, gemini_s2):
    print("\n⏳ Compressing Stage 2 responses... / 压缩Stage 2回答中...")
    t_start = time.time()  
    tasks = [
        ("claude", lambda p: summarize(claude_s2, "Claude"), ""),
        ("gpt", lambda p: summarize(gpt_s2, "GPT"), ""),
        ("gemini", lambda p: summarize(gemini_s2, "Gemini"), ""),
    ]
    results = run_parallel(tasks)

    claude_summary = results.get("claude") or claude_s2
    gpt_summary = results.get("gpt") or gpt_s2
    gemini_summary = results.get("gemini") or gemini_s2

    t_end = time.time()
    print(f"✅ Compression complete / 压缩完成 ⏱ {t_end - t_start:.1f}s\n")
    return claude_summary, gpt_summary, gemini_summary


# Main program / 主程序
if __name__ == "__main__":

    print("\n========== Current Models / 当前使用模型 ==========")
    print(f"Claude : {CLAUDE_MODEL}")
    print(f"GPT    : {GPT_MODEL}")
    print(f"Gemini : {GEMINI_MODEL}")
    print("====================================================\n")

    question = input("Please enter your decision question / 请输入你的决策问题：\n> ")

    print("\nAdding relevant background information can improve analysis quality.")
    print("请补充与问题相关的背景信息可以提高分析质量。")
    print("(Press Enter to skip / 直接回车跳过)")
    background = input("> ").strip()

    if background:
        full_question = f"{question}\n\n[User Background Information / 用户补充背景信息]：{background}"
    else:
        full_question = question

    print("\nSelect the final synthesizer / 选择最终总结者：")
    print("1. Claude (default / 默认)")
    print("2. GPT")
    print("3. Gemini")
    judge_choice = input("Enter number / 请输入数字（1/2/3），press Enter for default / 直接回车使用默认：").strip()

    claude_s1, gpt_s1, gemini_s1 = stage1(full_question)
    claude_s2, gpt_s2, gemini_s2 = stage2(full_question, claude_s1, gpt_s1, gemini_s1)

    # Compress Stage 2 for Stage 3 / 压缩Stage 2用于Stage 3
    claude_s2_summary, gpt_s2_summary, gemini_s2_summary = summarize_stage2(claude_s2, gpt_s2, gemini_s2)

    print("\n========== STAGE 3: Final Synthesis / 最终综合总结 ==========\n")

    template = load_prompt("stage3")
    prompt = template.replace("{{question}}", question)\
                     .replace("{{background}}", background if background else "Not provided / 未提供")\
                     .replace("{{Claude_round1}}", claude_s1)\
                     .replace("{{ChatGPT_round1}}", gpt_s1)\
                     .replace("{{Gemini_round1}}", gemini_s1)\
                     .replace("{{Claude_round2}}", claude_s2_summary)\
                     .replace("{{ChatGPT_round2}}", gpt_s2_summary)\
                     .replace("{{Gemini_round2}}", gemini_s2_summary)

    t_start = time.time()
    print("⏳ Generating final synthesis... / 综合总结生成中...")
    if judge_choice == "2":
        print("Synthesizer / 总结者：GPT")
        summary = ask_gpt(prompt)
    elif judge_choice == "3":
        print("Synthesizer / 总结者：Gemini")
        summary = ask_gemini(prompt)
    else:
        print("Synthesizer / 总结者：Claude (default / 默认)")
        summary = ask_claude(prompt)

    if summary is None:
        summary = "[Final synthesis unavailable / 最终总结生成失败]"

    print(summary)

    # Decision sufficiency reminder / 决策充分度提示
    print("\n========== Decision Sufficiency / 决策充分度提示 ==========")
    print("Check the 'Information Still Needed' section in the summary for scores.")
    print("请查看总结中\"还应该补充的信息\"部分的评分。")
    print("• All items ≤ 4  →  Recommendation can be acted upon directly")
    print("• 所有缺失项 ≤ 4分 →  当前建议可直接参考执行")
    print("• Any item ≥ 5   →  Recommend providing that information first")
    print("• 存在 ≥ 5分的缺失项 →  建议补充该信息后重新运行")
    print("============================================================\n")

    # Auto-save results / 自动保存结果
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"outputs/{timestamp}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Decision Analysis Report / 决策分析报告\n\n")
        f.write(f"**Question / 问题：** {question}\n\n")
        if background:
            f.write(f"**Background / 背景信息：** {background}\n\n")
        f.write(f"**Total Time / 总耗时：** {total_time:.1f}s\n\n")
        f.write(f"---\n\n## Stage 1: Independent Analysis / 独立分析\n\n")
        f.write(f"### Claude\n{claude_s1}\n\n")
        f.write(f"### GPT\n{gpt_s1}\n\n")
        f.write(f"### Gemini\n{gemini_s1}\n\n")
        f.write(f"---\n\n## Stage 2: Comparison & Critique / 对比评论\n\n")
        f.write(f"### Claude\n{claude_s2}\n\n")
        f.write(f"### GPT\n{gpt_s2}\n\n")
        f.write(f"### Gemini\n{gemini_s2}\n\n")
        f.write(f"---\n\n## Stage 2 Summaries / Stage 2摘要\n\n")
        f.write(f"### Claude Summary\n{claude_s2_summary}\n\n")
        f.write(f"### GPT Summary\n{gpt_s2_summary}\n\n")
        f.write(f"### Gemini Summary\n{gemini_s2_summary}\n\n")
        f.write(f"---\n\n## Stage 3: Final Synthesis / 最终总结\n\n{summary}\n")

    t_end = time.time()
    print(f"\n⏱ Total time taken / 总耗时：{t_end - t_start:.1f}s")

    print(f"✓ Report saved / 结果已保存：{filename}")