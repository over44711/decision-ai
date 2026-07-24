import os
from dotenv import load_dotenv
import anthropic
import openai
from google import genai
from datetime import datetime


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
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# Call GPT / 调用GPT
def ask_gpt(prompt):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=GPT_MODEL,
        max_completion_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Call Gemini / 调用Gemini
def ask_gemini(prompt):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return response.text

# Stage 1: Independent analysis / 三个AI独立回答
def stage1(question):
    print("\n========== STAGE 1: Independent Analysis / 独立分析 ==========\n")
    template = load_prompt("stage1")
    prompt = template.replace("{{question}}", question)

    print("⏳ Claude analyzing... / 分析中...（1/3）")
    claude_answer = ask_claude(prompt)
    print("✅ Claude done / 完成")
    print(claude_answer)

    print("\n⏳ GPT analyzing... / 分析中...（2/3）")
    gpt_answer = ask_gpt(prompt)
    print("✅ GPT done / 完成")
    print(gpt_answer)

    print("\n⏳ Gemini analyzing... / 分析中...（3/3）")
    gemini_answer = ask_gemini(prompt)
    print("✅ Gemini done / 完成")
    print(gemini_answer)

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

    print("⏳ Claude critiquing... / 评论中...（1/3）")
    claude_s2 = ask_claude(claude_prompt)
    print("✅ Claude done / 完成")
    print(claude_s2)

    print("\n⏳ GPT critiquing... / 评论中...（2/3）")
    gpt_s2 = ask_gpt(gpt_prompt)
    print("✅ GPT done / 完成")
    print(gpt_s2)

    print("\n⏳ Gemini critiquing... / 评论中...（3/3）")
    gemini_s2 = ask_gemini(gemini_prompt)
    print("✅ Gemini done / 完成")
    print(gemini_s2)

    return claude_s2, gpt_s2, gemini_s2


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

    print("\n========== STAGE 3: Final Synthesis / 最终综合总结 ==========\n")

    template = load_prompt("stage3")
    prompt = template.replace("{{question}}", question)\
                     .replace("{{background}}", background if background else "Not provided / 未提供")\
                     .replace("{{Claude_round1}}", claude_s1)\
                     .replace("{{ChatGPT_round1}}", gpt_s1)\
                     .replace("{{Gemini_round1}}", gemini_s1)\
                     .replace("{{Claude_round2}}", claude_s2)\
                     .replace("{{ChatGPT_round2}}", gpt_s2)\
                     .replace("{{Gemini_round2}}", gemini_s2)

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

    print(summary)

    # Decision sufficiency reminder / 决策充分度提示
    print("\n========== Decision Sufficiency / 决策充分度提示 ==========")
    print("Check the 'Information Still Needed' section in the summary for scores.")
    print("请查看总结中"还应该补充的信息"部分的评分。")
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
        f.write(f"---\n\n## Stage 1: Independent Analysis / 独立分析\n\n")
        f.write(f"### Claude\n{claude_s1}\n\n")
        f.write(f"### GPT\n{gpt_s1}\n\n")
        f.write(f"### Gemini\n{gemini_s1}\n\n")
        f.write(f"---\n\n## Stage 2: Comparison & Critique / 对比评论\n\n")
        f.write(f"### Claude\n{claude_s2}\n\n")
        f.write(f"### GPT\n{gpt_s2}\n\n")
        f.write(f"### Gemini\n{gemini_s2}\n\n")
        f.write(f"---\n\n## Stage 3: Final Synthesis / 最终总结\n\n{summary}\n")

    print(f"✓ Report saved / 结果已保存：{filename}")