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

# 读取提示词文件
def load_prompt(stage):
    with open(f"prompts/{stage}.txt", "r", encoding="utf-8") as f:
        return f.read()

# 调用Claude
def ask_claude(prompt):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# 调用GPT
def ask_gpt(prompt):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=GPT_MODEL,
        max_completion_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 调用Gemini
def ask_gemini(prompt):
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return response.text

# Stage 1：三个AI独立回答
def stage1(question):
    print("\n========== STAGE 1：独立分析 ==========\n")
    
    template = load_prompt("stage1")
    prompt = template.replace("{{问题}}", question)
    
    print("--- Claude 回答中... ---")
    claude_answer = ask_claude(prompt)
    print(claude_answer)
    
    print("\n--- GPT 回答中... ---")
    gpt_answer = ask_gpt(prompt)
    print(gpt_answer)
    
    print("\n--- Gemini 回答中... ---")
    gemini_answer = ask_gemini(prompt)
    print(gemini_answer)
    
    return claude_answer, gpt_answer, gemini_answer

# Stage 2：三个AI互相评论
def stage2(question, claude_s1, gpt_s1, gemini_s1):
    print("\n========== STAGE 2：对比与评论 ==========\n")
    
    template = load_prompt("stage2")
    
    # 给Claude看GPT和Gemini的回答
    claude_prompt = template.replace("{{问题}}", question)\
                            .replace("{{当前AI第一轮回答}}", claude_s1)\
                            .replace("{{AI_A_名称}}", "GPT")\
                            .replace("{{另一个AI的回答}}", gpt_s1)\
                            .replace("{{AI_B_名称}}", "Gemini")\
                            .replace("{{第三个AI的回答}}", gemini_s1)
    
    # 给GPT看Claude和Gemini的回答
    gpt_prompt = template.replace("{{问题}}", question)\
                         .replace("{{当前AI第一轮回答}}", gpt_s1)\
                         .replace("{{AI_A_名称}}", "Claude")\
                         .replace("{{另一个AI的回答}}", claude_s1)\
                         .replace("{{AI_B_名称}}", "Gemini")\
                         .replace("{{第三个AI的回答}}", gemini_s1)
    
    # 给Gemini看Claude和GPT的回答
    gemini_prompt = template.replace("{{问题}}", question)\
                            .replace("{{当前AI第一轮回答}}", gemini_s1)\
                            .replace("{{AI_A_名称}}", "Claude")\
                            .replace("{{另一个AI的回答}}", claude_s1)\
                            .replace("{{AI_B_名称}}", "GPT")\
                            .replace("{{第三个AI的回答}}", gpt_s1)
    
    print("--- Claude 评论中... ---")
    claude_s2 = ask_claude(claude_prompt)
    print(claude_s2)
    
    print("\n--- GPT 评论中... ---")
    gpt_s2 = ask_gpt(gpt_prompt)
    print(gpt_s2)
    
    print("\n--- Gemini 评论中... ---")
    gemini_s2 = ask_gemini(gemini_prompt)
    print(gemini_s2)
    
    return claude_s2, gpt_s2, gemini_s2

# Stage 3：最终综合总结
def stage3(question, claude_s1, gpt_s1, gemini_s1, claude_s2, gpt_s2, gemini_s2):
    print("\n========== STAGE 3：最终综合总结 ==========\n")
    
    template = load_prompt("stage3")
    
    prompt = template.replace("{{问题}}", question)\
                     .replace("{{Claude第一轮回答}}", claude_s1)\
                     .replace("{{ChatGPT第一轮回答}}", gpt_s1)\
                     .replace("{{Gemini第一轮回答}}", gemini_s1)\
                     .replace("{{Claude第二轮回答}}", claude_s2)\
                     .replace("{{ChatGPT第二轮回答}}", gpt_s2)\
                     .replace("{{Gemini第二轮回答}}", gemini_s2)
    
    print("--- 综合总结生成中... ---")
    summary = ask_claude(prompt)
    print(summary)
    
    return summary


# 主程序
if __name__ == "__main__":

    print("\n========== 当前使用模型 ==========")
    print(f"Claude：{CLAUDE_MODEL}")
    print(f"GPT：{GPT_MODEL}")
    print(f"Gemini：{GEMINI_MODEL}")
    print("==================================\n")
    
    question = input("请输入你的决策问题：")
    
    print("\n选择最终总结者（直接回车默认使用Claude）：")
    print("1. Claude（默认）")
    print("2. GPT")
    print("3. Gemini")
    judge_choice = input("请输入数字（1/2/3），或直接回车：").strip()
    
    claude_s1, gpt_s1, gemini_s1 = stage1(question)
    claude_s2, gpt_s2, gemini_s2 = stage2(question, claude_s1, gpt_s1, gemini_s1)
    
    print("\n========== STAGE 3：最终综合总结 ==========\n")
    
    template = load_prompt("stage3")
    prompt = template.replace("{{问题}}", question)\
                     .replace("{{Claude第一轮回答}}", claude_s1)\
                     .replace("{{ChatGPT第一轮回答}}", gpt_s1)\
                     .replace("{{Gemini第一轮回答}}", gemini_s1)\
                     .replace("{{Claude第二轮回答}}", claude_s2)\
                     .replace("{{ChatGPT第二轮回答}}", gpt_s2)\
                     .replace("{{Gemini第二轮回答}}", gemini_s2)
    
    print("--- 综合总结生成中... ---")
    if judge_choice == "2":
        print("总结者：GPT")
        summary = ask_gpt(prompt)
    elif judge_choice == "3":
        print("总结者：Gemini")
        summary = ask_gemini(prompt)
    else:
        print("总结者：Claude（默认）")
        summary = ask_claude(prompt)
    
    print(summary)


    # 自动保存结果
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"outputs/{timestamp}.md"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# 决策分析报告\n\n")
        f.write(f"**问题：** {question}\n\n")
        f.write(f"---\n\n## Stage 1：独立分析\n\n")
        f.write(f"### Claude\n{claude_s1}\n\n")
        f.write(f"### GPT\n{gpt_s1}\n\n")
        f.write(f"### Gemini\n{gemini_s1}\n\n")
        f.write(f"---\n\n## Stage 2：对比评论\n\n")
        f.write(f"### Claude\n{claude_s2}\n\n")
        f.write(f"### GPT\n{gpt_s2}\n\n")
        f.write(f"### Gemini\n{gemini_s2}\n\n")
        f.write(f"---\n\n## Stage 3：最终总结\n\n{summary}\n")
    
    print(f"\n✓ 结果已保存至：{filename}")