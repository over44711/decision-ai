import os
from dotenv import load_dotenv
import anthropic
import openai
from google import genai

load_dotenv()

def test_claude():
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "用一句话介绍你自己。"}]
    )
    print("Claude回复：", message.content[0].text)

def test_gpt():
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=100,
        messages=[{"role": "user", "content": "用一句话介绍你自己。"}]
    )
    print("GPT回复：", response.choices[0].message.content)

def test_gemini():
    
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="用一句话介绍你自己。"
    )
    print("Gemini回复：", response.text)

if __name__ == "__main__":
    test_claude()
    test_gpt()
    test_gemini()