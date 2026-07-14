# Decision AI

一个多AI决策分析平台，通过结构化三阶段框架，让Claude、GPT、Gemini独立分析同一问题并相互辩论，最终生成综合建议。

## 核心理念

结构化的跨AI辩论比任何单一AI的回答更全面可靠。

## 三阶段框架

- **Stage 1**：三个AI独立分析，互不参考
- **Stage 2**：三个AI互相评论，指出对方的漏洞和假设
- **Stage 3**：由选定的AI综合总结，给出最终建议

## 使用的模型

- Claude：claude-sonnet-4-6
- GPT：gpt-5.4
- Gemini：gemini-3.5-flash

## 快速开始

1. 克隆项目
2. 安装依赖：`pip install -r requirements.txt`
3. 创建`.env`文件，填入三家API密钥：

    ANTHROPIC_API_KEY=你的密钥
    OPENAI_API_KEY=你的密钥
    GOOGLE_API_KEY=你的密钥

4. 运行：`python main.py`

## 项目结构
    decision-ai/
    ├── main.py          # 主程序
    ├── prompts/         # 提示词模板
    │   ├── stage1.txt
    │   ├── stage2.txt
    │   └── stage3.txt
    ├── outputs/         # 分析结果（自动保存）
    ├── .env             # API密钥（不上传）
    └── requirements.txt
    