# OpenAI → Anthropic Proxy

将任意 **OpenAI 兼容** 的 LLM API 转换为 **Anthropic Messages API** 格式的本地代理服务器。
运行后即可让 [Claude Code](https://docs.anthropic.com/en/docs/claude-code) 接入 OpenAI、
Groq、Together、Ollama、LM Studio 等任意 OpenAI 格式的后端模型。

```
┌─────────────┐   Anthropic 格式    ┌──────────────┐   OpenAI 格式    ┌──────────────┐
│ Claude Code │ ─────────────────▶ │  本代理服务   │ ───────────────▶ │ OpenAI 后端   │
│             │ ◀───────────────── │ (localhost)  │ ◀─────────────── │ (任意兼容源)  │
└─────────────┘   Anthropic 格式    └──────────────┘   OpenAI 格式    └──────────────┘
```

## ✨ 特性

- 完整实现 Anthropic `POST /v1/messages` 端点（流式 + 非流式）
- 双向转换：Anthropic ⇄ OpenAI 请求 / 响应格式
- 🖥️ **本地 Web 控制台**：内置对话调试台 + 数据看板，浏览器直接输入与查看
- 📊 **数据统计**：实时统计请求数、输入/输出 tokens、平均延迟、错误数，并按模型聚合、记录最近请求
- 支持 **工具调用 (function calling / tool use)**
- 支持 **多模态图片** 输入（base64 / url）
- 支持 **系统提示、温度、top_p、stop 序列、max_tokens** 等参数
- 流式 SSE 事件完整还原（`message_start` → `content_block_*` → `message_delta` → `message_stop`）
- 模型分层映射：Claude 的 sonnet/opus/haiku 自动映射到你配置的大小模型
- `count_tokens` 端点估算
- 零数据库、纯本地运行、单文件配置

## 🚀 快速开始

### 1. 一键启动（推荐）

```bash
./run.sh
```

脚本会自动创建虚拟环境、安装依赖、复制 `.env`，然后启动服务。
首次运行后请编辑生成的 `.env` 填入你的后端信息。

### 2. 手动启动

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # 编辑填入你的 API Key / Base URL
python server.py
```

服务默认监听 `http://127.0.0.1:8082`。

## 🖥️ 本地控制台

启动后用浏览器打开 **http://127.0.0.1:8082** 即可看到内置控制台：

- **💬 对话调试**：直接输入消息与后端模型对话（支持流式输出、system 提示、温度、max_tokens），
  方便快速验证代理是否工作，无需再敲 curl。
- **📊 数据看板**：实时展示总请求数、流式请求数、累计输入/输出 tokens、平均延迟、错误数；
  并按模型聚合统计，列出最近请求明细，每 3 秒自动刷新，可一键清空。

统计数据持久化在项目根目录的 `stats.json`（已加入 `.gitignore`），重启后不丢失。

## ⚙️ 配置（.env）

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `HOST` / `PORT` | 本地监听地址与端口 | `127.0.0.1` / `8082` |
| `OPENAI_BASE_URL` | 上游 OpenAI 兼容 API 地址（含 `/v1`） | `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | 上游 API Key | — |
| `BIG_MODEL` | sonnet / opus 映射到的模型 | `gpt-4o` |
| `SMALL_MODEL` | haiku 映射到的模型 | `gpt-4o-mini` |
| `REQUEST_TIMEOUT` | 请求超时（秒） | `120` |
| `MAX_TOKENS_LIMIT` | max_tokens 上限，0 表示不限制 | `0` |
| `ANTHROPIC_API_KEY` | 若设置，则要求 Claude Code 携带该 Key | 空（不校验） |

常见后端示例：

```bash
# OpenAI
OPENAI_BASE_URL=https://api.openai.com/v1
# Groq
OPENAI_BASE_URL=https://api.groq.com/openai/v1
# Together
OPENAI_BASE_URL=https://api.together.xyz/v1
# 本地 Ollama
OPENAI_BASE_URL=http://localhost:11434/v1
# LM Studio
OPENAI_BASE_URL=http://localhost:1234/v1
```

## 🔌 接入 Claude Code

启动代理后，在另一个终端设置环境变量并运行 `claude`：

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8082
export ANTHROPIC_API_KEY=any-value   # 若 .env 中未设置 ANTHROPIC_API_KEY，可任意填
claude
```

之后 Claude Code 的所有请求都会经由本代理转发给你配置的 OpenAI 兼容后端。

## 🧪 测试

```bash
source .venv/bin/activate

# 单元测试（转换逻辑，无需网络）
python tests/test_conversion.py

# 端到端测试（内置 mock 上游，验证完整链路）
python tests/e2e_mock.py
```

## 📡 API 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/v1/messages` | 主聊天端点（流式 / 非流式） |
| `POST` | `/v1/messages/count_tokens` | 估算输入 token 数 |
| `GET` | `/` | 本地 Web 控制台（对话调试 + 数据看板） |
| `GET` | `/info` | 服务信息（JSON） |
| `GET` | `/dashboard/stats` | 聚合统计数据（JSON） |
| `POST` | `/dashboard/reset` | 清空统计 |
| `GET` | `/health` | 健康检查 |

手动调用示例：

```bash
curl http://127.0.0.1:8082/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: any-value" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

## 📁 项目结构

```
.
├── app/
│   ├── __init__.py
│   ├── config.py       # 环境变量配置与模型映射
│   ├── models.py       # Anthropic 请求 Pydantic 模型
│   ├── converter.py    # Anthropic ⇄ OpenAI 转换（非流式）
│   ├── streaming.py    # 流式 SSE 事件转换
│   ├── stats.py        # 请求统计（持久化到 stats.json）
│   ├── static/
│   │   └── index.html  # 本地 Web 控制台（对话调试 + 数据看板）
│   └── main.py         # FastAPI 应用与端点
├── tests/
│   ├── test_conversion.py
│   └── e2e_mock.py
├── server.py           # 启动入口
├── run.sh              # 一键安装 + 启动脚本
├── requirements.txt
└── .env.example
```

## 📄 License

MIT
