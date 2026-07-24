# OpenAI → Anthropic Proxy

将任意 **OpenAI 兼容** 的 LLM API 转换为 **Anthropic Messages API** 格式的本地代理服务器。
适用于：**你购买的 API 只支持 OpenAI 格式，但想用 [Claude Code](https://docs.anthropic.com/en/docs/claude-code)**。
在本地运行本服务，在网页上填入你的 OpenAI 接口与 Key，系统会生成一个 Anthropic 格式的
接口地址与 Key，把它们填入 Claude Code 即可——对你而言只是换了一个地址。

```
┌─────────────┐   Anthropic 格式    ┌──────────────┐   OpenAI 格式    ┌──────────────┐
│ Claude Code │ ─────────────────▶ │  本代理服务   │ ───────────────▶ │ OpenAI 后端   │
│             │ ◀───────────────── │ (localhost)  │ ◀─────────────── │ (任意兼容源)  │
└─────────────┘   Anthropic 格式    └──────────────┘   OpenAI 格式    └──────────────┘
```

## ✨ 特性

- 🖥️ **网页接入配置**：浏览器填写 OpenAI 接口与 Key，自动生成 Anthropic 接口地址与 Key，一键复制到 Claude Code
- ⚙️ **运行时配置**：在网页修改上游地址/密钥/模型映射并立即生效，持久化到 `config.json`，无需重启
- 🔐 **自动鉴权**：自动生成 Anthropic Key 并对 `/v1/*` 鉴权，可一键重生
- 完整实现 Anthropic `POST /v1/messages` 端点（流式 + 非流式）
- 双向转换：Anthropic ⇄ OpenAI 请求 / 响应格式，支持工具调用、多模态图片
- 🧠 **推理模型支持**：自动把上游的 `reasoning_content`（如 DeepSeek 思维链）转换成 Anthropic `thinking` 块（流式 + 非流式）
- 📊 **数据看板**：实时统计请求数、输入/输出 tokens、平均延迟、错误数，按模型聚合
- 流式 SSE 事件完整还原；模型分层映射（sonnet/opus → 大模型，haiku → 小模型）
- 零数据库、纯本地运行

## 🚀 快速开始

### 1. 启动服务

```bash
./run.sh          # 自动建虚拟环境、装依赖并启动
# 或手动：
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python server.py
```

### 2. 用浏览器打开控制台 http://127.0.0.1:8082

1. 在 **⚙️ 接入配置** 页填入你的 **OpenAI Base URL** 与 **API Key**，根据需要调整大/小模型，点“保存配置”（可先“测试连接”）。
2. 页面会自动生成一个 **ANTHROPIC_BASE_URL** 与 **ANTHROPIC_API_KEY**。
3. 把这两个值填入 Claude Code 即可：

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8082"
export ANTHROPIC_API_KEY="页面上生成的 sk-ant-proxy-..."
claude
```

或写入 `~/.claude/settings.json`（页面提供一键复制）：

```json
{ "env": { "ANTHROPIC_BASE_URL": "http://127.0.0.1:8082", "ANTHROPIC_API_KEY": "sk-ant-proxy-..." } }
```

之后 Claude Code 的所有请求都会经由本转换器转发到你配置的 OpenAI 兼容后端。

> 提示：上游地址也可直接写在 `.env`（`cp .env.example .env`）作为启动默认值；网页保存的 `config.json` 会覆盖它。

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
| `POST` | `/v1/messages` | 主聊天端点（流式 / 非流式，需携带 Anthropic Key） |
| `POST` | `/v1/messages/count_tokens` | 估算输入 token 数 |
| `GET` | `/` | 本地接入配置页 + 数据看板 |
| `GET` | `/info` | 服务信息（JSON） |
| `GET`/`POST` | `/config` | 读取 / 保存运行时配置 |
| `POST` | `/config/regenerate-key` | 重新生成 Anthropic Key |
| `POST` | `/config/test` | 测试上游连接 |
| `GET` | `/dashboard/stats` | 聚合统计数据（JSON） |
| `POST` | `/dashboard/reset` | 清空统计 |
| `GET` | `/health` | 健康检查 |

手动调用示例（`x-api-key` 填页面生成的 Anthropic Key）：

```bash
curl http://127.0.0.1:8082/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: sk-ant-proxy-..." \
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
│   ├── config.py       # 静态服务设置（host/port/超时等，来自环境变量）
│   ├── runtime_config.py  # 运行时配置（上游/密钥/模型映射/Anthropic Key，持久化 config.json）
│   ├── models.py       # Anthropic 请求 Pydantic 模型
│   ├── converter.py    # Anthropic ⇄ OpenAI 转换（非流式）
│   ├── streaming.py    # 流式 SSE 事件转换
│   ├── stats.py        # 请求统计（持久化到 stats.json）
│   ├── static/
│   │   └── index.html  # 本地接入配置页 + 数据看板
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
