# nebul-cli

A minimal command-line client for the [Nebul](https://nebul.io) inference API.
Send single prompts, pipe in text, attach files, or start an interactive
multi-turn chat session — all from your terminal.

## Features

- **Single-shot queries** — pass a prompt as an argument, via `--prompt`, or
  piped through stdin.
- **File input** — attach a file with `-f`/`--file` and optionally prepend
  instructions.
- **Interactive chat** — multi-turn REPL with conversation history, slash
  commands, and streaming responses.
- **Chat file & image input** — `/file` in chat takes multiple files, an inline
  instruction, Tab completion, and images for vision-capable models.
- **Model selection** — override the default model per query or per session.
- **Streaming** — responses stream by default; disable with `--no-stream`.
- **Usage stats** — optional token usage reported to stderr.
- **JSON mode** — request JSON-formatted output with `--json`.

## Requirements

- Python 3.8+
- [`openai`](https://pypi.org/project/openai/) Python package

```bash
pip install openai
```

## Setup

1. Get an API key from <https://studio.nebul.io>.
2. Export it as an environment variable:

```bash
export NEBUL_API_KEY=sk-...
```

3. (Optional) Make the script executable and put it on your `PATH`:

```bash
chmod +x nebul.py
ln -s "$(pwd)/nebul.py" ~/.local/bin/nebul
```

## Usage

```
nebul [query ...] [options]
```

### Options

| Flag | Description |
| --- | --- |
| `query` | Prompt text; multiple words are joined. |
| `-m, --model MODEL` | Model id (default: `openai/gpt-oss-120b` or `$NEBUL_MODEL`). |
| `-s, --system TEXT` | System prompt. |
| `-p, --prompt TEXT` | Prompt text (alternative to positional). |
| `-f, --file PATH` | Read `PATH` and use its contents as the prompt. Combine with a query/prompt to prepend instructions. |
| `-c, --chat` | Start an interactive multi-turn chat session. |
| `--temperature N` | Sampling temperature (default: `0.7`). |
| `--max-tokens N` | Maximum tokens to generate. |
| `--json` | Request JSON-formatted output. |
| `--no-stream` | Disable streaming output. |
| `--no-newline-on-stream` | Do not append a trailing newline after streaming. |
| `--usage` | Print token usage to stderr (default: on for single queries, off in chat). |
| `--no-usage` | Do not print token usage. |
| `-l, --list-models` | List available models and exit. |
| `-V, --version` | Print the version (date of last change, YYYYMMDD) and exit. |

### Environment variables

| Variable | Description |
| --- | --- |
| `NEBUL_API_KEY` | **Required.** Your API key (`sk-...`). |
| `NEBUL_MODEL` | Default model if `--model` is not given. |
| `NEBUL_BASE_URL` | Override the API base URL. |

## Examples

### Single-shot queries

```bash
nebul 'Why is privacy important in the age of AI?'

nebul -m deepseek-ai/deepseek-r1 'Explain the CAP theorem'

echo 'summarize this text' | nebul

nebul --system 'You are a haiku bot' 'write one about cats'

nebul --no-usage 'just give me the answer'
```

### File input

```bash
# Use the file contents as the prompt
nebul -f notes.txt

# Prepend an instruction to the file contents
nebul -f notes.txt 'summarize this'

nebul -f main.py -s 'You are a code reviewer' 'review this script'
```

### Interactive chat

```bash
# Start a chat session
nebul --chat

# With a system prompt
nebul --chat --system 'You are a helpful coding assistant'

# Seed the first turn with a file
nebul --chat -f main.py

# Combine system prompt + file seed
nebul --chat --system 'You are a code reviewer' -f main.py
```

#### Chat commands

| Command | Description |
| --- | --- |
| `/help` | Show command help. |
| `/quit` or `/exit` | Exit the chat (also Ctrl+D / Ctrl+C). |
| `/clear` | Clear conversation history (keeps the system prompt). |
| `/system <text>` | Set or replace the system prompt. |
| `/file <path> [more paths] [instruction]` | Send file(s) as your next message, optionally with an instruction. Supports Tab completion, `~`, quoted paths, and image files (PNG/JPG/... for vision models). |
| `/model [id]` | Show or switch the current model. |
| `/show` | Print the current conversation. |
| `/export [path]` | Save the conversation to a markdown file (default `nebul-chat-<timestamp>.md`). |

### List available models

```bash
nebul -l            # or --list-models
```

### Use a different default model

```bash
NEBUL_MODEL=mistralai/mistral-large nebul 'hello'
```

## License

This project is provided as-is for use with the Nebul inference API.
