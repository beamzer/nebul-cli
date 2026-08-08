# Nebul AI chat and file input support

**Session ID:** ses_01f3ac6cdffeDvTArmbur64PWC
**Created:** 8/8/2026, 11:47:29 AM
**Updated:** 8/8/2026, 11:48:44 AM

---

## User

modify this python program for nebul AI so it also supports chat and can take a file as input

---

## Assistant (Build · GLM 5.2 FP8 · 2.1s)

_Thinking:_

The user wants me to modify a Python program for nebul AI so it supports chat and can take a file as input. Let me first explore the codebase to understand what we're working with.

**Tool: bash**

**Input:**
```json
{
  "command": "ls -la",
  "workdir": "/home/ewald/OpenCode/nebul-cli"
}
```

**Output:**
```
total 16
drwxr-xr-x 2 ewald ewald 4096 Jul 17 08:50 .
drwxr-xr-x 7 ewald ewald 4096 Aug  6 08:40 ..
-rwxr-xr-x 1 ewald ewald 6339 Jul 17 10:50 nebul.py

```

---

## Assistant (Build · GLM 5.2 FP8 · 0.6s)

_Thinking:_

Let me read the nebul.py file to understand the current implementation.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/ewald/OpenCode/nebul-cli/nebul.py"
}
```

**Output:**
```
<path>/home/ewald/OpenCode/nebul-cli/nebul.py</path>
<type>file</type>
<content>
1: #!/usr/bin/env python3
2: """Generic Nebul inference API command-line client."""
3: import argparse
4: import os
5: import sys
6: 
7: from openai import OpenAI
8: 
9: DEFAULT_BASE_URL = "https://api.inference.nebul.io/v1"
10: DEFAULT_MODEL = "openai/gpt-oss-120b"
11: 
12: 
13: def get_client(api_key: str, base_url: str) -> OpenAI:
14:     if not api_key:
15:         sys.stderr.write(
16:             "error: NEBUL_API_KEY environment variable is not set.\n"
17:             "Get a key from https://studio.nebul.io and export it:\n"
18:             "  export NEBUL_API_KEY=sk-...\n"
19:         )
20:         sys.exit(2)
21:     return OpenAI(api_key=api_key, base_url=base_url)
22: 
23: 
24: def list_models(client: OpenAI) -> None:
25:     models = client.models.list()
26:     ids = sorted(m.id for m in models.data)
27:     for mid in ids:
28:         print(mid)
29: 
30: 
31: def read_prompt(args: argparse.Namespace) -> str:
32:     if args.prompt:
33:         return args.prompt
34:     if not sys.stdin.isatty():
35:         data = sys.stdin.read()
36:         if data.strip():
37:             return data.rstrip("\n")
38:     if args.query:
39:         return " ".join(args.query)
40:     sys.stderr.write("error: no prompt provided. Pass a query or pipe text via stdin.\n")
41:     sys.exit(2)
42: 
43: 
44: def chat(args: argparse.Namespace) -> None:
45:     api_key = os.environ.get("NEBUL_API_KEY", "")
46:     base_url = os.environ.get("NEBUL_BASE_URL", DEFAULT_BASE_URL)
47:     client = get_client(api_key, base_url)
48: 
49:     messages = []
50:     if args.system:
51:         messages.append({"role": "system", "content": args.system})
52:     messages.append({"role": "user", "content": read_prompt(args)})
53: 
54:     model_explicit = args.model is not None
55:     model = args.model or os.environ.get("NEBUL_MODEL", DEFAULT_MODEL)
56: 
57:     kwargs = {
58:         "model": model,
59:         "messages": messages,
60:         "temperature": args.temperature,
61:     }
62:     if args.max_tokens is not None:
63:         kwargs["max_tokens"] = args.max_tokens
64:     if args.json:
65:         kwargs["response_format"] = {"type": "json_object"}
66: 
67:     if args.no_stream:
68:         response = client.chat.completions.create(**kwargs)
69:         content = response.choices[0].message.content
70:         sys.stdout.write(content or "")
71:         if not content or not content.endswith("\n"):
72:             sys.stdout.write("\n")
73:         if args.usage:
74:             if not model_explicit:
75:                 sys.stderr.write(f"[model] {model}\n")
76:             if response.usage:
77:                 u = response.usage
78:                 sys.stderr.write(
79:                     f"[usage] prompt={u.prompt_tokens} "
80:                     f"completion={u.completion_tokens} total={u.total_tokens}\n"
81:                 )
82:         return
83: 
84:     stream_kwargs = dict(kwargs)
85:     if args.usage:
86:         stream_kwargs["stream_options"] = {"include_usage": True}
87:     stream = client.chat.completions.create(stream=True, **stream_kwargs)
88:     saw_content = False
89:     usage = None
90:     for chunk in stream:
91:         if chunk.usage:
92:             usage = chunk.usage
93:         if not chunk.choices:
94:             continue
95:         delta = chunk.choices[0].delta
96:         if delta and delta.content:
97:             saw_content = True
98:             sys.stdout.write(delta.content)
99:             sys.stdout.flush()
100:     if saw_content and not args.no_newline_on_stream:
101:         sys.stdout.write("\n")
102:     if args.usage:
103:         if not model_explicit:
104:             sys.stderr.write(f"[model] {model}\n")
105:         if usage:
106:             sys.stderr.write(
107:                 f"[usage] prompt={usage.prompt_tokens} "
108:                 f"completion={usage.completion_tokens} total={usage.total_tokens}\n"
109:             )
110: 
111: 
112: def build_parser() -> argparse.ArgumentParser:
113:     parser = argparse.ArgumentParser(
114:         prog="nebul",
115:         description="Query Nebul inference models from the command line.",
116:         formatter_class=argparse.RawDescriptionHelpFormatter,
117:         epilog=(
118:             "Examples:\n"
119:             "  nebul 'Why is privacy important in the age of AI?'\n"
120:             "  nebul -m deepseek-ai/deepseek-r1 'Explain CAP theorem'\n"
121:             "  echo 'summarize this' | nebul\n"
122:             "  nebul --system 'You are a haiku bot' 'write one about cats'\n"
123:             "  nebul --no-usage 'just give me the answer'\n"
124:             "  nebul --list-models\n"
125:             "  NEBUL_MODEL=mistralai/mistral-large nebul 'hello'\n"
126:             "\n"
127:             "Environment:\n"
128:             "  NEBUL_API_KEY      Required. Your API key (sk-...).\n"
129:             "  NEBUL_MODEL        Default model if --model is not given.\n"
130:             "  NEBUL_BASE_URL     Override the API base URL.\n"
131:         ),
132:     )
133:     parser.add_argument(
134:         "query",
135:         nargs="*",
136:         help="The prompt text. Multiple words are joined. May be omitted if "
137:         "piped via stdin or if --prompt is used.",
138:     )
139:     parser.add_argument("-m", "--model", default=None,
140:                         help=f"Model id (default: {DEFAULT_MODEL} or $NEBUL_MODEL)")
141:     parser.add_argument("-s", "--system", help="Optional system prompt.")
142:     parser.add_argument("-p", "--prompt", help="Prompt text (alternative to positional).")
143:     parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
144:     parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens to generate.")
145:     parser.add_argument("--json", action="store_true", help="Request JSON-formatted output.")
146:     parser.add_argument("--no-stream", action="store_true", help="Disable streaming output.")
147:     parser.add_argument("--no-newline-on-stream", action="store_true",
148:                         help="Do not append a trailing newline after streaming output.")
149:     parser.add_argument("--usage", action="store_true", default=True,
150:                         help="Print token usage to stderr after the query (default).")
151:     parser.add_argument("--no-usage", action="store_false", dest="usage",
152:                         help="Do not print token usage.")
153:     parser.add_argument("--list-models", action="store_true", help="List available models and exit.")
154:     return parser
155: 
156: 
157: def main(argv=None) -> int:
158:     args = build_parser().parse_args(argv)
159:     if args.list_models:
160:         api_key = os.environ.get("NEBUL_API_KEY", "")
161:         base_url = os.environ.get("NEBUL_BASE_URL", DEFAULT_BASE_URL)
162:         list_models(get_client(api_key, base_url))
163:         return 0
164:     chat(args)
165:     return 0
166: 
167: 
168: if __name__ == "__main__":
169:     raise SystemExit(main())

(End of file - total 169 lines)
</content>
```

---

## Assistant (Build · GLM 5.2 FP8 · 65.6s)

_Thinking:_

Now I understand the codebase. It's a single-file Python CLI client for the Nebul AI inference API. The user wants me to:

1. Add support for "chat" - meaning an interactive multi-turn chat mode where the user can have a back-and-forth conversation
2. Add support for taking a file as input - so a user can pass a file (e.g. `--file path/to/file.txt`) and the contents are used as the prompt

Let me look at what's there. The program already has a single-turn `chat` function that does one query and exits. The user wants to add a real interactive chat mode (multi-turn).

Let me plan the changes:

1. **File input**: Add a `-f`/`--file` argument that reads the content of a file and uses it as the prompt (or part of the user message). Need to update `read_prompt` to also check `args.file`.

2. **Interactive chat mode**: Add a `--chat` flag that enters a REPL-style loop where the user can type messages, and the assistant responds, maintaining conversation history across turns.

Let me also consider whether the file input should be combined with the prompt or just used as the prompt. A natural approach: the file content becomes the user message (possibly combined with a prompt that gives instructions). Actually, a common pattern is: `--file file.txt "Summarize this"` would combine the file contents with the instruction. But to keep it simple, I'll make `--file` read the file and use it as the prompt content. If both `--prompt`/`--query` and `--file` are given, I could combine them. Let me make the file content the primary content, and if a query/prompt is also given, prepend it as instruction. Actually, let me think about the most natural design.

A clean design:
- `read_prompt` returns a single string for the single-turn case.
- For file input, read the file content.
- If both file and prompt/query are provided, combine: e.g. prompt + "\n\n" + file_content. This is useful for "summarize this file" use cases.

For the interactive chat mode:
- Maintain a `messages` list across turns.
- If `--system` is given, add it once at the start.
- Loop: read a line from stdin (using `input()`), append as user message, stream the response, append as assistant message.
- Handle exit on EOF (Ctrl+D) or commands like `/exit`, `/quit`, `/clear` (clear history), `/help`.
- Support `--file` in chat too? Could load a file and use it as the first user message, or as context. Maybe allow `/file <path>` command within chat to load a file into the conversation. That'd be a nice touch. But let me keep it focused. I'll support `--file` for the initial message in chat too, and also add a `/file` command within the chat REPL for loading files mid-conversation. Actually, let me keep it simpler but still useful.

Let me design the chat REPL:
- Print a banner showing the model.
- Show usage hints.
- On each prompt (`input("you> ")`):
  - Strip whitespace.
  - If empty, continue.
  - If starts with `/`:
    - `/quit` or `/exit` or `/q` -> break
    - `/clear` -> reset messages to just system (if any)
    - `/help` -> show commands
    - `/file <path>` -> load file content as next user message (or append). Actually simpler: load and send as user message.
    - `/system <text>` -> set/replace system message
    - `/show` -> print current messages
  - Otherwise: treat as user message, send to API, stream response.

For streaming in chat, reuse the streaming logic. I'll refactor the streaming into a helper that takes messages and args and returns the full content (so it can be appended to history) while also streaming to stdout.

Let me refactor `chat` to extract a `stream_response` helper that handles the streaming and returns the content + usage. And a non-stream variant.

Let me think about how to structure this cleanly.

I'll write a helper:

```python
def run_completion(client, messages, args, model, model_explicit):
    """Run a single completion. Returns (content, usage)."""
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": args.temperature,
    }
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens
    if args.json:
        kwargs["response_format"] = {"type": "json_object"}

    if args.no_stream:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        sys.stdout.write(content)
        if not content.endswith("\n"):
            sys.stdout.write("\n")
        return content, response.usage
```

But wait—the original non-stream path only writes newline conditionally and prints usage inline. For chat, I want to handle usage printing separately or suppress it per turn. Let me make usage optional in chat (print per turn if `--usage`). Hmm, in interactive chat, printing usage per turn could be noisy but it's behind the flag (default True currently). In chat mode, maybe default usage off, or respect the flag. I'll respect the flag but print to stderr per turn.

Actually, let me reconsider. The original default is `usage=True`. For chat, per-turn usage to stderr is fine and useful. Let me keep respecting the flag.

Let me refactor the streaming into a function that returns the accumulated content and usage object, and writes chunks to stdout. Then both single-turn and multi-turn can use it.

Let me write:

```python
def stream_completion(client, kwargs, want_usage):
    if want_usage:
        kwargs = dict(kwargs)
        kwargs["stream_options"] = {"include_usage": True}
    stream = client.chat.completions.create(stream=True, **kwargs)
    parts = []
    usage = None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            parts.append(delta.content)
            sys.stdout.write(delta.content)
            sys.stdout.flush()
    return "".join(parts), usage
```

For non-stream:

```python
def complete_once(client, kwargs, no_stream, want_usage):
    if no_stream:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        sys.stdout.write(content)
        if not content or not content.endswith("\n"):
            sys.stdout.write("\n")
        return content, response.usage
    return stream_completion(client, kwargs, want_usage)
```

Hmm but the original non-stream path doesn't add a newline if content already ends with one, and adds newline if content is empty. Let me match: original does:
```
sys.stdout.write(content or "")
if not content or not content.endswith("\n"):
    sys.stdout.write("\n")
```
So if content is empty -> writes "". Then `not content` is True -> writes "\n". Good. If content ends with "\n" -> no extra. Otherwise -> "\n". I'll keep that for single-turn. For chat, I think similar is fine.

Now `--no-newline-on-stream` only applies to single-turn streaming. In chat, each turn's response—should I add a newline? In a REPL, after streaming the assistant response, I want a newline before the next `you> ` prompt so they're on separate lines. So yes, always add a trailing newline after each assistant turn in chat (regardless of `--no-newline-on-stream`, which is a single-turn flag). Actually, to be safe and clean, in chat I'll write a newline after each response if it doesn't end with one, plus a blank line separator. Let me just ensure a trailing newline.

Now the usage printing. Let me make a helper:

```python
def print_usage(usage, model, model_explicit, want_usage):
    if not want_usage:
        return
    if not model_explicit:
        sys.stderr.write(f"[model] {model}\n")
    if usage:
        sys.stderr.write(
            f"[usage] prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} total={usage.total_tokens}\n"
        )
```

Now, for single-turn, the original printed usage even in non-stream mode. Let me keep behavior equivalent.

Now let me restructure `chat` for single-turn:

```python
def chat_single_turn(args):
    client = ...
    messages = build initial messages
    model, model_explicit = resolve model
    kwargs = base kwargs
    content, usage = complete_once(client, kwargs, args.no_stream, args.usage)
    if not args.no_stream and not args.no_newline_on_stream:
        # original: only when saw_content. We have content; if empty, stream path didn't add newline.
        ...
    print_usage(...)
```

Wait, the newline logic: original streaming path:
```
if saw_content and not args.no_newline_on_stream:
    sys.stdout.write("\n")
```
So only adds newline if there was content. My `stream_completion` appends content but doesn't add the final newline. So in single-turn I need: `if content and not args.no_newline_on_stream: sys.stdout.write("\n")`. But if content is empty (no content streamed), no newline. Matches original. Good.

For non-stream, the newline is handled inside complete_once. Hmm, inconsistent. Let me restructure so complete_once does NOT handle newlines, and the caller decides. That's cleaner.

Let me redefine:

```python
def stream_completion(client, kwargs, want_usage):
    if want_usage:
        kwargs = {**kwargs, "stream_options": {"include_usage": True}}
    stream = client.chat.completions.create(stream=True, **kwargs)
    parts = []
    usage = None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            parts.append(delta.content)
            sys.stdout.write(delta.content)
            sys.stdout.flush()
    return "".join(parts), usage


def complete_once(client, kwargs, no_stream):
    if no_stream:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        sys.stdout.write(content)
        return content, response.usage
    return stream_completion(client, kwargs, want_usage=True)
```

Wait, `want_usage` for streaming — should only request usage if `args.usage`. Let me pass it.

```python
def complete_once(client, kwargs, no_stream, want_usage):
    if no_stream:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        sys.stdout.write(content)
        return content, response.usage
    return stream_completion(client, kwargs, want_usage)
```

Then single-turn:

```python
content, usage = complete_once(client, kwargs, args.no_stream, args.usage)
# newline handling
if args.no_stream:
    if not content or not content.endswith("\n"):
        sys.stdout.write("\n")
else:
    if content and not args.no_newline_on_stream:
        sys.stdout.write("\n")
print_usage(usage, model, model_explicit, args.usage)
```

That matches original behavior. 

For chat multi-turn, after each turn:
```python
content, usage = complete_once(client, kwargs, args.no_stream, args.usage)
if not content.endswith("\n"):
    sys.stdout.write("\n")
print_usage(usage, model, model_explicit, args.usage)
messages.append({"role": "assistant", "content": content})
```

And `kwargs` needs `stream` not set (since complete_once handles it). Wait, kwargs shouldn't include stream; complete_once adds stream=True for the streaming branch. For non-stream branch, no stream param. Good. But the `stream_options` is added inside stream_completion. Good.

Now `--max-tokens` in chat: if set, applies to each turn. Fine.

`--json` in chat: response_format json would force each assistant response to JSON. That might be okay but unusual for chat. I'll just pass it through; user's choice.

Now the file input. Update `read_prompt`:

```python
def read_prompt(args):
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                file_content = f.read()
        except OSError as e:
            sys.stderr.write(f"error: could not read file {args.file!r}: {e}\n")
            sys.exit(2)
        instruction = None
        if args.prompt:
            instruction = args.prompt
        elif not sys.stdin.isatty():
            data = sys.stdin.read()
            if data.strip():
                instruction = data.rstrip("\n")
        elif args.query:
            instruction = " ".join(args.query)
        if instruction:
            return f"{instruction}\n\n{file_content}"
        return file_content.rstrip("\n")
    if args.prompt:
        return args.prompt
    ...
```

Hmm, this is getting complex. Let me think about what's the most intuitive behavior for `--file`:
- `nebul --file notes.txt` -> uses file content as prompt.
- `nebul --file notes.txt "summarize"` -> combines instruction + file content.

That's a reasonable, intuitive design. Let me keep it.

Now the `--chat` flag (interactive). Let me add argument `--chat` / `-c`? `-c` might conflict... there's no `-c` currently. But `-c` could be confused; I'll use `--chat` with no short, or `-c`. Let me check existing short flags: `-m`, `-s`, `-p`. `-c` is free. I'll add `--chat` and `-c`.

Wait, but there's already a function named `chat` for single-turn. I'll rename the single-turn function or keep it. Let me restructure:
- `run_single_turn(args)` — the single query.
- `run_interactive(args)` — the REPL.

`main` decides which to call based on `args.chat`.

Let me also support `--file` in interactive mode: if a file is given, send it as the first user message (with optional instruction). Then continue the conversation. That's useful: `nebul --chat --file code.py` then ask questions about it.

Let me now also think about the interactive prompt reading. Use `input()`. For multi-line, keep simple single-line input (Enter sends). That's typical for these CLIs. Could support backslash continuation but let's keep simple.

EOF (Ctrl+D) raises EOFError — catch and break gracefully.

Let me also handle KeyboardInterrupt — print newline and continue or break. I'll break on Ctrl+C with a newline.

Commands in REPL:
- `/help` — show commands
- `/quit`, `/exit`, `/q` — exit
- `/clear` — reset conversation (keep system message)
- `/system <text>` — set/replace system prompt
- `/file <path>` — load a file and send as the next user message
- `/show` — print conversation
- `/save <path>` — save conversation to file (nice extra)
- `/model <id>` — change model mid-chat (nice extra)

Let me include a reasonable subset: help, quit/exit, clear, system, file, show, model. Maybe save too. Let me include help, quit, exit, clear, system, file, show, model, save.

Actually, let me not over-engineer. I'll include: `/help`, `/quit`/`/exit`/`/q`, `/clear`, `/system`, `/file`, `/show`, `/model`. These are genuinely useful.

Now let me write the code. Let me also update the epilog/examples and the docstring.

Let me also make sure `read_prompt` still works for single-turn when no file. Keep existing logic, just add the file branch at the top.

Let me think about stdin in `--file` mode. If `--file` is given, we probably shouldn't read stdin unless combining. Actually I included stdin as a possible instruction source. That might be confusing. Let me simplify: with `--file`, the instruction comes from `--prompt` or positional `query` only (not stdin). If neither, file content alone. This avoids surprising stdin reads. Let me do that.

So:
```python
def read_prompt(args):
    if args.file:
        file_content = read_file(args.file)
        instruction = args.prompt or (args.query and " ".join(args.query)) or None
        if instruction:
            return f"{instruction}\n\n{file_content}"
        return file_content
    if args.prompt:
        return args.prompt
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            return data.rstrip("\n")
    if args.query:
        return " ".join(args.query)
    sys.stderr.write("error: no prompt provided. Pass a query, --file, --prompt, or pipe via stdin.\n")
    sys.exit(2)
```

Wait, `args.query and " ".join(args.query)` — if query is `[]` (empty list), `args.query` is falsy, so `" ".join` not evaluated, returns `[]`? No: `([] and X)` returns `[]` which is falsy, then `or None` returns None. Good. If query has items, `args.query` is truthy, returns `" ".join(args.query)`. Good.

Now `read_file` helper:
```python
def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().rstrip("\n")
    except OSError as e:
        sys.stderr.write(f"error: could not read file {path!r}: {e}\n")
        sys.exit(2)
```

Now the interactive mode. Let me write `run_interactive(args)`:

```python
def run_interactive(args):
    client = get_client(...)
    model_explicit = args.model is not None
    model = args.model or os.environ.get("NEBUL_MODEL", DEFAULT_MODEL)
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    def base_kwargs(msgs):
        kw = {"model": model, "messages": msgs, "temperature": args.temperature}
        if args.max_tokens is not None:
            kw["max_tokens"] = args.max_tokens
        if args.json:
            kw["response_format"] = {"type": "json_object"}
        return kw
```

Hmm, model can change via `/model`. Let me use a mutable holder. I'll use a list `state = {"model": model, "model_explicit": model_explicit}`. Or just reassign a local — but nested functions can't reassign enclosing locals without nonlocal. I'll use nonlocal. Let me structure with nonlocal model.

Let me write it directly without too many nested helpers.

```python
def run_interactive(args):
    api_key = os.environ.get("NEBUL_API_KEY", "")
    base_url = os.environ.get("NEBUL_BASE_URL", DEFAULT_BASE_URL)
    client = get_client(api_key, base_url)

    model_explicit = args.model is not None
    model = args.model or os.environ.get("NEBUL_MODEL", DEFAULT_MODEL)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    sys.stderr.write(f"nebul chat — model: {model} (type /help for commands, Ctrl+D to exit)\n")

    # If a file or initial prompt is provided, seed the first user turn.
    initial = None
    if args.file:
        initial = read_file(args.file)
        if args.prompt or args.query:
            instr = args.prompt or " ".join(args.query)
            initial = f"{instr}\n\n{initial}"
    elif args.prompt:
        initial = args.prompt
    elif args.query:
        initial = " ".join(args.query)
    elif not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            initial = data.rstrip("\n")

    if initial is not None:
        if not send_turn(client, args, messages, model, model_explicit, initial):
            pass

    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write("\n")
            break
        text = line.strip()
        if not text:
            continue
        if text.startswith("/"):
            handled, model, model_explicit, should_exit = handle_command(
                text, messages, args, model, model_explicit)
            if should_exit:
                break
            continue
        send_turn(client, args, messages, model, model_explicit, text)
```

Hmm, the `send_turn` returns content; returns False if no content? Let me define `send_turn` to send a user message, stream/print the response, append assistant message, print usage. Returns nothing special. Let me write it.

Actually, let me simplify by inlining. But a helper reduces duplication. Let me define `send_turn`.

```python
def send_turn(client, args, messages, model, model_explicit, user_text):
    messages.append({"role": "user", "content": user_text})
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": args.temperature,
    }
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens
    if args.json:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        content, usage = complete_once(client, kwargs, args.no_stream, args.usage)
    except Exception as e:
        sys.stderr.write(f"\n[error] {e}\n")
        # remove the user message so it can be retried
        messages.pop()
        return
    if not content.endswith("\n"):
        sys.stdout.write("\n")
    print_usage(usage, model, model_explicit, args.usage)
    messages.append({"role": "assistant", "content": content})
```

Wait, kwargs has `messages` referencing the same list; the API call will serialize it. Since we append assistant after the call, fine. But note: if `complete_once` streams, it uses `messages` list as given; we don't modify during streaming. Good.

But there's subtlety: `complete_once` for streaming passes `kwargs` to `stream_completion` which may add `stream_options`. It does `kwargs = dict(kwargs)` so fine.

Now `print_usage`:
```python
def print_usage(usage, model, model_explicit, want_usage):
    if not want_usage:
        return
    if not model_explicit:
        sys.stderr.write(f"[model] {model}\n")
    if usage:
        sys.stderr.write(
            f"[usage] prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} total={usage.total_tokens}\n"
        )
```

Now `complete_once`:
```python
def complete_once(client, kwargs, no_stream, want_usage):
    if no_stream:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        sys.stdout.write(content)
        return content, response.usage
    return stream_completion(client, kwargs, want_usage)


def stream_completion(client, kwargs, want_usage):
    if want_usage:
        kwargs = {**kwargs, "stream_options": {"include_usage": True}}
    stream = client.chat.completions.create(stream=True, **kwargs)
    parts = []
    usage = None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            parts.append(delta.content)
            sys.stdout.write(delta.content)
            sys.stdout.flush()
    return "".join(parts), usage
```

Now `handle_command`:
```python
def handle_command(text, messages, args, model, model_explicit):
    parts = text.split(maxsplit=1)
    cmd = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    if cmd in ("/quit", "/exit", "/q"):
        return True, model, model_explicit, True
    if cmd == "/help":
        sys.stdout.write(
            "Commands:\n"
            "  /help            show this help\n"
            "  /quit | /exit    exit the chat\n"
            "  /clear           clear conversation history (keeps system prompt)\n"
            "  /system <text>   set or replace the system prompt\n"
            "  /file <path>     send a file's contents as your next message\n"
            "  /model <id>      switch model\n"
            "  /show            print the current conversation\n"
        )
        return True, model, model_explicit, False
    if cmd == "/clear":
        messages.clear()
        if args.system:
            messages.append({"role": "system", "content": args.system})
        sys.stdout.write("[history cleared]\n")
        return True, model, model_explicit, False
    if cmd == "/system":
        if not rest:
            sys.stdout.write("usage: /system <text>\n")
            return True, model, model_explicit, False
        # remove existing system
        messages[:] = [m for m in messages if m["role"] != "system"]
        messages.insert(0, {"role": "system", "content": rest})
        args.system = rest
        sys.stdout.write("[system prompt set]\n")
        return True, model, model_explicit, False
    if cmd == "/file":
        if not rest:
            sys.stdout.write("usage: /file <path>\n")
            return True, model, model_explicit, False
        content = read_file(rest.strip())
        # mark to send as user message
        ...
```

Hmm, `/file` is tricky because it needs to trigger a send_turn. Let me restructure so `/file` returns the text to send, and the main loop sends it. Let me change the return signature to optionally include a message to send.

This is getting complicated with multiple return values. Let me restructure the main loop to handle commands inline rather than via a separate function returning many things. Inline is clearer.

Let me rewrite `run_interactive` with inline command handling:

```python
def run_interactive(args):
    api_key = os.environ.get("NEBUL_API_KEY", "")
    base_url = os.environ.get("NEBUL_BASE_URL", DEFAULT_BASE_URL)
    client = get_client(api_key, base_url)

    model_explicit = args.model is not None
    model = args.model or os.environ.get("NEBUL_MODEL", DEFAULT_MODEL)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    sys.stderr.write(
        f"nebul chat — model: {model} "
        f"(type /help for commands, Ctrl+D or /quit to exit)\n"
    )

    # Seed an initial turn if a file, prompt, or piped input was provided.
    initial = None
    if args.file:
        initial = read_file(args.file)
        if args.prompt or args.query:
            initial = f"{args.prompt or ' '.join(args.query)}\n\n{initial}"
    elif args.prompt:
        initial = args.prompt
    elif args.query:
        initial = " ".join(args.query)
    elif not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            initial = data.rstrip("\n")

    if initial is not None:
        send_turn(client, args, messages, model, model_explicit, initial)

    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write("\n")
            break
        text = line.strip()
        if not text:
            continue

        if text.startswith("/"):
            parts = text.split(maxsplit=1)
            cmd = parts[0].lower()
            rest = parts[1] if len(parts) > 1 else ""
            if cmd in ("/quit", "/exit", "/q"):
                break
            elif cmd == "/help":
                sys.stdout.write(_CHAT_HELP)
            elif cmd == "/clear":
                messages.clear()
                if args.system:
                    messages.append({"role": "system", "content": args.system})
                sys.stdout.write("[history cleared]\n")
            elif cmd == "/system":
                if not rest:
                    sys.stdout.write("usage: /system <text>\n")
                else:
                    messages[:] = [m for m in messages if m["role"] != "system"]
                    messages.insert(0, {"role": "system", "content": rest})
                    args.system = rest
                    sys.stdout.write("[system prompt set]\n")
            elif cmd == "/file":
                if not rest:
                    sys.stdout.write("usage: /file <path>\n")
                else:
                    file_text = read_file(rest.strip())
                    send_turn(client, args, messages, model, model_explicit, file_text)
            elif cmd == "/model":
                if not rest:
                    sys.stdout.write(f"[model] {model}\n")
                else:
                    model = rest.strip()
                    model_explicit = True
                    sys.stdout.write(f"[model] set to {model}\n")
            elif cmd == "/show":
                for m in messages:
                    sys.stdout.write(f"{m['role']}: {m['content']}\n")
            else:
                sys.stdout.write(f"unknown command: {cmd} (try /help)\n")
            continue

        send_turn(client, args, messages, model, model_explicit, text)

    return 0
```

Wait, `args.system = rest` — args is a Namespace; assigning is fine. But then `/clear` uses `args.system` to restore. Good.

Note: when seeding with file and there's also piped stdin — I only check stdin in the `elif not sys.stdin.isatty()` branch when no file/prompt/query. If `--file` given, stdin not consumed. That's fine (avoids blocking). Good.

One concern: when `--chat` is used without any input and stdin is a TTY, we skip seeding and go straight to the REPL. Good. But if stdin is piped (not a TTY) and user wants chat... e.g. `echo hi | nebul --chat`. My code reads all of stdin as initial in the `elif not sys.stdin.isatty()` branch. That sends "hi" as first turn, then the REPL `input()` would read from stdin too — but stdin is already consumed/EOF, so `input()` raises EOFError immediately -> break. So effectively it'd do one turn then exit. That's a reasonable behavior. Fine.

Now `_CHAT_HELP` constant.

Now `send_turn` as above. Let me make sure error handling removes the appended user message on failure so retry works. Actually, I append the user message at the start of send_turn. On exception, I pop it. Good. But if streaming partially succeeded and then errored, the partial content was printed but not appended to messages (assistant message not appended). And user message popped. So retry would re-send. But the partial output already printed to terminal. That's acceptable for error cases.

Hmm, but popping the user message means the printed partial response is "orphaned" and on retry the full response re-prints. Minor. Acceptable.

Now the single-turn `run_single_turn` (renamed from `chat`). Let me keep the function name `chat`? The original `chat` is the single-turn handler. To minimize confusion and changes, let me keep `chat` as single-turn and add `run_interactive`. Then `main`:

```python
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_models:
        ...
        return 0
    if args.chat:
        return run_interactive(args)
    chat(args)
    return 0
```

Now refactor `chat` to use `complete_once` and `print_usage`:

```python
def chat(args):
    api_key = os.environ.get("NEBUL_API_KEY", "")
    base_url = os.environ.get("NEBUL_BASE_URL", DEFAULT_BASE_URL)
    client = get_client(api_key, base_url)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": read_prompt(args)})

    model_explicit = args.model is not None
    model = args.model or os.environ.get("NEBUL_MODEL", DEFAULT_MODEL)

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": args.temperature,
    }
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens
    if args.json:
        kwargs["response_format"] = {"type": "json_object"}

    content, usage = complete_once(client, kwargs, args.no_stream, args.usage)

    if args.no_stream:
        if not content or not content.endswith("\n"):
            sys.stdout.write("\n")
    else:
        if content and not args.no_newline_on_stream:
            sys.stdout.write("\n")

    print_usage(usage, model, model_explicit, args.usage)
```

Wait, in the original non-stream path, usage is printed regardless of `args.usage`? No: original checks `if args.usage:`. So only when flag set. My `print_usage` respects that. Good.

But original non-stream: the newline logic was:
```
sys.stdout.write(content or "")
if not content or not content.endswith("\n"):
    sys.stdout.write("\n")
```
My `complete_once` non-stream writes `content` (or ""). Then I do `if not content or not content.endswith("\n"): sys.stdout.write("\n")`. Matches. Good.

Original streaming: `complete_once` -> `stream_completion` writes chunks. Then `if content and not args.no_newline_on_stream: write "\n"`. Original: `if saw_content and not args.no_newline_on_stream`. `saw_content` is True iff any content delta. My `content` is concatenation of deltas; truthy iff saw_content. Matches. Good.

Now the `--usage` default. Original default True. In chat mode, per-turn usage to stderr might be noisy but it's behind the flag (default True). For interactive chat, maybe default off is nicer. But changing default based on mode is inconsistent. Let me keep default True everywhere; user can `--no-usage`. Actually, for chat I think defaulting usage off is more pleasant. But the argparse default is shared. I could set: in `run_interactive`, if user didn't explicitly set usage... hard to know. Let me just keep usage on by default and let users pass `--no-usage`. Actually, I'll keep it simple and respect the flag as-is (default True). Hmm, but streaming usage per turn prints `[model]` and `[usage]` each turn — verbose. 

Let me reconsider: I'll make interactive chat default usage OFF unless `--usage` explicitly passed. To detect explicit, I can use a different sentinel. argparse: I can set default=None and then resolve. Let me set `--usage` default to None, `--no-usage` sets to False, and if None: for single-turn default True, for chat default False. That's a nice touch.

Implementation:
```python
parser.add_argument("--usage", action="store_true", default=None, ...)
parser.add_argument("--no-usage", action="store_false", dest="usage", ...)
```
Then `args.usage` is True, False, or None. Then:
- single-turn: `want_usage = True if args.usage is None else args.usage`
- chat: `want_usage = False if args.usage is None else args.usage`

Let me add a helper `resolve_usage(args, default)`.

Actually, let me keep it simpler and just keep default True for both. The verbosity in chat is acceptable and informative, and users can disable. But... I think defaulting off in chat is genuinely better UX. Let me do the sentinel approach; it's clean enough.

Let me define:
```python
def want_usage(args, interactive):
    if args.usage is None:
        return not interactive
    return args.usage
```
Then pass `want_usage(...)` to `print_usage` and `complete_once`/`stream_completion`.

Hmm, but `complete_once` takes `want_usage` only for streaming. And `print_usage` takes it. Let me thread a single `show_usage` boolean resolved once per mode.

Let me just resolve at top of each function:
- single-turn: `show_usage = True if args.usage is None else args.usage`
- chat: `show_usage = False if args.usage is None else args.usage`

And in `send_turn`, use `show_usage`. But `send_turn` is called from `run_interactive`; I'll capture `show_usage` as a closure or pass it. Let me pass it. Actually I'll define `send_turn` inside `run_interactive`? That would capture locals. But I also use it for the seeded turn. Let me make `send_turn` a module function taking a `show_usage` param. Wait it already takes `args`; I can compute inside. Let me have `send_turn` compute `show_usage` from args and a flag. Simpler: pass `show_usage` explicitly.

Let me restructure `send_turn(client, args, messages, model, model_explicit, user_text, show_usage)`.

And `complete_once(client, kwargs, no_stream, show_usage)`.

OK let me also update `print_usage(usage, model, model_explicit, show_usage)`.

Now the argument for `--file`:
```python
parser.add_argument("-f", "--file", default=None,
                    help="Read the contents of FILE and use it as the prompt "
                         "(combine with a query/prompt to add instructions).")
```

And `--chat`:
```python
parser.add_argument("-c", "--chat", action="store_true",
                    help="Start an interactive multi-turn chat session.")
```

Now `read_prompt` references `args.file`. Good. But in single-turn, `--file` works via read_prompt. In chat, I read file directly. Fine.

Now update epilog examples to mention `--file` and `--chat`.

Let me also update the docstring at top.

Now, the `read_file` is used both in read_prompt (single-turn) and run_interactive. Good, shared.

Let me also make sure `-c` doesn't conflict. Existing shorts: `-m`, `-s`, `-p`, `-f`? `-f` not used. `-c` not used. Good. But `--file` short `-f` — is `-f` used anywhere? No. Good. Actually, wait: argparse with `-f` and positional `query` — fine.

Hmm, one issue: `--json` plus `--chat` — response_format json on every turn. That's fine, user's choice.

Now, let me also consider: in interactive mode, `args.no_stream` — if user wants non-streaming chat, each turn is non-stream. Fine.

Edge: `complete_once` non-stream writes content then returns; in `send_turn` I then do `if not content.endswith("\n"): write "\n"`. For non-stream, `complete_once` wrote content (no trailing newline added). Good, then send_turn adds newline if missing. Good. For streaming, `stream_completion` wrote chunks, returns content; send_turn adds newline if missing. Good.

But in single-turn non-stream, I have separate logic: `if not content or not content.endswith("\n"): write "\n"`. The difference: single-turn adds newline even if content empty. In chat, if content empty, `not content.endswith("\n")` — empty string `.endswith("\n")` is False, so writes "\n". Same result. Actually for empty content, both add newline. Fine, consistent enough.

Wait, but should chat add a newline for empty assistant response? Rare. Fine.

Now let me write the whole file. Let me also double check the `read_prompt` for the `--file` + query combine in single-turn uses `args.prompt or " ".join(args.query)`. If both empty, `args.prompt` is None (falsy) -> `" ".join([])` = "" -> `"" or ...`? `None or ""` -> "". Then `f"{''}\n\n{file}"` = "\n\nfile". Hmm, leading blank line. Let me guard: only combine if there's an instruction.

```python
def read_prompt(args):
    if args.file:
        file_content = read_file(args.file)
        instruction = args.prompt
        if not instruction and args.query:
            instruction = " ".join(args.query)
        if instruction:
            return f"{instruction}\n\n{file_content}"
        return file_content
    ...
```

Good. And in run_interactive seeding:
```python
if args.file:
    initial = read_file(args.file)
    instruction = args.prompt
    if not instruction and args.query:
        instruction = " ".join(args.query)
    if instruction:
        initial = f"{instruction}\n\n{initial}"
```

Let me factor a helper `load_initial_message(args, allow_stdin=True)` to avoid duplication. Actually single-turn uses `read_prompt` which has its own stdin handling. Chat seeding has different stdin handling (only if no file/prompt/query). Let me just keep them separate; minor duplication is fine.

Actually, I realize for chat seeding with stdin, if I read stdin as initial and then enter REPL, input() will EOF. That's acceptable as discussed. But maybe better: if not a TTY and no other input, in chat mode, I should probably just not seed and let the user know. Hmm. But `echo hi | nebul --chat` is a reasonable way to ask one question in chat then continue? No, stdin exhausted. I think seeding from piped stdin in chat is fine and then it exits. Alternatively, I could skip stdin seeding in chat to avoid confusion. But then `echo hi | nebul --chat` would enter REPL reading from now-empty stdin -> immediate EOF -> just exits without doing anything. That's worse. So seeding is better. Keep it.

Let me write the final file now.

Let me also add a `--version`? Not requested. Skip.

Let me write it.

Actually, let me reconsider the `_CHAT_HELP` — put as module constant.

Let me also reconsider whether to print the model banner to stderr or stdout. Original usage goes to stderr. Banner to stderr keeps stdout clean for assistant output. Good, use stderr.

Now, the prompt `you> ` — use a constant. Fine.

Let me also handle the case where `input()` prompt should go to stdout (input writes prompt to stdout). In a terminal that's fine. For cleanliness, `input("you> ")` writes to stdout. That interleaves with assistant output. Acceptable for a CLI chat. Many CLIs do this.

Now let me write the code.

Let me also verify the openai library usage: `client.chat.completions.create(stream=True, **kwargs)` where kwargs has model, messages, temperature, maybe max_tokens, response_format, and stream_options added. For non-stream, no stream arg. Good. Note: when `stream=True` is passed positionally along with kwargs containing no `stream`, fine.

One more: in `stream_completion`, I do `kwargs = {**kwargs, "stream_options": {"include_usage": True}}` only if want_usage. Then `client.chat.completions.create(stream=True, **kwargs)`. Good.

Let me write the complete file now.

Let me also make sure I import what's needed: argparse, os, sys, OpenAI. Already imported.

Let me write it.

I'll read the file and then modify it to add an interactive multi-turn chat mode and file input support.

**Tool: write**

**Input:**
```json
{
  "content": "#!/usr/bin/env python3\n\"\"\"Generic Nebul inference API command-line client.\"\"\"\nimport argparse\nimport os\nimport sys\n\nfrom openai import OpenAI\n\nDEFAULT_BASE_URL = \"https://api.inference.nebul.io/v1\"\nDEFAULT_MODEL = \"openai/gpt-oss-120b\"\n\n_CHAT_HELP = (\n    \"Commands:\\n\"\n    \"  /help            show this help\\n\"\n    \"  /quit | /exit    exit the chat\\n\"\n    \"  /clear           clear conversation history (keeps system prompt)\\n\"\n    \"  /system <text>   set or replace the system prompt\\n\"\n    \"  /file <path>     send a file's contents as your next message\\n\"\n    \"  /model [id]      show or switch the current model\\n\"\n    \"  /show            print the current conversation\\n\"\n)\n\n\ndef get_client(api_key: str, base_url: str) -> OpenAI:\n    if not api_key:\n        sys.stderr.write(\n            \"error: NEBUL_API_KEY environment variable is not set.\\n\"\n            \"Get a key from https://studio.nebul.io and export it:\\n\"\n            \"  export NEBUL_API_KEY=sk-...\\n\"\n        )\n        sys.exit(2)\n    return OpenAI(api_key=api_key, base_url=base_url)\n\n\ndef list_models(client: OpenAI) -> None:\n    models = client.models.list()\n    ids = sorted(m.id for m in models.data)\n    for mid in ids:\n        print(mid)\n\n\ndef read_file(path: str) -> str:\n    try:\n        with open(path, \"r\", encoding=\"utf-8\") as fh:\n            return fh.read().rstrip(\"\\n\")\n    except OSError as exc:\n        sys.stderr.write(f\"error: could not read file {path!r}: {exc}\\n\")\n        sys.exit(2)\n\n\ndef read_prompt(args: argparse.Namespace) -> str:\n    if args.file:\n        file_content = read_file(args.file)\n        instruction = args.prompt\n        if not instruction and args.query:\n            instruction = \" \".join(args.query)\n        if instruction:\n            return f\"{instruction}\\n\\n{file_content}\"\n        return file_content\n    if args.prompt:\n        return args.prompt\n    if not sys.stdin.isatty():\n        data = sys.stdin.read()\n        if data.strip():\n            return data.rstrip(\"\\n\")\n    if args.query:\n        return \" \".join(args.query)\n    sys.stderr.write(\n        \"error: no prompt provided. Pass a query, --file, --prompt, or pipe via stdin.\\n\"\n    )\n    sys.exit(2)\n\n\ndef stream_completion(client: OpenAI, kwargs: dict, show_usage: bool):\n    if show_usage:\n        kwargs = {**kwargs, \"stream_options\": {\"include_usage\": True}}\n    stream = client.chat.completions.create(stream=True, **kwargs)\n    parts = []\n    usage = None\n    for chunk in stream:\n        if chunk.usage:\n            usage = chunk.usage\n        if not chunk.choices:\n            continue\n        delta = chunk.choices[0].delta\n        if delta and delta.content:\n            parts.append(delta.content)\n            sys.stdout.write(delta.content)\n            sys.stdout.flush()\n    return \"\".join(parts), usage\n\n\ndef complete_once(client: OpenAI, kwargs: dict, no_stream: bool, show_usage: bool):\n    if no_stream:\n        response = client.chat.completions.create(**kwargs)\n        content = response.choices[0].message.content or \"\"\n        sys.stdout.write(content)\n        return content, response.usage\n    return stream_completion(client, kwargs, show_usage)\n\n\ndef print_usage(usage, model: str, model_explicit: bool, show_usage: bool) -> None:\n    if not show_usage:\n        return\n    if not model_explicit:\n        sys.stderr.write(f\"[model] {model}\\n\")\n    if usage:\n        sys.stderr.write(\n            f\"[usage] prompt={usage.prompt_tokens} \"\n            f\"completion={usage.completion_tokens} total={usage.total_tokens}\\n\"\n        )\n\n\ndef resolve_usage(args: argparse.Namespace, interactive: bool) -> bool:\n    if args.usage is None:\n        return not interactive\n    return args.usage\n\n\ndef base_kwargs(model: str, messages: list, args: argparse.Namespace) -> dict:\n    kwargs = {\n        \"model\": model,\n        \"messages\": messages,\n        \"temperature\": args.temperature,\n    }\n    if args.max_tokens is not None:\n        kwargs[\"max_tokens\"] = args.max_tokens\n    if args.json:\n        kwargs[\"response_format\"] = {\"type\": \"json_object\"}\n    return kwargs\n\n\ndef chat(args: argparse.Namespace) -> None:\n    api_key = os.environ.get(\"NEBUL_API_KEY\", \"\")\n    base_url = os.environ.get(\"NEBUL_BASE_URL\", DEFAULT_BASE_URL)\n    client = get_client(api_key, base_url)\n\n    messages = []\n    if args.system:\n        messages.append({\"role\": \"system\", \"content\": args.system})\n    messages.append({\"role\": \"user\", \"content\": read_prompt(args)})\n\n    model_explicit = args.model is not None\n    model = args.model or os.environ.get(\"NEBUL_MODEL\", DEFAULT_MODEL)\n    show_usage = resolve_usage(args, interactive=False)\n\n    kwargs = base_kwargs(model, messages, args)\n    content, usage = complete_once(client, kwargs, args.no_stream, show_usage)\n\n    if args.no_stream:\n        if not content or not content.endswith(\"\\n\"):\n            sys.stdout.write(\"\\n\")\n    else:\n        if content and not args.no_newline_on_stream:\n            sys.stdout.write(\"\\n\")\n\n    print_usage(usage, model, model_explicit, show_usage)\n\n\ndef send_turn(client: OpenAI, args: argparse.Namespace, messages: list,\n              model: str, model_explicit: bool, show_usage: bool,\n              user_text: str) -> None:\n    messages.append({\"role\": \"user\", \"content\": user_text})\n    kwargs = base_kwargs(model, messages, args)\n    try:\n        content, usage = complete_once(client, kwargs, args.no_stream, show_usage)\n    except Exception as exc:  # noqa: BLE001 - surface API errors to the user\n        sys.stderr.write(f\"\\n[error] {exc}\\n\")\n        messages.pop()\n        return\n    if not content.endswith(\"\\n\"):\n        sys.stdout.write(\"\\n\")\n    print_usage(usage, model, model_explicit, show_usage)\n    messages.append({\"role\": \"assistant\", \"content\": content})\n\n\ndef run_interactive(args: argparse.Namespace) -> int:\n    api_key = os.environ.get(\"NEBUL_API_KEY\", \"\")\n    base_url = os.environ.get(\"NEBUL_BASE_URL\", DEFAULT_BASE_URL)\n    client = get_client(api_key, base_url)\n\n    model_explicit = args.model is not None\n    model = args.model or os.environ.get(\"NEBUL_MODEL\", DEFAULT_MODEL)\n    show_usage = resolve_usage(args, interactive=True)\n\n    messages: list = []\n    if args.system:\n        messages.append({\"role\": \"system\", \"content\": args.system})\n\n    sys.stderr.write(\n        f\"nebul chat - model: {model} \"\n        f\"(type /help for commands, Ctrl+D or /quit to exit)\\n\"\n    )\n\n    # Seed an initial turn if a file, prompt, or piped input was provided.\n    initial = None\n    if args.file:\n        initial = read_file(args.file)\n        instruction = args.prompt\n        if not instruction and args.query:\n            instruction = \" \".join(args.query)\n        if instruction:\n            initial = f\"{instruction}\\n\\n{initial}\"\n    elif args.prompt:\n        initial = args.prompt\n    elif args.query:\n        initial = \" \".join(args.query)\n    elif not sys.stdin.isatty():\n        data = sys.stdin.read()\n        if data.strip():\n            initial = data.rstrip(\"\\n\")\n\n    if initial is not None:\n        send_turn(client, args, messages, model, model_explicit, show_usage, initial)\n\n    while True:\n        try:\n            line = input(\"you> \")\n        except (EOFError, KeyboardInterrupt):\n            sys.stdout.write(\"\\n\")\n            break\n        text = line.strip()\n        if not text:\n            continue\n\n        if text.startswith(\"/\"):\n            parts = text.split(maxsplit=1)\n            cmd = parts[0].lower()\n            rest = parts[1] if len(parts) > 1 else \"\"\n            if cmd in (\"/quit\", \"/exit\", \"/q\"):\n                break\n            elif cmd == \"/help\":\n                sys.stdout.write(_CHAT_HELP)\n            elif cmd == \"/clear\":\n                messages.clear()\n                if args.system:\n                    messages.append({\"role\": \"system\", \"content\": args.system})\n                sys.stdout.write(\"[history cleared]\\n\")\n            elif cmd == \"/system\":\n                if not rest:\n                    sys.stdout.write(\"usage: /system <text>\\n\")\n                else:\n                    messages[:] = [m for m in messages if m[\"role\"] != \"system\"]\n                    messages.insert(0, {\"role\": \"system\", \"content\": rest})\n                    args.system = rest\n                    sys.stdout.write(\"[system prompt set]\\n\")\n            elif cmd == \"/file\":\n                if not rest:\n                    sys.stdout.write(\"usage: /file <path>\\n\")\n                else:\n                    file_text = read_file(rest.strip())\n                    send_turn(client, args, messages, model, model_explicit,\n                              show_usage, file_text)\n            elif cmd == \"/model\":\n                if not rest:\n                    sys.stdout.write(f\"[model] {model}\\n\")\n                else:\n                    model = rest.strip()\n                    model_explicit = True\n                    sys.stdout.write(f\"[model] set to {model}\\n\")\n            elif cmd == \"/show\":\n                for m in messages:\n                    sys.stdout.write(f\"{m['role']}: {m['content']}\\n\")\n            else:\n                sys.stdout.write(f\"unknown command: {cmd} (try /help)\\n\")\n            continue\n\n        send_turn(client, args, messages, model, model_explicit, show_usage, text)\n\n    return 0\n\n\ndef build_parser() -> argparse.ArgumentParser:\n    parser = argparse.ArgumentParser(\n        prog=\"nebul\",\n        description=\"Query Nebul inference models from the command line.\",\n        formatter_class=argparse.RawDescriptionHelpFormatter,\n        epilog=(\n            \"Examples:\\n\"\n            \"  nebul 'Why is privacy important in the age of AI?'\\n\"\n            \"  nebul -m deepseek-ai/deepseek-r1 'Explain CAP theorem'\\n\"\n            \"  echo 'summarize this' | nebul\\n\"\n            \"  nebul --system 'You are a haiku bot' 'write one about cats'\\n\"\n            \"  nebul -f notes.txt 'summarize this'\\n\"\n            \"  nebul --chat\\n\"\n            \"  nebul --chat --system 'You are a code reviewer' -f main.py\\n\"\n            \"  nebul --no-usage 'just give me the answer'\\n\"\n            \"  nebul --list-models\\n\"\n            \"  NEBUL_MODEL=mistralai/mistral-large nebul 'hello'\\n\"\n            \"\\n\"\n            \"Environment:\\n\"\n            \"  NEBUL_API_KEY      Required. Your API key (sk-...).\\n\"\n            \"  NEBUL_MODEL        Default model if --model is not given.\\n\"\n            \"  NEBUL_BASE_URL     Override the API base URL.\\n\"\n        ),\n    )\n    parser.add_argument(\n        \"query\",\n        nargs=\"*\",\n        help=\"The prompt text. Multiple words are joined. May be omitted if \"\n        \"piped via stdin, or if --prompt/--file/--chat is used.\",\n    )\n    parser.add_argument(\"-m\", \"--model\", default=None,\n                        help=f\"Model id (default: {DEFAULT_MODEL} or $NEBUL_MODEL)\")\n    parser.add_argument(\"-s\", \"--system\", help=\"Optional system prompt.\")\n    parser.add_argument(\"-p\", \"--prompt\", help=\"Prompt text (alternative to positional).\")\n    parser.add_argument(\"-f\", \"--file\", default=None,\n                        help=\"Read FILE and use its contents as the prompt. \"\n                        \"Combine with a query/prompt to prepend instructions.\")\n    parser.add_argument(\"-c\", \"--chat\", action=\"store_true\",\n                        help=\"Start an interactive multi-turn chat session.\")\n    parser.add_argument(\"--temperature\", type=float, default=0.7, help=\"Sampling temperature.\")\n    parser.add_argument(\"--max-tokens\", type=int, default=None, help=\"Max tokens to generate.\")\n    parser.add_argument(\"--json\", action=\"store_true\", help=\"Request JSON-formatted output.\")\n    parser.add_argument(\"--no-stream\", action=\"store_true\", help=\"Disable streaming output.\")\n    parser.add_argument(\"--no-newline-on-stream\", action=\"store_true\",\n                        help=\"Do not append a trailing newline after streaming output.\")\n    parser.add_argument(\"--usage\", action=\"store_true\", default=None,\n                        help=\"Print token usage to stderr after each query \"\n                        \"(default: on for single queries, off in --chat).\")\n    parser.add_argument(\"--no-usage\", action=\"store_false\", dest=\"usage\",\n                        help=\"Do not print token usage.\")\n    parser.add_argument(\"--list-models\", action=\"store_true\", help=\"List available models and exit.\")\n    return parser\n\n\ndef main(argv=None) -> int:\n    args = build_parser().parse_args(argv)\n    if args.list_models:\n        api_key = os.environ.get(\"NEBUL_API_KEY\", \"\")\n        base_url = os.environ.get(\"NEBUL_BASE_URL\", DEFAULT_BASE_URL)\n        list_models(get_client(api_key, base_url))\n        return 0\n    if args.chat:\n        return run_interactive(args)\n    chat(args)\n    return 0\n\n\nif __name__ == \"__main__\":\n    raise SystemExit(main())\n",
  "filePath": "/home/ewald/OpenCode/nebul-cli/nebul.py"
}
```

**Output:**
```
Wrote file successfully.
```

---

## Assistant (Build · GLM 5.2 FP8 · 2.7s)

Let me verify it compiles and the help output looks right:

**Tool: bash**

**Input:**
```json
{
  "command": "python3 -m py_compile nebul.py && echo \"compile OK\"",
  "workdir": "/home/ewald/OpenCode/nebul-cli"
}
```

**Output:**
```
compile OK

```

**Tool: bash**

**Input:**
```json
{
  "command": "python3 nebul.py --help",
  "workdir": "/home/ewald/OpenCode/nebul-cli"
}
```

**Output:**
```
usage: nebul [-h] [-m MODEL] [-s SYSTEM] [-p PROMPT] [-f FILE] [-c]
             [--temperature TEMPERATURE] [--max-tokens MAX_TOKENS] [--json]
             [--no-stream] [--no-newline-on-stream] [--usage] [--no-usage]
             [--list-models]
             [query ...]

Query Nebul inference models from the command line.

positional arguments:
  query                 The prompt text. Multiple words are joined. May be
                        omitted if piped via stdin, or if
                        --prompt/--file/--chat is used.

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model id (default: openai/gpt-oss-120b or
                        $NEBUL_MODEL)
  -s SYSTEM, --system SYSTEM
                        Optional system prompt.
  -p PROMPT, --prompt PROMPT
                        Prompt text (alternative to positional).
  -f FILE, --file FILE  Read FILE and use its contents as the prompt. Combine
                        with a query/prompt to prepend instructions.
  -c, --chat            Start an interactive multi-turn chat session.
  --temperature TEMPERATURE
                        Sampling temperature.
  --max-tokens MAX_TOKENS
                        Max tokens to generate.
  --json                Request JSON-formatted output.
  --no-stream           Disable streaming output.
  --no-newline-on-stream
                        Do not append a trailing newline after streaming
                        output.
  --usage               Print token usage to stderr after each query (default:
                        on for single queries, off in --chat).
  --no-usage            Do not print token usage.
  --list-models         List available models and exit.

Examples:
  nebul 'Why is privacy important in the age of AI?'
  nebul -m deepseek-ai/deepseek-r1 'Explain CAP theorem'
  echo 'summarize this' | nebul
  nebul --system 'You are a haiku bot' 'write one about cats'
  nebul -f notes.txt 'summarize this'
  nebul --chat
  nebul --chat --system 'You are a code reviewer' -f main.py
  nebul --no-usage 'just give me the answer'
  nebul --list-models
  NEBUL_MODEL=mistralai/mistral-large nebul 'hello'

Environment:
  NEBUL_API_KEY      Required. Your API key (sk-...).
  NEBUL_MODEL        Default model if --model is not given.
  NEBUL_BASE_URL     Override the API base URL.

```

---

## Assistant (Build · GLM 5.2 FP8 · 2.2s)

_Thinking:_

The program compiles and the help looks good. Let me also quickly test the file-reading error path and the no-prompt error path to make sure argument handling works without an API key.

Let me verify the argument-handling paths that don't need an API key:

**Tool: bash**

**Input:**
```json
{
  "command": "python3 nebul.py --file /nonexistent.txt 2>&1; echo \"exit=$?\"; echo \"---\"; python3 nebul.py 2>&1; echo \"exit=$?\"",
  "workdir": "/home/ewald/OpenCode/nebul-cli"
}
```

**Output:**
```
error: could not read file '/nonexistent.txt': [Errno 2] No such file or directory: '/nonexistent.txt'
exit=2
---
error: no prompt provided. Pass a query, --file, --prompt, or pipe via stdin.
exit=2

```

---

## Assistant (Build · GLM 5.2 FP8 · 2.3s)

Done. The modified `nebul.py` now supports two new features:

**File input (`-f`/`--file`)**
- `nebul -f notes.txt` — uses the file contents as the prompt.
- `nebul -f notes.txt "summarize this"` — prepends an instruction to the file contents (also works with `--prompt`).
- `read_file` (nebul.py:33) handles errors; `read_prompt` (nebul.py:43) integrates it into the single-turn path.

**Interactive multi-turn chat (`-c`/`--chat`)**
- Maintains conversation history across turns with streaming per response.
- REPL commands: `/help`, `/quit`/`/exit`, `/clear`, `/system <text>`, `/file <path>`, `/model [id]`, `/show`.
- Seeds the first turn from `--file`, `--prompt`, positional query, or piped stdin.
- Ctrl+D / Ctrl+C exits cleanly.

I also refactored the completion logic into shared helpers (`complete_once`, `stream_completion`, `base_kwargs`, `print_usage`) so single-turn and chat modes reuse the same code, and made `--usage` default off in chat mode but on for single queries. The original single-turn behavior (streaming, newlines, usage reporting) is preserved.

Note: I couldn't run a live API call since `NEBUL_API_KEY` isn't set here, but the file compiles and the argument/error paths work.

---

