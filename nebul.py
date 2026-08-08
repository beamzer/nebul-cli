#!/usr/bin/env python3
"""Generic Nebul inference API command-line client."""
import argparse
import os
import sys

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.inference.nebul.io/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

_CHAT_HELP = (
    "Commands:\n"
    "  /help            show this help\n"
    "  /quit | /exit    exit the chat\n"
    "  /clear           clear conversation history (keeps system prompt)\n"
    "  /system <text>   set or replace the system prompt\n"
    "  /file <path>     send a file's contents as your next message\n"
    "  /model [id]      show or switch the current model\n"
    "  /show            print the current conversation\n"
)


def get_client(api_key: str, base_url: str) -> OpenAI:
    if not api_key:
        sys.stderr.write(
            "error: NEBUL_API_KEY environment variable is not set.\n"
            "Get a key from https://studio.nebul.io and export it:\n"
            "  export NEBUL_API_KEY=sk-...\n"
        )
        sys.exit(2)
    return OpenAI(api_key=api_key, base_url=base_url)


def list_models(client: OpenAI) -> None:
    models = client.models.list()
    ids = sorted(m.id for m in models.data)
    for mid in ids:
        print(mid)


def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().rstrip("\n")
    except OSError as exc:
        sys.stderr.write(f"error: could not read file {path!r}: {exc}\n")
        sys.exit(2)


def read_prompt(args: argparse.Namespace) -> str:
    if args.file:
        file_content = read_file(args.file)
        instruction = args.prompt
        if not instruction and args.query:
            instruction = " ".join(args.query)
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
    sys.stderr.write(
        "error: no prompt provided. Pass a query, --file, --prompt, or pipe via stdin.\n"
    )
    sys.exit(2)


def stream_completion(client: OpenAI, kwargs: dict, show_usage: bool):
    if show_usage:
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


def complete_once(client: OpenAI, kwargs: dict, no_stream: bool, show_usage: bool):
    if no_stream:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        sys.stdout.write(content)
        return content, response.usage
    return stream_completion(client, kwargs, show_usage)


def print_usage(usage, model: str, model_explicit: bool, show_usage: bool) -> None:
    if not show_usage:
        return
    if not model_explicit:
        sys.stderr.write(f"[model] {model}\n")
    if usage:
        sys.stderr.write(
            f"[usage] prompt={usage.prompt_tokens} "
            f"completion={usage.completion_tokens} total={usage.total_tokens}\n"
        )


def resolve_usage(args: argparse.Namespace, interactive: bool) -> bool:
    if args.usage is None:
        return not interactive
    return args.usage


def base_kwargs(model: str, messages: list, args: argparse.Namespace) -> dict:
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": args.temperature,
    }
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens
    if args.json:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def chat(args: argparse.Namespace) -> None:
    api_key = os.environ.get("NEBUL_API_KEY", "")
    base_url = os.environ.get("NEBUL_BASE_URL", DEFAULT_BASE_URL)
    client = get_client(api_key, base_url)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": read_prompt(args)})

    model_explicit = args.model is not None
    model = args.model or os.environ.get("NEBUL_MODEL", DEFAULT_MODEL)
    show_usage = resolve_usage(args, interactive=False)

    kwargs = base_kwargs(model, messages, args)
    content, usage = complete_once(client, kwargs, args.no_stream, show_usage)

    if args.no_stream:
        if not content or not content.endswith("\n"):
            sys.stdout.write("\n")
    else:
        if content and not args.no_newline_on_stream:
            sys.stdout.write("\n")

    print_usage(usage, model, model_explicit, show_usage)


def send_turn(client: OpenAI, args: argparse.Namespace, messages: list,
              model: str, model_explicit: bool, show_usage: bool,
              user_text: str) -> None:
    messages.append({"role": "user", "content": user_text})
    kwargs = base_kwargs(model, messages, args)
    try:
        content, usage = complete_once(client, kwargs, args.no_stream, show_usage)
    except Exception as exc:  # noqa: BLE001 - surface API errors to the user
        sys.stderr.write(f"\n[error] {exc}\n")
        messages.pop()
        return
    if not content.endswith("\n"):
        sys.stdout.write("\n")
    print_usage(usage, model, model_explicit, show_usage)
    messages.append({"role": "assistant", "content": content})


def run_interactive(args: argparse.Namespace) -> int:
    api_key = os.environ.get("NEBUL_API_KEY", "")
    base_url = os.environ.get("NEBUL_BASE_URL", DEFAULT_BASE_URL)
    client = get_client(api_key, base_url)

    model_explicit = args.model is not None
    model = args.model or os.environ.get("NEBUL_MODEL", DEFAULT_MODEL)
    show_usage = resolve_usage(args, interactive=True)

    messages: list = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    sys.stderr.write(
        f"nebul chat - model: {model} "
        f"(type /help for commands, Ctrl+D or /quit to exit)\n"
    )

    # Seed an initial turn if a file, prompt, or piped input was provided.
    initial = None
    if args.file:
        initial = read_file(args.file)
        instruction = args.prompt
        if not instruction and args.query:
            instruction = " ".join(args.query)
        if instruction:
            initial = f"{instruction}\n\n{initial}"
    elif args.prompt:
        initial = args.prompt
    elif args.query:
        initial = " ".join(args.query)
    elif not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            initial = data.rstrip("\n")

    if initial is not None:
        send_turn(client, args, messages, model, model_explicit, show_usage, initial)

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
                    send_turn(client, args, messages, model, model_explicit,
                              show_usage, file_text)
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

        send_turn(client, args, messages, model, model_explicit, show_usage, text)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nebul",
        description="Query Nebul inference models from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  nebul 'Why is privacy important in the age of AI?'\n"
            "  nebul -m deepseek-ai/deepseek-r1 'Explain CAP theorem'\n"
            "  echo 'summarize this' | nebul\n"
            "  nebul --system 'You are a haiku bot' 'write one about cats'\n"
            "  nebul -f notes.txt 'summarize this'\n"
            "  nebul --chat\n"
            "  nebul --chat --system 'You are a code reviewer' -f main.py\n"
            "  nebul --no-usage 'just give me the answer'\n"
            "  nebul --list-models\n"
            "  NEBUL_MODEL=mistralai/mistral-large nebul 'hello'\n"
            "\n"
            "Environment:\n"
            "  NEBUL_API_KEY      Required. Your API key (sk-...).\n"
            "  NEBUL_MODEL        Default model if --model is not given.\n"
            "  NEBUL_BASE_URL     Override the API base URL.\n"
        ),
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="The prompt text. Multiple words are joined. May be omitted if "
        "piped via stdin, or if --prompt/--file/--chat is used.",
    )
    parser.add_argument("-m", "--model", default=None,
                        help=f"Model id (default: {DEFAULT_MODEL} or $NEBUL_MODEL)")
    parser.add_argument("-s", "--system", help="Optional system prompt.")
    parser.add_argument("-p", "--prompt", help="Prompt text (alternative to positional).")
    parser.add_argument("-f", "--file", default=None,
                        help="Read FILE and use its contents as the prompt. "
                        "Combine with a query/prompt to prepend instructions.")
    parser.add_argument("-c", "--chat", action="store_true",
                        help="Start an interactive multi-turn chat session.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens to generate.")
    parser.add_argument("--json", action="store_true", help="Request JSON-formatted output.")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output.")
    parser.add_argument("--no-newline-on-stream", action="store_true",
                        help="Do not append a trailing newline after streaming output.")
    parser.add_argument("--usage", action="store_true", default=None,
                        help="Print token usage to stderr after each query "
                        "(default: on for single queries, off in --chat).")
    parser.add_argument("--no-usage", action="store_false", dest="usage",
                        help="Do not print token usage.")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_models:
        api_key = os.environ.get("NEBUL_API_KEY", "")
        base_url = os.environ.get("NEBUL_BASE_URL", DEFAULT_BASE_URL)
        list_models(get_client(api_key, base_url))
        return 0
    if args.chat:
        return run_interactive(args)
    chat(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
