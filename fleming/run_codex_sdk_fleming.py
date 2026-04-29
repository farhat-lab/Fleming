import argparse
import os

from codex_sdk import FlemingCodexSDK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the codex_sdk Fleming agent.")
    parser.add_argument("--config", default="config.yaml", help="Path to Fleming config.")
    parser.add_argument("--message", help="Single prompt to run.")
    parser.add_argument(
        "--optimization-strategy",
        default="auto",
        choices=["auto", "immediate", "synspace", "llm"],
        help="Default optimization backend.",
    )
    parser.add_argument("--model", default="gpt-5.4", help="OpenAI model for the SDK agent.")
    parser.add_argument("--max-turns", type=int, default=20, help="Max SDK agent turns.")
    parser.add_argument("--interactive", action="store_true", help="Run a basic REPL.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)

    agent = FlemingCodexSDK(
        config_path=config_path,
        model=args.model,
        optimization_strategy=args.optimization_strategy,
    )

    if args.interactive:
        while True:
            try:
                message = input("Fleming> ").strip()
            except EOFError:
                break
            if not message or message.lower() in {"quit", "exit"}:
                break
            result = agent.run(
                message,
                optimization_strategy=args.optimization_strategy,
                max_turns=args.max_turns,
            )
            print(result.final_output)
    elif args.message:
        result = agent.run(
            args.message,
            optimization_strategy=args.optimization_strategy,
            max_turns=args.max_turns,
        )
        print(result.final_output)
    else:
        raise SystemExit("Pass --message or use --interactive.")


if __name__ == "__main__":
    main()
