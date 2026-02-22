import argparse

from openai import APIConnectionError, AuthenticationError, RateLimitError

from .core.chat_service import ChatService
from .core.config import DEFAULT_MODEL, KNOWLEDGE_DIR
from .core.llm_wrapper import LLMWrapper
from .core.retriever import load_knowledge_chunks


SMALL_TALK = {"hi", "hello", "hey", "yo", "sup", "what up"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trent GPT - note grounded chat assistant")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = load_knowledge_chunks(KNOWLEDGE_DIR)
    if not chunks:
        raise SystemExit(f"No .txt files found in {KNOWLEDGE_DIR}")

    llm = LLMWrapper(model=args.model)
    chat = ChatService(llm=llm, chunks=chunks)

    print("Trent GPT running. Type 'exit' to quit.")
    print(f"Loaded {len(chunks)} chunks from {KNOWLEDGE_DIR}\n")

    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not user:
            continue
        if user.lower() in SMALL_TALK:
            print("\nAssistant: Hey. Ask me anything from your notes.\n")
            continue

        try:
            answer, hits, _, _ = chat.ask(user, use_web_search=True)
        except AuthenticationError:
            print(
                "\nAssistant: OpenAI authentication failed (invalid API key). "
                "Update OPENAI_API_KEY in .env and restart.\n"
            )
            continue
        except RateLimitError:
            print(
                "\nAssistant: OpenAI quota/rate limit reached. "
                "Check billing/usage and try again.\n"
            )
            continue
        except APIConnectionError:
            print(
                "\nAssistant: Could not reach OpenAI API (network issue). "
                "Check your internet and try again.\n"
            )
            continue

        print(f"\nAssistant: {answer}\n")

        if hits:
            print("Retrieved context:")
            for score, source, _ in hits:
                print(f"- {source} (score={score:.3f})")
            print("")


if __name__ == "__main__":
    main()
