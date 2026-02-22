import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env next to this file
load_dotenv(Path(__file__).with_name(".env"))


# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set. Set it and try again.")

client = OpenAI(api_key=api_key)

# -----------------------
# System Prompt for Code Assistant
# -----------------------
SYSTEM_PROMPT = """You are an expert Python code assistant. Your tasks:
1. Answer coding questions and provide best practices
2. Generate clean, well-documented code snippets
3. Explain code and help debug issues
4. Recommend libraries and design patterns

Keep responses concise and focused on the user's specific problem."""

# -----------------------
# Chat with your GPT
# -----------------------
def chat(user_message: str, model: str = "gpt-3.5-turbo", temperature: float = 0.7) -> str:
    """
    Send a message to OpenAI and get a response.
    
    Args:
        user_message: The user's question or request
        model: OpenAI model to use (gpt-3.5-turbo, gpt-4, etc.)
        temperature: Creativity level (0.0=deterministic, 1.0=creative)
    
    Returns:
        The assistant's response
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=temperature,
        max_tokens=1000
    )
    return response.choices[0].message.content


def chat_with_history(messages: list, model: str = "gpt-3.5-turbo", temperature: float = 0.7) -> str:
    """
    Chat with multi-turn conversation history.
    
    Args:
        messages: List of dicts with 'role' ('user' or 'assistant') and 'content' keys
        model: OpenAI model to use
        temperature: Creativity level
    
    Returns:
        The assistant's response
    """
    # Add system prompt to the beginning if not already there
    if messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1000
    )
    return response.choices[0].message.content


# -----------------------
# Prepare training data for fine-tuning (optional)
# -----------------------
def prepare_training_data(examples: list, output_file: str = "training_data.jsonl") -> str:
    """
    Convert examples to JSONL format for fine-tuning.
    
    Args:
        examples: List of dicts with 'prompt' and 'completion' keys
        output_file: Output JSONL file path
    
    Returns:
        Path to the prepared training file
    """
    with open(output_file, "w") as f:
        for example in examples:
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["completion"]}
                ]
            }
            f.write(json.dumps(record) + "\n")
    
    print(f"Training data saved to {output_file}")
    return output_file


# -----------------------
# Simple interactive chat
# -----------------------
def interactive_chat():
    """Run an interactive chat session with your code assistant."""
    print("🤖 Code Assistant (type 'exit' to quit)")
    print("-" * 50)
    
    conversation_history = []
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Add user message to history
        conversation_history.append({"role": "user", "content": user_input})
        
        # Get response
        response = chat_with_history(conversation_history)
        print(f"\nAssistant: {response}")
        
        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": response})


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # Example 1: Simple question
    print("Example 1: Simple code question")
    print("-" * 50)
    response = chat("How do I read a CSV file in Python?")
    print(response)
    
    print("\n" + "=" * 50 + "\n")
    
    # Example 2: Multi-turn conversation
    print("Example 2: Multi-turn conversation")
    print("-" * 50)
    messages = [
        {"role": "user", "content": "What's the best way to validate user input in Python?"}
    ]
    response = chat_with_history(messages)
    print(f"Q: What's the best way to validate user input in Python?\nA: {response}")
    
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": "Can you show me an example with regex?"})
    response = chat_with_history(messages)
    print(f"\nQ: Can you show me an example with regex?\nA: {response}")
    
    print("\n" + "=" * 50 + "\n")
    
    # Example 3: Training data prep (if you want to fine-tune later)
    print("Example 3: Preparing training data")
    print("-" * 50)
    training_examples = [
        {
            "prompt": "How do I sort a list of dictionaries by a key?",
            "completion": "Use the `sorted()` function with a `key` parameter: `sorted(list_of_dicts, key=lambda x: x['key_name'])`"
        },
        {
            "prompt": "What's the difference between == and is in Python?",
            "completion": "`==` checks value equality, `is` checks object identity (same memory address). Use `==` for value comparison and `is` for checking None/True/False."
        }
    ]
    prepare_training_data(training_examples)
    
    print("\n" + "=" * 50 + "\n")
    
    # Uncomment below to run interactive chat instead:
    interactive_chat()
