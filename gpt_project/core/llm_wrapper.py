import os
import time

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError


load_dotenv(override=True)


class LLMWrapper:
    def __init__(self, model: str):
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip().strip('"').strip("'")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
        self.client = OpenAI(api_key=api_key, timeout=45.0, max_retries=2)
        self.model = model

    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 1500) -> str:
        last_error = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = (response.choices[0].message.content or "").strip()
                if content:
                    return content
                return "I couldn't generate a full answer. Please try again."
            except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                raise

        if last_error:
            raise last_error
        return "I couldn't generate a response."

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "medium",
    ) -> str:
        response = self.client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=size,
            quality=quality,
        )
        b64 = response.data[0].b64_json
        if not b64:
            raise ValueError("Image generation returned no image data.")
        return f"data:image/png;base64,{b64}"

    def analyze_image(self, image_data_url: str, instruction: str | None = None) -> str:
        prompt = instruction or (
            "Analyze this image. Describe key visual details, visible text, and useful context for follow-up chat."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=400,
        )
        content = (response.choices[0].message.content or "").strip()
        return content or "Image uploaded. No detailed analysis was produced."
