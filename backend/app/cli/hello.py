import os

from google import genai


def main() -> None:
    # genai.Client() reads GEMINI_API_KEY from the environment.
    client = genai.Client()
    model = os.environ.get("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")

    response = client.models.generate_content(
        model=model,
        contents="Explain RAG in one sentence.",
    )
    print(f"model: {model}")
    print(response.text)


if __name__ == "__main__":
    main()
