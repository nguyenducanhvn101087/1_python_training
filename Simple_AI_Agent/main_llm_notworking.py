import json
import requests
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Optional: load .env if you need environment variables
from dotenv import load_dotenv
load_dotenv()

VLLM_URL = "http://localhost:8000/v1/completions"  # vLLM server URL

# ------------------------
# Custom LLM wrapper for vLLM
# ------------------------
class VLLMClient:
    def __init__(self, server_url=VLLM_URL, model_name="meta-llama/Llama-2-7b-hf"):
        self.server_url = server_url
        self.model_name = model_name

    def generate(self, prompt, max_tokens=128, temperature=0):
        """
        Sends the prompt to the vLLM server and returns the generated text.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        response = requests.post(self.server_url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        # vLLM returns a list of completions
        return data["completions"][0]["text"]

# ------------------------
# Main function
# ------------------------
def main():
    llm = VLLMClient()
    print("Welcome to the AI Agent Calculator! Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        print("\nAssistant: ", end="")
        output = llm.generate(prompt=user_input, max_tokens=128, temperature=0)
        print(output)


if __name__ == "__main__":
    main()
