import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

nemotron_key = os.getenv("NEMOTRON_API_KEY")

if not nemotron_key:
    print("No NEMOTRON_API_KEY found.")
    exit(1)

client = OpenAI(
    api_key=nemotron_key,
    base_url="https://openrouter.ai/api/v1"
)

SYSTEM_MESSAGE = (
    "You are the Chief Risk Officer.\n"
    "Evaluate ONLY financial capacity to repay.\n"
)

user_prompt = (
    "Sub-agent reports:\n"
    "CREDITANALYST: rec=APPROVE conf=0.8 flags=[]\n\n"
    "Respond ONLY:\n"
    "DECISION: <APPROVE|REJECT|REQUEST_MORE_INFO>\n"
    "CONFIDENCE: <0.0-1.0>\n"
    "REASONING: <2 sentences>"
)

try:
    response = client.chat.completions.create(
        model="nvidia/nemotron-4-340b-instruct:free",
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=200,
    )
    text = response.choices[0].message.content.strip()
    print("=== RAW LLM RESPONSE ===")
    print(text)
    print("========================")

    conf_match = re.search(r"CONFIDENCE[^\w]*([0-9.]+)(%?)", text, re.IGNORECASE)
    print(f"Old Regex Match: {conf_match}")

    new_conf_match = re.search(r"CONFIDENCE.*?([0-9.]+)", text, re.IGNORECASE)
    print(f"New Regex Match: {new_conf_match}")

except Exception as e:
    print(f"API Error: {e}")
