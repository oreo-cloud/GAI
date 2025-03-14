from openai import OpenAI
import os
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

OpenAI_api_key = os.getenv("GPT_API")

client = OpenAI(
    api_key=OpenAI_api_key
)




completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    messages=[{"role": "user", "content": "write a haiku about ai"}])

print(completion.choices[0].message.content)
