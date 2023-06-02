##
#
#
#
# Source origin
# - DeepLearning.ai / OpenAI course: https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers
##

import openai
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_completion(user_prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": user_prompt}]
    ai_response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return ai_response.choices[0].message["content"]

prompt = "Name five reasons to become a developer"
result = get_completion(prompt)

print(result)
