
import os

import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro")
response = model.generate_content("what is the tallest building in india?")
summary = response.text
print(summary)