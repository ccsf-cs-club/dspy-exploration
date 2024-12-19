# DSPy is a prompt engineering framework
import dspy
# Load API keys via .env file
import os
from dotenv import load_dotenv
load_dotenv()

# Configure DSPy to use OpenAI
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.environ.get("OPENAI_API_KEY"))
dspy.configure(lm=lm)

print(lm("Say this is a test!", temperature=0.7))  # => ['This is a test! How can I assist you further?']
