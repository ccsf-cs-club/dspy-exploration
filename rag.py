# DSPy is a prompt engineering framework
import dspy
# Load API keys via .env file
import os
from dotenv import load_dotenv
load_dotenv()

# Configure DSPy to use OpenAI
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.environ.get("OPENAI_API_KEY"))
dspy.configure(lm=lm)

def search_wikipedia(query: str) -> list[str]:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

rag = dspy.ChainOfThought('context, question -> response')

question = "What stands out about City College of San Francisco?"
ans = rag(context=search_wikipedia(question), question=question)
print(ans)

# Prediction(
#    reasoning='City College of San Francisco (CCSF) stands out for its significant role
#       in the local community, as it enrolls a large portion of San Francisco residents—up to
#       one in nine. This highlights its importance as an accessible educational institution 
#       in the area. Additionally, being a public two-year community college, it provides opport
#       unities for higher education to a diverse population, which is a key aspect of its mission.',
#   response='City College of San Francisco is notable for its vital role in the local
#     community, enrolling as many as one in nine San Francisco residents, which underscores
#     its accessibility and importance as a public two-year community college.'
# )
