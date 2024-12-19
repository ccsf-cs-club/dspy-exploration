# DSPy Tutorial

## Background
> DSPy is a fantastic framework for LLMs that introduces an automatic compiler that teaches LMs how to conduct the declarative steps in your program. Specifically, the DSPy compiler will internally trace your program and then craft high-quality prompts for large LMs (or train automatic finetunes for small LMs) to teach them the steps of your task.

## Requirements
Python 3.12+ Recommended
## Setup
### Virtual Env
```bash
python -m venv venv
source venv/bin/activate
```
### Installation
```bash
pip install python-dotenv
pip install -U dspy
```
### Configuration

We'll use a .env file for this process, starting with creating a .env.sample

```bash
touch .env.sample
touch main.py
```

Set up your API keys as needed

```md
# .env
OPENAI_API_KEY=...
```

### Writing the dspy hello world
```python
  # main.py
  import os
  import dspy
  from dotenv import load_dotenv
  load_dotenv()

  # Accessing an environment variable
  lm = dspy.LM('openai/gpt-4o-mini', api_key=os.environ.get("OPENAI_API_KEY"))
  dspy.configure(lm=lm)

  print(lm("Say this is a test!", temperature=0.7))  # => ['This is a test! How can I assist you further?']
```

### Other Examples
#### Chain of Thought + Math
```python
math = dspy.ChainOfThought("question -> answer: float")
math(question="Two dice are tossed. What is the probability that the sum equals two?")
```
#### Retrieval Augmented Generation
```python
def search_wikipedia(query: str) -> list[str]:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

rag = dspy.ChainOfThought('context, question -> response')

question = "What stands out about City College of San Francisco?"
rag(context=search_wikipedia(question), question=question)
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
```

