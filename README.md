
This repository contains a Jupyter Notebook demonstrating the use of LangChain, a framework for developing applications powered by language models. The notebook covers various aspects of LangChain, including installation, setup, and usage of different components like LLMs (Large Language Models), prompts, chains, and memory.

## Table of Contents
1. [Installation](#installation)
2. [Setup the Environment](#setup-the-environment)
3. [Large Language Models](#large-language-models)
4. [Prompt Templates](#prompt-templates)
5. [Chains](#chains)
6. [Memory](#memory)
7. [Usage Examples](#usage-examples)

## Installation

To install the required dependencies, run the following command:

```bash
!pip install langchain langchain_community
```

Additionally, you need to install the `huggingface_hub` package to interact with Hugging Face models:

```bash
!pip install huggingface_hub
```

## Setup the Environment

Set up the environment by configuring the Hugging Face API token:

```python
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_api_token_here"
```

## Large Language Models

The notebook demonstrates how to initialize and use a Large Language Model (LLM) from Hugging Face. The example uses the `google/flan-t5-large` model with specific parameters like `temperature` and `max_length`.

```python
from langchain import HuggingFaceHub

llm = HuggingFaceHub(repo_id='google/flan-t5-large', model_kwargs={"temperature":0, "max_length":64})
```

## Prompt Templates

Prompt templates are used to structure the input to the LLM. The notebook shows how to create and use a prompt template:

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is a good name for a company that makes {products}?")
```

## Chains

Chains allow you to combine multiple components, such as LLMs and prompts, to create more complex workflows. The notebook demonstrates the use of `LLMChain`, `SimpleSequentialChain`, and `SequentialChain`.

### LLMChain

```python
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run("football")
print(response)
```

### SimpleSequentialChain

```python
from langchain.chains import SimpleSequentialChain

chain = SimpleSequentialChain(chains=[name_chain, food_items_chain])
content = chain.run("indian")
print(content)
```

### SequentialChain

```python
from langchain.chains import SequentialChain

chain = SequentialChain(
    chains=[name_chain, food_items_chain],
    input_variables=['cuisine'],
    output_variables=['restaurant_name', 'menu_items']
)

print(chain({"cuisine": "indian"}))
```

## Memory

Memory allows the model to remember previous interactions. The notebook demonstrates the use of `ConversationBufferMemory` and `ConversationBufferWindowMemory`.

### ConversationBufferMemory

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

chain = LLMChain(llm=llm, prompt=prompt_template_name, memory=memory)
name = chain.run("Mexican")
print(name)
```

### ConversationBufferWindowMemory

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=3)

convo = ConversationChain(llm=llm, memory=memory)
convo.run("Who won the first cricket world cup?")
```

## Usage Examples

The notebook includes several examples of how to use LangChain components to generate text, create chains, and manage memory. Below are some of the key examples:

- **Generating a company name based on a product description.**
- **Creating a restaurant name and suggesting menu items based on cuisine.**
- **Using memory to remember previous interactions in a conversation.**

## Conclusion

This notebook provides a comprehensive introduction to LangChain, covering the basics of setting up and using LLMs, prompts, chains, and memory. It is a great starting point for anyone looking to build applications powered by language models.



---

Feel free to explore the notebook and experiment with the code. Happy coding! 🚀
