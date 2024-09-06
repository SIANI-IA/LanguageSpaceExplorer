from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import openai
import json
import tqdm
import os
from utils_key import API_KEY
#delete warnings
import warnings
warnings.filterwarnings("ignore")

# Constants
FOLDER = 'data'
MAX_ITERATIONS = 100  # Maximum number of iterations
TEMPERATURE = 0.7  # Temperature for the OpenAI model
DOMAIN = 'technology'  # Domain for the OpenAI model sports and technology
PROMPT_TEMPLATE = """
Generate {len} phrases about {domain}, each with no more than 50 words.
"""  # Fixed prompt template
OUTPUT_NAME = f'{"_".join(DOMAIN.split())}_phrases.json'  # Output file name
OUTPUT_FILE = os.path.join(FOLDER, OUTPUT_NAME)

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = API_KEY

# Define the OpenAI model
llm = OpenAI(temperature=TEMPERATURE)

# Create the prompt
prompt = PromptTemplate(
    input_variables=["len", "domain"],
    template=PROMPT_TEMPLATE,
)

data = []
# Create the execution chain

for _ in tqdm.tqdm(range(MAX_ITERATIONS), desc="Generating phrases"):
    chain = LLMChain(llm=llm, prompt=prompt)

    # Execute the chain to get the phrases
    output = chain.run(len=5, domain=DOMAIN)

    # Separate the phrases by line if they are not already separated (optional, depending on the original output)
    phrases = output.split("\n")

    # Create a list with the structure {"text": ---}
    data += [{"text": phrase.strip()} for phrase in phrases if phrase.strip()]

# Save the result to a JSON file
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"JSON file saved successfully. Total phrases: {len(data)}")
