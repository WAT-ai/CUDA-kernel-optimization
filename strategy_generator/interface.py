import argparse
import openai
import os

# timing
from datetime import datetime

from dotenv import load_dotenv

MODEL_NAME = "gpt-4-turbo-preview"

# Load environment variables from .env file
load_dotenv(override=True)

print(f"OpenAI API Key: {os.getenv('OPENAI_API_KEY')}")

client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

"""
Task Decsription
User Question
Dynamic Context
Examples
"""
chat = [
    {
        "role": "system",
        "content": "Act as a kernel_tuner optimization strategy expert for enhancing parameter search efficiency. Given a CUDA kernel, write an algorithm in python using kernel tuner along with the exampels provided to perform an efficient parameter search over the searchable parameters specified. The algorithm should resemble the examples provided."
    }
]

# TODO: Implement RAG for context and example generation and ingestion (tokenization for quicker inference) maybe LangChain integration?
def inference(kernel_definition, context=None, examples=None):
    msg = {
        "role": "user",
        "content": f"Given the following CUDA kernel:\n{kernel_definition}\n*** If provided, use the performance metrics, and example optimization strategies below as a reference for creating a new novel optimization strategy. The function name for the strategy should be 'generated_optimization_strategy'. ***\n\nPerformance Context:\n{context}\n\nExamples:\n{examples}"
    }

    chat.append(msg)

    response = client.chat.completions.create(
        messages=chat,
        model=MODEL_NAME,
        presence_penalty=0,
        frequency_penalty=1.0,
        temperature=0.2,
        n=1,
    )

    """
    {
        role: "assistant",
        content: "<kernel optimization strategy>",
    }
    """
    reply = {
        "role": response.choices[0].message.role,
        "content": response.choices[0].message.content
    }

    # Append the response to the chat and print it
    chat.append(reply)
    print(f"Response: {reply['content']}")

    return reply["content"]

def extract_python_code(response_content):
    code_start = response_content.find("```python")
    code_end = response_content.find("```", code_start + 1)

    if code_start != -1 and code_end != -1:
        return response_content[code_start + len("```python"):code_end].strip()
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description='CLI for OpenAI GPT-4 Turbo')
    parser.add_argument('--cuda-file', type=str, help='Path to the CUDA file')
    # Add more arguments as needed

    args = parser.parse_args()

    # Access the CUDA file path using args.cuda_file
    cuda_file_path = args.cuda_file

    # Read the CUDA file
    with open(cuda_file_path, 'r') as file:
        cuda_code = file.read()

    # Extract the CUDA kernel starting from and including __global__
    kernel_start = cuda_code.find('__global__')
    if kernel_start != -1:
        cuda_kernel = cuda_code[kernel_start:]
    else:
        raise Exception("Kernel not found in CUDA code")

    # Print the CUDA file path and the CUDA kernel
    print(f"CUDA file path: {cuda_file_path}\n\n")
    print(f"CUDA kernel:\n{cuda_kernel}\n\n")

    # Load examples from examples directory and append into a string
    examples = ""
    examples_dir = "optimization_strategy_examples/"
    for filename in os.listdir(examples_dir):
        with open(os.path.join(examples_dir, filename), 'r') as file:
            examples += f"\n******** EXAMPLE {filename} ********\n\n" + file.read() + "\n"

    # build optimization strategy
    optim_strat = inference(cuda_kernel, examples=examples)
    optim_strat_python_code = extract_python_code(optim_strat)

    if optim_strat_python_code:
        output_dir = "strategy_generator/strategies/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_name = "generated_optimization_strategy.py"

        with open(os.path.join(output_dir, file_name), 'w') as file:
            file.write(optim_strat_python_code)
        
        print(f"saved at: {os.path.join(output_dir, file_name)}")
    else:
        print("No Python code block was found in the response.")

if __name__ == '__main__':
    main()
