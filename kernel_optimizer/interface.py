import argparse
import openai
import os

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
        "content": "Act as a CUDA kernel optimizer. Given a CUDA kernel, provide the optimization strategy and parameters to improve its performance.\n\nThe CUDA kernel will be formatted as follows:\n__global__ void kernel_name(inputs) {\n    // CUDA kernel code\n}"
    }
]

# TODO: Implement RAG for context and example generation and ingestion (tokenization for quicker inference) maybe LangChain integratio?
def inference(query, context=None, examples=None):
    msg = {
        "role": "user",
        "content": f"Below is the CUDA kernel that needs optimization:\n{query}\n*** If available, use the provided performance metrics, example optimization strategies, and parameters below as a reference for creating the optimization strategy. ***\n\nPerformance Context:\n{context}\n\nExamples:\n{examples}"
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
    print(f"CUDA file path: {cuda_file_path}")
    print(f"CUDA kernel:\n{cuda_kernel}")

    # Run the inference
    inference(cuda_kernel)

if __name__ == '__main__':
    main()
