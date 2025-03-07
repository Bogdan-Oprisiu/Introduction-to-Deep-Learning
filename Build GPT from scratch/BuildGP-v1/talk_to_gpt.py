import torch
from train_gpt import encode, decode, device, GPTLanguageModel

# Load the trained model
model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load('gpt_language_model.pth', map_location=device))
model.eval()

# Function to generate a response from a prompt
def generate_response(model, prompt, max_new_tokens=100):
    prompt_tokens = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    response_tokens = model.generate(prompt_tokens, max_new_tokens=max_new_tokens)
    response = decode(response_tokens[0].tolist())
    return response

# Main loop to accept prompts from the CMD
if __name__ == "__main__":
    while True:
        prompt = input("Enter a prompt (or type 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        response = generate_response(model, prompt, max_new_tokens=200)
        print(f"Response: {response}")
