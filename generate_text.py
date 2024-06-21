import torch
from tokenizers import Tokenizer
from model_training import TransformerModel

def generate_text(model, tokenizer, prompt, max_length=50):
    model.eval()
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(tokens).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_ids)
            next_token = output.argmax(-1)[:, -1].unsqueeze(0)
            input_ids = torch.cat((input_ids, next_token), dim=1)
    return tokenizer.decode(input_ids[0].tolist())

# Load the model
model = TransformerModel(30000, 512, 8, 6, 2048, 512)
model.load_state_dict(torch.load('math_transformer_model.pth'))
model.eval()

# Load the tokenizer
tokenizer = Tokenizer.from_file('math_tokenizer.json')

# Generate text
prompt = "The fundamental theorem of calculus states"
generated_text = generate_text(model, tokenizer, prompt, max_length=100)
print(generated_text)
