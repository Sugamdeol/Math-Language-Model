import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

class MathDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        self.examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(tokenizer.encode(line.strip()).ids)
       
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx])

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        output = self.transformer_encoder(src)
        output = self.fc(output)
        return output

# Load dataset
tokenizer = Tokenizer.from_file('math_tokenizer.json')
dataset = MathDataset(tokenizer, 'math_articles.txt')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True))

# Hyperparameters
vocab_size = 30000
d_model = 512
nhead = 8
num_encoder_layers = 6
dim_feedforward = 2048
max_seq_length = 512
num_epochs = 10

model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output.view(-1, vocab_size), batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Save the model
torch.save(model.state_dict(), 'math_transformer_model.pth')
