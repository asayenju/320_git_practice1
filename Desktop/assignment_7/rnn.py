import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import re

# ===================== Dataset =====================
class CharDataset(Dataset):
    def __init__(self, data, sequence_length, stride, vocab_size):
        self.data = data
        self.sequence_length = sequence_length
        self.stride = stride
        self.vocab_size = vocab_size
        self.sequences = []
        self.targets = []

        for i in range(0, len(data) - sequence_length, stride):
            self.sequences.append(data[i:i + sequence_length])
            self.targets.append(data[i + 1:i + sequence_length + 1])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return sequence, target

# ===================== Model =====================
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim=30):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_dim)

        self.W_xh = nn.Parameter(torch.randn(embedding_dim, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        self.W_hy = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x_embed = self.embedding(x)  # [b, l, e]
        b, l, _ = x_embed.size()
        x_embed = x_embed.transpose(0, 1)  # [l, b, e]

        h_t = self.init_hidden(b) if hidden is None else hidden
        output = []

        for t in range(l):
            x_t = x_embed[t]  # [b, e]
            h_t = torch.tanh(x_t @ self.W_xh + h_t @ self.W_hh + self.b_h)
            output.append(h_t)

        output = torch.stack(output)  # [l, b, h]
        output = output.transpose(0, 1)  # [b, l, h]
        logits = self.W_hy(output)  # [b, l, vocab_size]

        final_hidden = h_t.clone()
        return logits, final_hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).to(device)

# ===================== Training =====================
device = 'cpu'
print(f"Using device: {device}")

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read().lower()
        text = re.sub(r'[^a-z.,!?;:()\[\] ]+', '', text)
    return text

# Use alphabet sequence for debugging
sequence = "abcdefghijklmnopqrstuvwxyz" * 100
# sequence = read_file("warandpeace.txt")
vocab = sorted(set(sequence))
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}
data = [char_to_idx[ch] for ch in sequence]

# Hyperparameters
sequence_length = 50
stride = 1
embedding_dim = 32
hidden_size = 128
learning_rate = 1e-3
num_epochs = 5
batch_size = 64
vocab_size = len(vocab)
input_size = len(vocab)
output_size = len(vocab)

model = CharRNN(input_size, hidden_size, output_size, embedding_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_tensor = torch.tensor(data, dtype=torch.long)
train_size = int(len(data_tensor) * 0.9)
train_data = data_tensor[:train_size]
test_data = data_tensor[train_size:]

train_dataset = CharDataset(train_data, sequence_length, stride, vocab_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# ===================== Training Loop =====================
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    hidden = None
    for batch_inputs, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        output, hidden = model(batch_inputs, hidden)
        hidden = hidden.detach()

        loss = criterion(output.view(-1, output_size), batch_targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# ===================== Test Loop =====================
model.eval()
test_dataset = CharDataset(test_data, sequence_length, stride, vocab_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

test_loss = 0
hidden = None

with torch.no_grad():
    for batch_inputs, batch_targets in tqdm(test_loader, desc="Testing"):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        output, hidden = model(batch_inputs, hidden)
        hidden = hidden.detach()
        loss = criterion(output.view(-1, output_size), batch_targets.view(-1))
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")

# ===================== Text Generation =====================
def sample_from_output(logits, temperature=1.0):
    if temperature <= 0:
        temperature = 0.00001
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=1)
    sampled_idx = torch.multinomial(probabilities, 1)
    return sampled_idx

def generate_text(model, start_text, n, k, temperature=1.0):
    model.eval()
    generated_text = start_text.lower()

    input_seq = [char_to_idx.get(c, 0) for c in start_text[-n:]]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
    hidden = None

    for _ in range(k):
        logits, hidden = model(input_tensor, hidden)
        hidden = hidden.detach()

        last_logits = logits[:, -1, :]
        next_idx = sample_from_output(last_logits, temperature).item()
        next_char = idx_to_char[next_idx]
        generated_text += next_char

        input_tensor = torch.cat([input_tensor[:, 1:], torch.tensor([[next_idx]], device=device)], dim=1)

    return generated_text

# ===================== Interactive Loop =====================
print("Training complete. Now you can generate text.")
while True:
    start_text = input("Enter the initial text (n characters, or 'exit' to quit): ")
    if start_text.lower() == 'exit':
        print("Exiting...")
        break

    n = len(start_text)
    k = int(input("Enter the number of characters to generate: "))
    temperature_input = input("Enter the temperature value (1.0 is default, >1 is more random): ")
    temperature = float(temperature_input) if temperature_input else 1.0

    completed_text = generate_text(model, start_text, n, k, temperature)
    print(f"Generated text: {completed_text}")
