import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from conllu import parse
import json

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim_char, hidden_dim_char, embedding_dim_word, hidden_dim_word, vocab_size, tagset_size, character_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim_word = hidden_dim_word
        self.hidden_dim_char = hidden_dim_char

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim_word)

        self.char_embeddings = nn.Embedding(character_size, embedding_dim_char)
        self.lstm_chars = nn.LSTM(embedding_dim_char, hidden_dim_char)

        self.lstm = nn.LSTM(hidden_dim_char + embedding_dim_word, hidden_dim_word)

        self.hidden2tag = nn.Linear(hidden_dim_word, tagset_size)

    def forward(self, sentence):
        embeds_word = self.word_embeddings(sentence)

        embeds_char = torch.zeros([len(sentence), 1, HIDDEN_DIM_CHAR])

        count = 0
        for word_idx in sentence:
            word = idx_to_word[word_idx.item()]
            ascii_char_values = [ord(c) for c in word]
            ascii_char_values_tensor = torch.tensor(ascii_char_values, dtype=torch.long)
            embeds_char_curr = self.char_embeddings(ascii_char_values_tensor)
            _, lstm_char_final = self.lstm_chars(embeds_char_curr.view(len(ascii_char_values), 1, -1))
            embeds_char[count] = lstm_char_final[0].view(1, 1, -1)
            count += 1
            
        embeds = torch.cat((embeds_word.view(len(sentence), 1, -1), embeds_char), 2)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def create_word_to_idx(sentences):
    word_to_idx = {}
    for sentence, _ in sentences:
        for word in sentence:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
    word_to_idx["__UNKNOWN_WORD__"] = len(word_to_idx)
    return word_to_idx, {v: k for (k, v) in word_to_idx.items()}

def create_input(seq, idx_to):
    idx = []
    for i, word in enumerate(seq):
        try:
            idx.append(idx_to[word])
        except:
            idx.append(idx_to["__UNKNOWN_WORD__"])
    return torch.tensor(idx, dtype=torch.long)

EMBEDDING_DIM_CHAR = 5
HIDDEN_DIM_CHAR = 3

EMBEDDING_DIM_WORD = 10
HIDDEN_DIM_WORD = 5

NUM_EPOCHS = 12

ASCII_CHARACTERS = 2**16

with open ("/home/nvinden/DLProjects/SENTANCE_GRAMMAR_STRUCT_LSTM/data.json", "r") as myfile:
    data=myfile.read()

train_data_json = json.loads(data)
train_data = []

for entry in train_data_json:
    train_data.append((entry["sentence"].split(), entry["tags"]))

tag_to_idx = {"WH": 0, "ADV": 1, "MOD": 2, "PRON": 3, "VERB": 4, "TO": 5, "DT": 6, "ADJ": 7, "NOUN": 8, "PREP": 9, "CONJ": 10, "NUMB": 11, "PART": 12, "AUX": 13}

word_to_idx, idx_to_word = create_word_to_idx(train_data)


model = LSTMTagger(EMBEDDING_DIM_CHAR, HIDDEN_DIM_CHAR, EMBEDDING_DIM_WORD, HIDDEN_DIM_WORD, len(word_to_idx), len(tag_to_idx), ASCII_CHARACTERS)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(NUM_EPOCHS):
    for count, (sentence, tags) in enumerate(train_data):
        model.zero_grad()
        sentence_in = create_input(sentence, word_to_idx)
        targets = create_input(tags, tag_to_idx)
        out = model(sentence_in)
        loss = loss_function(out, targets)
        loss.backward()
        optimizer.step()
        if count % 100 == 0 and count != 0:
            print("Epoch: {} count: {} Loss = {}".format(epoch, count, loss))
    torch.save(model.state_dict(), "./model")
