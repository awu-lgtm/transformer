import torch
from torch import nn
from data import TransformerDataset
from torch.utils.data import DataLoader
from torch.optim import Adam, Optimizer
from tqdm import tqdm

class Attention(nn.Module):
    def __init__(self, d_k, mask):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.d_k = d_k
        self.mask = mask
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        # dim Q = dim K = dim V = (batch_size x heads x seq len x d_k)
        # dim w = (batch_size x heads x seq len x seq len)
        # dim a = (batch_size x heads x seq len x d_k)
        e = (Q @ K.transpose(2, 3))/torch.sqrt(self.d_k)
        if self.mask:
            neg_inf = -1e10
            mask = e.triu(diagonal=1).to(torch.bool)
            e.masked_fill(mask, neg_inf)

        w = self.softmax(e)
        a = w @ V
        return a
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, mask):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model//heads
        self.heads = heads
        self.d_model = d_model
        self.mask = mask

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention(self.d_k, mask)

    def linear_project(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        batch_size, seq_len, _ = Q.shape
        # (batch size x seq len x emb dim) -> (batch size x seq len x heads x d_k) -> (batch size x heads x seq len x d_k)
        Q = Q.view([batch_size, seq_len, self.heads, self.d_k]).transpose(1,2)
        K = K.view([batch_size, seq_len, self.heads, self.d_k]).transpose(1,2)
        V = V.view([batch_size, seq_len, self.heads, self.d_k]).transpose(1,2)

        return Q, K, V

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)
        Q, K, V = self.linear_project(Q, K, V)
        heads: torch.Tensor = self.attention(Q, K, V)

        batch_size, seq_len, _ = Q.shape
        heads = heads.transpose(1, 2).contiguous().view([batch_size, seq_len, self.d_model])
        O = self.W_O(heads)
        return O

class PositionalEncoder(nn.Module):
    def __init__(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        for pos in range(seq_len):
            for i in range(d_model):
                if i%2 == 0:
                    pe[pos, i] = torch.sin(pos/10000**(2 * i/d_model))
                else:
                    pe[pos, i] = torch.cos(pos/10000**(2 * i/d_model))
        self.pe = pe
        self.register_buffer("positional encoding", pe)
    
    def forward(self, x: torch.Tensor):
        # (batch size x seq len x emb dim)
        seq_len = x.size(dim=1)
        x += self.pe[:seq_len, :].detach()
        return x
    
class ResidualLayer(nn.Module):
    def __init__(self, sublayer):
        super(ResidualLayer, self).__init__()
        self.sublayer = sublayer

    def forward(self, x):
        x += self.sublayer(x)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForward, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation_fn = nn.ReLU()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation_fn(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, heads, d_model):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(heads=heads, d_model=d_model, mask=False)
        self.dropout1 = nn.Dropout(p=0.1)
        self.ln1 = nn.LayerNorm(),
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        self.dropout2 = nn.Dropout(p=0.1)
        self.ln2 = nn.LayerNorm()

    def forward(self, x: torch.Tensor):
        res = x
        x = self.attention(x, x, x)
        x = self.dropout1(x)
        with torch.no_grad():
            x = x + res
        x = self.ln1(x)
        
        res = x
        x = self.ffn(x)
        x = self.dropout2(x)
        with torch.no_grad():
            x = x + res
        x = self.ln2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, N, heads, d_model):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(*[EncoderBlock(heads, d_model) for _ in range(N)])
    
    def forward(self, x):
        x = self.encoder(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, heads, d_model):
        super(EncoderBlock, self).__init__()
        self.masked_attention = MultiHeadAttention(heads=heads, d_model=d_model, mask=True)
        self.dropout1 = nn.Dropout(p=0.1)
        self.ln1 = nn.LayerNorm(),
        self.attention = MultiHeadAttention(heads=heads, d_model=d_model, mask=True)
        self.dropout2 = nn.Dropout(p=0.1)
        self.ln2 = nn.LayerNorm(),
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        self.dropout3 = nn.Dropout(p=0.1)
        self.ln3 = nn.LayerNorm()

    def forward(self, x: torch.Tensor, encoded: torch.Tensor):
        res = x
        x = self.masked_attention(x, x, x)
        x = self.dropout1(x)
        with torch.no_grad():
            x = x + res
        x = self.ln1(x)
        
        res = x
        x = self.attention(encoded, encoded, x)
        x = self.dropout2(x)
        with torch.no_grad():
            x = x + res
        x = self.ln2(x)

        res = x
        x = self.ffn(x)
        x = self.dropout3(x)
        with torch.no_grad():
            x = x + res
        x = self.ln3(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, N, heads, d_model):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList([DecoderBlock(heads, d_model) for _ in range(N)])

    def forward(self, x, encoded):
        for layer in self.decoder:
            x = layer(x, encoded)
        return x

class Transformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, seq_len, N=6, heads=8, d_model=512):
        super(Transformer, self).__init__()
        self.positional_encoder = PositionalEncoder(seq_len, d_model)
        self.input_embedding = nn.Embedding(input_vocab_size, d_model)
        self.output_embedding = nn.Embedding(output_vocab_size, d_model)
        self.input_dropout = nn.Dropout(p=0.1)
        self.output_dropout = nn.Dropout(p=0.1)
        self.d_model = d_model

        self.encoder = Encoder(N, heads, d_model)
        self.decoder = Decoder(N, heads, d_model)

        self.linear = nn.Linear(d_model, output_vocab_size)

    def forward(self, input, output):
        input = self.input_embedding(input)
        input = self.positional_encoder(input)
        input = self.input_dropout(input)
        output = self.output_embedding(output)
        output = self.positional_encoder(output)
        output = self.output_dropout(output)

        input = self.encoder(input)
        output = self.decoder(output, input)
        x = self.linear(output)
        return x
    
class LRScheduler():
    def __init__(self, optimizer: Optimizer, d_model: int, warmup_steps: int):
        self.step_num = 0
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.d_model = d_model

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * torch.min(self.step_num, self.step_num * self.warmup_steps ** -1.5)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

def collate_fn(batch, pad_id_input, pad_id_output):
    en_batch, de_batch = zip(*batch)
    en_batch = torch.nn.utils.rnn.pad_sequence(en_batch, padding_value=pad_id_input, batch_first=True)
    de_batch = torch.nn.utils.rnn.pad_sequence(de_batch, padding_value=pad_id_output, batch_first=True)
    return en_batch, de_batch


def train(model: Transformer, ds: TransformerDataset, epochs: int, device):
    model.train()
    pad_id_input, pad_id_output = ds.tokenizer_input.token_to_id("<pad>"), ds.tokenizer_output.token_to_id("<pad>")
    dl = DataLoader(ds, batch_size=1024, shuffle=True, collate_fn=lambda batch: collate_fn(batch, pad_id_input, pad_id_output), pin_memory=True, pin_memory_device=device)
    optimizer = Adam(lr=0, betas=[0.9, 0.98], eps=1e-9)
    scheduler = LRScheduler(optimizer, model.d_model, 4000)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id_output)
    
    for _ in (bar := tqdm(range(epochs))):
        for x, y in dl:
            y_input = y[:, :-1]
            y_next = y[:, 1:]

            y_hat = model(x, y_input)
            y_hat = y_hat.view(-1, y_hat.shape[-1])
            y_output = y_output.contiguous().view(-1)
            loss = loss_fn(y_hat, y_next)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        bar.set_description(str(loss.item()))