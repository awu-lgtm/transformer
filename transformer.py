import torch
from torch import nn
from data import TransformerDataset, SeqLenBatchSampler
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.optim import Adam, Optimizer, AdamW
from tqdm import tqdm
import math
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import evaluate


class Attention(nn.Module):
    def __init__(self, d_k):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.d_k = d_k

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask):
        # dim Q = dim K = dim V = (batch_size x heads x seq len x d_k)
        # dim e = dim w = (batch_size x heads x Q seq len x K seq len)
        # dim a = (batch_size x heads x Q seq len x d_k)
        e = (Q @ K.transpose(2, 3)) / math.sqrt(self.d_k)
        e.masked_fill_(mask, float("-inf"))

        w = self.softmax(e)
        a = w @ V
        return a


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention(self.d_k)

    def linear_project(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        batch_size = Q.size(0)
        # (batch size x seq len x d_model) -> (batch size x seq len x heads x d_k) -> (batch size x heads x seq len x d_k)
        Q = Q.view([batch_size, Q.size(1), self.heads, self.d_k]).transpose(1, 2)
        K = K.view([batch_size, K.size(1), self.heads, self.d_k]).transpose(1, 2)
        V = V.view([batch_size, V.size(1), self.heads, self.d_k]).transpose(1, 2)

        return Q, K, V

    def concatenate_heads(self, heads: torch.Tensor):
        batch_size, _, seq_len, _ = heads.shape
        return (
            heads.transpose(1, 2).contiguous().view([batch_size, seq_len, self.d_model])
        )

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask):
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)
        Q, K, V = self.linear_project(Q, K, V)
        heads = self.attention(Q, K, V, mask)
        heads = self.concatenate_heads(heads)
        O = self.W_O(heads)
        return O


class PositionalEncoder(nn.Module):
    def __init__(self, seq_len, d_model):
        super(PositionalEncoder, self).__init__()
        # pe = torch.zeros(seq_len, d_model)
        # for pos in range(seq_len):
        #     for i in range(d_model):
        #         if i%2 == 0:
        #             pe[pos, i] = math.sin(pos/(10000**(2 * i/d_model)))
        #         else:
        #             pe[pos, i] = math.cos(pos/(10000**(2 * i/d_model)))

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.d_model = d_model
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        # (batch size x seq len x emb dim)
        seq_len = x.size(dim=1)
        x += self.pe[:seq_len, :]
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_dim, d_ff, out_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, d_ff), nn.Linear(d_ff, out_dim), nn.ReLU()
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, heads, d_model, d_ff, pre_ln):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(heads=heads, d_model=d_model)
        self.drop1 = nn.Dropout(p=0.1)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, d_model)
        self.drop2 = nn.Dropout(p=0.1)
        self.ln2 = nn.LayerNorm(d_model)
        self.pre_ln = pre_ln

    def forward(self, x: torch.Tensor, mask):
        res = x
        if self.pre_ln:
            x = self.ln1(x)
        x = self.attention(x, x, x, mask)
        x = self.drop1(x)
        x = x + res
        if not self.pre_ln:
            x = self.ln1(x)

        res = x
        if self.pre_ln:
            x = self.ln2(x)
        x = self.ffn(x)
        x = self.drop2(x)
        x = x + res
        if not self.pre_ln:
            x = self.ln2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, N, heads, d_model, d_ff, pre_ln):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList(
            [EncoderBlock(heads, d_model, d_ff, pre_ln) for _ in range(N)]
        )

    def forward(self, x, mask):
        for layer in self.encoder:
            x = layer(x, mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, heads, d_model, d_ff, pre_ln):
        super(DecoderBlock, self).__init__()
        self.masked_attention = MultiHeadAttention(heads=heads, d_model=d_model)
        self.drop1 = nn.Dropout(p=0.1)
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(heads=heads, d_model=d_model)
        self.drop2 = nn.Dropout(p=0.1)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff, d_model)
        self.drop3 = nn.Dropout(p=0.1)
        self.ln3 = nn.LayerNorm(d_model)
        self.pre_ln = pre_ln

    def forward(self, x: torch.Tensor, encoded: torch.Tensor, in_mask, la_mask):
        res = x
        if self.pre_ln:
            x = self.ln1(x)
        x = self.masked_attention(x, x, x, la_mask)
        x = self.drop1(x)
        x = x + res
        if not self.pre_ln:
            x = self.ln1(x)

        res = x
        if self.pre_ln:
            x = self.ln2(x)
        x = self.attention(x, encoded, encoded, in_mask)
        x = self.drop2(x)
        x = x + res
        if not self.pre_ln:
            x = self.ln2(x)

        res = x
        if self.pre_ln:
            x = self.ln3(x)
        x = self.ffn(x)
        x = self.drop3(x)
        x = x + res
        if not self.pre_ln:
            x = self.ln3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, N, heads, d_model, d_ff, pre_ln):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList(
            [DecoderBlock(heads, d_model, d_ff, pre_ln) for _ in range(N)]
        )

    def forward(self, x, encoded, in_mask, la_mask):
        for layer in self.decoder:
            x = layer(x, encoded, in_mask, la_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        in_vocab_size,
        out_vocab_size,
        max_seq_len,
        in_pad_id,
        out_pad_id,
        N=6,
        heads=8,
        d_model=512,
        d_ff=2048,
        pre_ln=False,
    ):
        super(Transformer, self).__init__()
        self.pe = PositionalEncoder(max_seq_len, d_model)
        self.enc_embedding = nn.Embedding(in_vocab_size, d_model)
        self.dec_embedding = nn.Embedding(out_vocab_size, d_model)
        self.enc_drop = nn.Dropout(p=0.1)
        self.dec_drop = nn.Dropout(p=0.1)
        self.d_model = d_model
        self.in_pad_id = in_pad_id
        self.out_pad_id = out_pad_id

        self.encoder = Encoder(N, heads, d_model, d_ff, pre_ln)
        self.decoder = Decoder(N, heads, d_model, d_ff, pre_ln)

        self.linear = nn.Linear(d_model, out_vocab_size)

    def forward(self, input, output):
        in_mask, la_mask = self.masks(input, output)

        input = self.enc_embedding(input)
        input = self.pe(input)
        input = self.enc_drop(input)

        output = self.dec_embedding(output)
        output = self.pe(output)
        output = self.dec_drop(output)

        input = self.encoder(input, in_mask)
        output = self.decoder(output, input, in_mask, la_mask)
        x = self.linear(output)
        return x

    def masks(self, input: torch.Tensor, output: torch.Tensor):
        _, seq_len = output.shape
        in_mask = (input == self.in_pad_id).unsqueeze(1).unsqueeze(2)
        out_mask = (output == self.out_pad_id).unsqueeze(1).unsqueeze(2)
        la_mask = (
            torch.ones([1, seq_len, seq_len])
            .triu(diagonal=1)
            .bool()
            .to(out_mask.device)
        )
        la_mask = out_mask | la_mask
        return in_mask, la_mask


class LRScheduler:
    def __init__(self, optimizer: Optimizer, d_model: int, warmup_steps: int, pre_ln):
        self.step_num = 0
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.d_model = d_model
        self.pre_ln = pre_ln

    def step(self):
        self.step_num += 1
        if self.pre_ln:
            lr = self.step_num**-0.5
        else:
            lr = (self.d_model**-0.5) * min(
                self.step_num**-0.5, self.step_num * (self.warmup_steps**-1.5)
            )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def collate_fn(batch, pad_id_input, pad_id_output):
    en_batch, de_batch = zip(*batch)
    en_batch = torch.nn.utils.rnn.pad_sequence(
        en_batch, padding_value=pad_id_input, batch_first=True
    )
    de_batch = torch.nn.utils.rnn.pad_sequence(
        de_batch, padding_value=pad_id_output, batch_first=True
    )
    return en_batch, de_batch


def train(model: Transformer, ds: TransformerDataset, epochs: int, device: str, name):
    model.to(device)
    model.train()
    accumulation_steps = 25
    pad_id_input, pad_id_output = ds.tokenizer_input.token_to_id(
        "<pad>"
    ), ds.tokenizer_output.token_to_id("<pad>")
    sampler = SeqLenBatchSampler(
        ds=ds, batch_size=25000, accumlation_steps=accumulation_steps
    )
    dl = DataLoader(
        ds,
        collate_fn=lambda batch: collate_fn(batch, pad_id_input, pad_id_output),
        batch_sampler=sampler,
    )
    optimizer = Adam(params=model.parameters(), lr=0, betas=[0.9, 0.98], eps=1e-9)
    scheduler = LRScheduler(optimizer, model.d_model, 4000, False)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id_output, label_smoothing=0.1)
    writer = SummaryWriter()
    Path(f"models/{name}").mkdir(parents=True, exist_ok=True)

    def train_loop(start_epoch, end_epoch, with_scheduler):
        for epoch in (bar := tqdm(range(start_epoch, end_epoch))):
            sampler.shuffle_batches()
            accumulated_loss = 0
            optimizer.zero_grad()
            with tqdm(total=len(dl) // accumulation_steps, leave=False) as epoch_bar:
                for i, (x, y) in enumerate(dl):
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        y_input = y[:, :-1]
                        y_next = y[:, 1:]

                        y_hat = model(x, y_input)
                        y_hat = y_hat.view(-1, y_hat.shape[-1])
                        y_next = y_next.contiguous().view(-1)
                        loss = loss_fn(y_hat, y_next)
                        loss.backward()
                        accumulated_loss += loss.item()

                        if (i + 1) % accumulation_steps == 0:
                            writer.add_scalar(
                                "Loss/train",
                                accumulated_loss / accumulation_steps,
                                ((i + 1) + epoch * len(dl)) // accumulation_steps,
                            )
                            optimizer.step()
                            accumulated_loss = 0
                            optimizer.zero_grad()
                            if with_scheduler:
                                scheduler.step()
                            epoch_bar.update()
                    if i % 100 == 0:
                        writer.flush()
                        bar.set_description(str(round(loss.item(), 3)))
            writer.flush()
            torch.save(model.state_dict(), f"models/{name}/{epoch}.pt")

    train_loop(0, epochs, True)
    writer.close()


def inference(model: Transformer, inputs: list, tokenizer: Tokenizer):
    model.eval()
    model.to("cuda")
    outputs = []
    for input in tqdm(inputs):
        with torch.no_grad():
            output, _ = beam_search(
                model,
                torch.tensor([input]).to("cuda"),
                tokenizer.token_to_id("<s>"),
                tokenizer.token_to_id("</s>"),
            )
        outputs.append(output)
    return outputs


def score(
    model, inputs, outputs, tokenizer_input: Tokenizer, tokenizer_output: Tokenizer
):
    bleu = evaluate.load("bleu")
    inferences = inference(model, inputs, tokenizer_output)
    translations = tokenizer_output.decode_batch([seq.tolist() for seq in inferences])
    score = bleu.compute(
        predictions=translations,
        references=tokenizer_output.decode_batch(outputs),
    )
    return score, translations


def beam_search(
    model: Transformer,
    input: torch.Tensor,
    start_token,
    end_token,
    beam_width=4,
    alpha=0.6,
    maximum_length=50,
):
    start = torch.tensor([[start_token]]).to("cuda")
    probs = F.softmax(model(input, start)[-1, -1], -1)
    sorted_probs, sorted_candidates = probs.sort(descending=True)
    scores = torch.log(sorted_probs[:beam_width])
    candidates = torch.concat(
        [
            start.repeat_interleave(beam_width, 0),
            sorted_candidates[:beam_width].unsqueeze(1),
        ],
        -1,
    )
    input = input.expand(beam_width, -1)
    final_candidates = []
    final_scores = []

    i = 0
    while i <= len(input) + maximum_length:
        probs = F.softmax(
            model(input, candidates)[:, -1], dim=-1
        )  # (beam_width  x vocab size)
        top_probs, top_next_tokens = probs.topk(beam_width, -1)
        scores = (
            (scores.unsqueeze(1).expand(beam_width, beam_width) + torch.log(top_probs))
            * 1
            / (5 + i + 1) ** alpha
        )  # (beam_width x beam_width)
        scores = scores.view(-1)
        top_scores, top_candidates = scores.sort(-1, descending=True)

        top_next_tokens = top_next_tokens.contiguous().view(-1, 1)
        candidates = torch.concat(
            [
                candidates.repeat_interleave(beam_width, 0),
                top_next_tokens.contiguous().view(-1, 1),
            ],
            -1,
        )

        indices = []
        for candidate in top_candidates:
            if len(indices) >= beam_width:
                break
            if top_next_tokens[candidate] == end_token:
                final_candidates.append(candidates[candidate])
                final_scores.append(scores[candidate])
            else:
                indices.append(candidate)
        candidates = candidates.index_select(0, torch.tensor(indices).to("cuda"))
        scores = top_scores.index_select(0, torch.tensor(indices).to("cuda"))

        i += 1
    if len(final_candidates) == 0:
        return candidates[0], scores[0]
    index = torch.argmax(torch.tensor(final_scores))
    return final_candidates[index], final_scores[index]


def average_models(model: Transformer, files: list[str]):
    n = len(files)
    avg = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    for file in files:
        for k, v in torch.load(file, map_location="cpu").items():
            avg[k] += v / n
    model.load_state_dict(avg)
    return model
