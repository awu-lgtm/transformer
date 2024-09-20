import math
import torch
from torch import nn
from tqdm import tqdm
from torch.optim import Adam, Optimizer
from data import TransformerDataset, SeqLenBatchSampler
from torch.utils.data import DataLoader
from torch.nn import Transformer
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from transformer import PositionalEncoder


class Transformer2(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size):
        super(Transformer2, self).__init__()
        self.transformer = Transformer(batch_first=True, norm_first=True)
        self.pe = PositionalEncoder(200, 512)
        self.emb1 = nn.Embedding(in_vocab_size, 512)
        self.emb2 = nn.Embedding(out_vocab_size, 512)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.linear = nn.Linear(512, out_vocab_size)

    def forward(self, x, y):
        src_mask, tgt_mask, la_mask = masks(x, y)
        x = self.drop1(self.pe(self.emb1(x)))
        y = self.drop2(self.pe(self.emb2(y)))
        a = self.transformer.forward(
            x,
            y,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask,
            tgt_mask=la_mask,
        )
        return self.linear(a)


class LRScheduler:
    def __init__(self, optimizer: Optimizer, d_model: int, warmup_steps: int):
        self.step_num = 0
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.d_model = d_model

    def step(self):
        self.step_num += 1
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


def masks(input: torch.Tensor, output: torch.Tensor):
    _, seq_len = output.shape
    in_mask = input == 1
    out_mask = output == 1
    la_mask = torch.ones([seq_len, seq_len]).triu(diagonal=1).bool().to(out_mask.device)
    # la_mask = out_mask | la_mask
    return in_mask, out_mask, la_mask


def train(model: Transformer2, ds: TransformerDataset, epochs: int, device: str, name):
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
    scheduler = LRScheduler(optimizer, 512, 4000)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id_output)
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

                        y_hat = model.forward(x, y_input)
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
