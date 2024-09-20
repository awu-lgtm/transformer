# from transformer import train, Transformer
from data import EN_DE_Dataset, en_tokenizer_path, de_tokenizer_path, get_ds, test_path
import torch
import transformer2
from torch.nn import Transformer
from transformer import inference, average_models, score
import os
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # ds = EN_DE_Dataset(None, None, None, None, 512, 256, 200)
    # ds.load_tokenizers()
    # ds.load_tokens()
    # max_seq_len_in, max_seq_len_out = ds.max_seq_lens()
    # pad_id_in, pad_id_out = ds.tokenizer_input.token_to_id("<pad>"), ds.tokenizer_output.token_to_id("<pad>")
    # model = Transformer(in_vocab_size=ds.vocab_size_input, out_vocab_size=ds.vocab_size_output,
    # max_seq_len=200, in_pad_id=pad_id_in, out_pad_id=pad_id_out)
    # train(model, ds, 32, device, "2")
    # print(ds.vocab_size_input, ds.vocab_size_output)
    model = transformer2.Transformer2(30257, 30257)
    # transformer2.train(model, ds, 32, "cuda", "control")
    model = average_models(
        model,
        batch_modify_paths(["10", "11", "12", "13", "14"], "models/control", "pt"),
    )
    inference(model, torch.tensor([[1, 2, 3]]), torch.tensor([[1, 2, 3]]))


def batch_modify_paths(paths, prefix, suffix):
    new_paths = []
    for path in paths:
        new_paths.append(os.path.join(prefix, f"{path}.{suffix}"))
    return new_paths


def test():
    df = pd.read_parquet(test_path)
    ds = EN_DE_Dataset(df["en"], df["de"], None, None, 512, 256, 200)
    ds.load_tokenizers()
    ds.tokenize(False, [])
    model = transformer2.Transformer2(30257, 30257)
    model = average_models(
        model,
        batch_modify_paths(["14"], "models/control", "pt"),
    )
    return score(model, ds.inputs, ds.outputs, ds.tokenizer_input, ds.tokenizer_output)


if __name__ == "__main__":
    test()
