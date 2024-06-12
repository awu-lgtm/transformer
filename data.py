from torch.utils.data import Dataset
import pandas as pd
from tokenizers import Tokenizer
import torch

train_path = "wmt14_translate_de-en_train.parquet"
test_path = "wmt14_translate_de-en_test.parquet"
train_path_csv = "wmt14_translate_de-en_train.csv"
test_path_csv = "wmt14_translate_de-en_test.csv"
de_txt_path = "wmt14_de_train.txt"
en_txt_path = "wmt14_en_train.txt"
en_tokenizer_path = "tokenizers/wmt14_en.json"
de_tokenizer_path = "tokenizers/wmt14_de.json"

class TransformerDataset(Dataset):
    def __init__(self, inputs, outputs, tokenizer_input: Tokenizer, tokenizer_output: Tokenizer):
        super(TransformerDataset, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer_input = tokenizer_input
        self.tokenizer_output = tokenizer_output

    def tokenize(self):
        self.inputs = [tokens.ids for tokens in self.tokenizer_input.encode_batch(self.inputs)]
        print("hello")
        self.outputs = [tokens.ids for tokens in self.tokenizer_output.encode_batch(self.outputs)]

    def max_seq_len(self):
        return max(max(self.inputs, key=len), max(self.outputs, key=len))

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return torch.Tensor(self.inputs[index]), torch.Tensor(self.outputs[index])
    
class EN_DE_Dataset(TransformerDataset):
    def load_tokenizers(self):
        self.tokenizer_input = Tokenizer.from_file(en_tokenizer_path)
        self.tokenizer_output = Tokenizer.from_file(de_tokenizer_path)
        self.vocab_size_input = self.tokenizer_input.get_vocab_size()
        self.vocab_size_output = self.tokenizer_output.get_vocab_size()

def get_ds(ds: str):
    path = train_path if ds == "train" else test_path
    df = pd.read_parquet(path)
    return df["en"].to_list(), df["de"].to_list()

if __name__ == "__main__":
    b = pd.read_csv(test_path_csv, lineterminator="\n")
    b.to_parquet(test_path)
    b.to_csv("sasdf.csv")