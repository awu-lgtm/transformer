from torch.utils.data import Dataset, Sampler
import pandas as pd
from tokenizers import Tokenizer
import torch
import pickle
import numpy as np
import random

train_path = "wmt14_translate_de-en_train.parquet"
test_path = "wmt14_translate_de-en_test.parquet"
train_path_csv = "wmt14_translate_de-en_train.csv"
test_path_csv = "wmt14_translate_de-en_test.csv"
de_txt_path = "wmt14_de_train.txt"
en_txt_path = "wmt14_en_train.txt"
en_tokenizer_path = "tokenizers/wmt14_en.json"
de_tokenizer_path = "tokenizers/wmt14_de.json"
cache_train_in = "tokens_input.pkl"
cache_train_out = "tokens_output.pkl"

class TransformerDataset(Dataset):
    def __init__(self, inputs, outputs, tokenizer_input: Tokenizer, tokenizer_output: Tokenizer, chunk_len, step_size, max_len):
        super(TransformerDataset, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.tokenizer_input = tokenizer_input
        self.tokenizer_output = tokenizer_output
        self.max_len = max_len
        self.chunk_len = chunk_len
        self.step_size = step_size

    def tokenize(self, cache, paths):
        if self.tokenizer_input and self.tokenizer_output:
            self.inputs = [tokens.ids for tokens in self.tokenizer_input.encode_batch(self.inputs)]
            self.outputs = [tokens.ids for tokens in self.tokenizer_output.encode_batch(self.outputs)]

            if cache:
                with open(paths[0], "wb") as file:
                    pickle.dump(self.inputs, file)
                with open(paths[1], "wb") as file:
                    pickle.dump(self.outputs, file)

    def load_tokens(self, paths):
        with open(paths[0],'rb') as file:
            self.inputs = pickle.load(file)
            self.inputs = [self.truncate_sample(input, self.max_len) for input in self.inputs]
        with open(paths[1],'rb') as file:
            self.outputs = pickle.load(file)
            self.outputs = [self.truncate_sample(output, self.max_len) for output in self.outputs]

    def truncate_sample(self, x, length):
        return x[:length]
    
    def sliding_window(self, x, padding):
        if len(x) <= self.chunk_len:
            return x
        if len(x) > self.max_len:
            x = self.truncate_sample(x, self.max_len)
        chunks = [torch.tensor(x[i:i+self.chunk_len]) for i in range(0, len(x), self.step_size)]
        chunks = torch.nested.nested_tensor(chunks).to_padded_tensor(padding)
        return chunks
            
    def max_seq_lens(self):
        return len(max(self.inputs, key=len)), len(max(self.outputs, key=len))

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return torch.tensor(self.inputs[index]), torch.tensor(self.outputs[index])
        # ds.tokenizer_input.token_to_id("<pad>"), self.tokenizer_output.token_to_id("<pad>")
        # return self.sliding_window(self.inputs[index], self.tokenizer_input.token_to_id("<pad>")), self.sliding_window(self.outputs[index], self.tokenizer_output.token_to_id("<pad>"))
    
class EN_DE_Dataset(TransformerDataset):
    def load_tokenizers(self):
        self.tokenizer_input = Tokenizer.from_file(en_tokenizer_path)
        self.tokenizer_output = Tokenizer.from_file(de_tokenizer_path)
        self.vocab_size_input = self.tokenizer_input.get_vocab_size()
        self.vocab_size_output = self.tokenizer_output.get_vocab_size()

class SeqLenBatchSampler(Sampler):
    def __init__(self, ds: TransformerDataset, batch_size, accumlation_steps):
        self.ds = ds
        self.batch_size = batch_size
        self.accumlation_steps = accumlation_steps

        batches = []
        sorted_lengths = np.argsort([len(input) for input in self.ds.inputs])
        i = 0
        sum_length = 0
        for j, index in enumerate(sorted_lengths):
            sum_length += len(self.ds.inputs[index])
            if sum_length > batch_size:
                batches.append(self.batch_to_steps(i, j, sorted_lengths))
                i = j
                sum_length = 0
        if i != len(ds) - 1:
            batches.append(self.batch_to_steps(i, len(ds), sorted_lengths))
        self.batches = batches

    def batch_to_steps(self, i, j, sorted_lengths):
        steps = []
        step_size = (j - i)//self.accumlation_steps
        for step in range(self.accumlation_steps - 1):
            steps.append(sorted_lengths[i+step_size*step:i+step_size*(step+1)])
        steps.append(sorted_lengths[i+step_size*(self.accumlation_steps-1):j])
        return steps
        
    def shuffle_batches(self):
        random.shuffle(self.batches)
        for batch in self.batches:
            for step in batch:
                random.shuffle(step)
            random.shuffle(batch)
        
    def __iter__(self):
        for batch in self.batches:
            for step in batch:
                yield step

    def __len__(self):
        return len(self.batches) * self.accumlation_steps

def get_ds(ds: str):
    path = train_path if ds == "train" else test_path
    df = pd.read_parquet(path)
    return df["en"].to_list(), df["de"].to_list()

if __name__ == "__main__":
    for csv, parquet in [(train_path_csv, train_path), (test_path_csv, test_path)]:
        pd.read_csv(csv, lineterminator="\n").to_parquet(parquet)