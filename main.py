from transformer import train, Transformer
from data import EN_DE_Dataset, en_tokenizer_path, de_tokenizer_path, get_ds

def main():
    ds = EN_DE_Dataset(*get_ds("train"), None, None)
    ds.load_tokenizers()
    ds.tokenize()
    # print(ds.inputs)
    print(ds.max_seq_len())
    # model = Transformer(input_vocab_size=ds.vocab_size_input, output_vocab_size=ds.vocab_size_output, seq_len=)

if __name__ == "__main__":
    main()