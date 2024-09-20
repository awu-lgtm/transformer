from tokenizers import normalizers, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers.trainers import BpeTrainer
from data import (
    train_path,
    de_tokenizer_path,
    en_tokenizer_path,
    de_txt_path,
    en_txt_path,
)
import pandas as pd


def csv_to_txts():
    df = pd.read_parquet(train_path)

    for l in ["de", "en"]:
        f = open(globals()[f"{l}_txt_path"], "w")
        f.writelines([line + "\n" for line in df[l].to_list()])
        f.close()


def en_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def de_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def make_tokenizers():
    for l in ["de", "en"]:
        tokenizer = globals()[f"{l}_tokenizer"]()
        trainer = BpeTrainer(
            vocab_size=30257,
            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train([globals()[f"{l}_txt_path"]], trainer)
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B:1 </s>:1",
            special_tokens=[
                ("<s>", tokenizer.token_to_id("<s>")),
                ("</s>", tokenizer.token_to_id("</s>")),
            ],
        )
        tokenizer.save(globals()[f"{l}_tokenizer_path"])


def make_tokenizer():
    tokenizer = en_tokenizer()
    trainer = BpeTrainer(
        vocab_size=30257,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train([en_txt_path, de_txt_path], trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )
    tokenizer.save("tokenizers/wmt14_en_de.json")


# make_tokenizers()
# tokenizer = Tokenizer.from_file(en_tokenizer_path)
# a = tokenizer.encode("Hello, y'all! How are you 😁 ?")
# b = tokenizer.decode(a.ids)
# print(b)

if __name__ == "__main__":
    # tokenizer = Tokenizer.from_file(de_tokenizer_path)
    # a = tokenizer.encode("Rammstein ist eine bekannte deutsche Band. Die Gruppe hat fünf Mitglieder und Till Lindemann ist der Sänger der Band. Rammstein spielt hauptsächlich Metal- und Industrial-Musik.")
    # print(tokenizer.encode_batch(["<s>", "<s>"]))
    # print(tokenizer.token_to_id("<s>"))
    # print(a.tokens)
    # b = tokenizer.decode(a.ids)
    # print(b)
    tokenizer = make_tokenizer()
    # csv_to_txts()
