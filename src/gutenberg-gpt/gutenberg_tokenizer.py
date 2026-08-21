import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import random
from paths import DATA_DIRECTORY, TOKENIZER_PATH
from config import TRAINING_CONFIG


class GutenbergTokenizer:
    def train(self, train_files: str, destination: str):
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=16384,
            special_tokens=["<|endoftext|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            max_token_length=16,
        )
        tokenizer.train(train_files, trainer)
        tokenizer.save(f"{destination}")
        self.tokenizer = tokenizer
        self.VOCABULARY_SIZE = tokenizer.get_vocab_size()
        self.EOT = tokenizer.token_to_id("<|endoftext|>")
        return self

    def encode(self, sequences: str) -> list[int]:
        tokens = self.tokenizer.encode(sequences).ids
        #tokens.append(self.EOT)
        return tokens

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def load(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)
        self.EOT = self.tokenizer.token_to_id("<|endoftext|>")
        self.VOCABULARY_SIZE = self.tokenizer.get_vocab_size()
        return self


if __name__ == "__main__":
    tokenizer = GutenbergTokenizer()
    data_directory = DATA_DIRECTORY
    destination = TOKENIZER_PATH
    data_files = [
        f"{data_directory}/{file_name}" for file_name in os.listdir(data_directory)
    ]
    n_train = int(len(data_files) * TRAINING_CONFIG["train_split"])
    train_files = data_files[:n_train]
    tokenizer = tokenizer.train(
        random.sample(train_files, int(n_train * 0.05)), destination=destination
    )
