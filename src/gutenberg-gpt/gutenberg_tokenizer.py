import os
import random
from pathlib import Path

from config import TRAINING_CONFIG
from paths import DATA_DIRECTORY, TOKENIZER_PATH
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


class GutenbergTokenizer:
    """
    A byte-level BPE tokenizer for the French Gutenberg corpus.

    Wraps a Hugging Face `Tokenizer` and exposes only what the rest of the
    codebase needs. Build one either by training it from scratch with `train`
    or by reading a saved one back with `load`, both of which fill in the
    attributes below.

    Attributes
    ----------
    tokenizer : tokenizers.Tokenizer
        The wrapped Hugging Face tokenizer.
    VOCABULARY_SIZE : int
        Number of tokens in the vocabulary, special token included.
    EOT : int
        Id of the ``<|endoftext|>`` token.
    """

    tokenizer: Tokenizer
    VOCABULARY_SIZE: int
    EOT: int

    def train(
        self, train_files: list[str], destination: str | Path
    ) -> "GutenbergTokenizer":
        """
        Train a fresh BPE and save it to disk.

        Vocabulary of 16384 tokens, ``<|endoftext|>`` as the only special token.
        Only training books should be passed here, otherwise the validation set
        leaks into the vocabulary.

        Parameters
        ----------
        train_files : list of str
            Paths of the text files to learn the merges from.
        destination : str or pathlib.Path
            Where to write the tokenizer JSON.

        Returns
        -------
        GutenbergTokenizer
            Self, so the call can be chained.
        """
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
        """
        Turn text into token ids.

        No EOT is appended on purpose: doing it here would push one into every
        prompt. `build_bin` in train.py adds it once per book instead.

        Parameters
        ----------
        sequences : str
            Text to encode.

        Returns
        -------
        list of int
            The token ids, in order.
        """
        tokens = self.tokenizer.encode(sequences).ids
        #tokens.append(self.EOT)
        return tokens

    def decode(self, ids: list[int]) -> str:
        """
        Turn token ids back into text.

        Parameters
        ----------
        ids : list of int
            Token ids to decode.

        Returns
        -------
        str
            The decoded text.
        """
        return self.tokenizer.decode(ids)

    def load(self, path: str) -> "GutenbergTokenizer":
        """
        Read a saved tokenizer back from disk.

        Parameters
        ----------
        path : str
            Path of the tokenizer JSON written by `train`.

        Returns
        -------
        GutenbergTokenizer
            Self, so the call can be chained.
        """
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
