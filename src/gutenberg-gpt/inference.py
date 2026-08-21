import argparse

import torch
from gutenberg_tokenizer import GutenbergTokenizer
from model import LanguageModel
from paths import MODEL_PATH, TOKENIZER_PATH
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import Frame

parser = argparse.ArgumentParser()
parser.add_argument(
    "-o",
    "--output",
    required=False,
    help="File to write the output to. If omitted, output is not saved.",
    default=None
)

parser.add_argument(
    "-nt",
    "--nbtokens",
    required=False,
    help="Number of tokens to generate per round.",
    default=100,
    type=int
)

arguments = parser.parse_args()

number_tokens = arguments.nbtokens

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GutenbergTokenizer().load(str(TOKENIZER_PATH))
vocabulary_size = tokenizer.VOCABULARY_SIZE

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = LanguageModel(
    vocab_size=vocabulary_size,
    n_head=checkpoint["n_head"],
    n_layer=checkpoint["n_layer"],
    context_length=checkpoint["context_length"],
    n_embed=checkpoint["n_embed"],
    dropout_prob=checkpoint["dropout_prob"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


def inference(text: str) -> str:
    """
    Continue a piece of text with the model.

    Temperature and top-k are fixed here, only the number of tokens per round
    is configurable, through -nt/--nbtokens.

    Parameters
    ----------
    text : str
        Everything written so far, prompt and past output alike.

    Returns
    -------
    str
        `text` followed by the newly generated tokens, decoded.
    """
    idx = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
    generated_text = tokenizer.decode(
        model.generate(idx, number_tokens, EOT=tokenizer.EOT, temperature=1, top_k=10)[0].tolist()
    )
    return generated_text


def run_cli() -> None:
    """
    Run the REPL until the user submits an empty line or hits Ctrl-C / Ctrl-D.

    Each round feeds the model everything written so far, the typed text and
    the model's own output, prints only the new tail and keeps the whole string
    for the next round. With -o/--output the running text is rewritten to that
    file after every round, otherwise nothing touches the disk.
    """
    kb = KeyBindings()

    @kb.add("enter")
    def _(event: KeyPressEvent) -> None:
        event.app.exit(result=event.app.buffer.text)

    @kb.add("c-c")
    @kb.add("c-d")
    def _(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    def prompt() -> str | None:
        """
        Show a one-line input frame and wait for the user.

        Returns
        -------
        str or None
            The typed text, or None when the user asked to quit with Ctrl-C or
            Ctrl-D.
        """
        buffer = Buffer(multiline=False)
        zone = VSplit(
            [
                Window(FormattedTextControl("› "), width=2),
                Window(
                    BufferControl(buffer),
                    height=Dimension(min=1, max=1),
                    wrap_lines=True,
                ),
            ]
        )
        app = Application(
            layout=Layout(Frame(zone)),
            key_bindings=kb,
            full_screen=False,
            erase_when_done=True,
        )
        app.buffer = buffer
        return app.run()

    initial_text = ""
    while True:
        print()
        prompted = prompt()
        if not prompted:
            break
        text = inference(initial_text + prompted)
        print(text[len(initial_text):], end="", flush=True)
        initial_text = text
        if arguments.output:
            with open(arguments.output, "w") as file:
                file.write(text)


if __name__ == "__main__":
    run_cli()
