import torch
from torch import Tensor

from .decoder import RNNTHead


class Tokenizer:
    """
    Tokenizer for converting between text and token IDs.
    The tokenizer can operate either character-wise or using a pre-trained SentencePiece model.
    """

    def __init__(self, vocab: list[str], model_path: str | None = None) -> None:
        self.charwise = model_path is None
        if self.charwise:
            self.vocab = vocab
        else:
            from sentencepiece import SentencePieceProcessor

            self.model = SentencePieceProcessor()
            self.model.load(model_path)

    def decode(self, tokens: list[int]) -> str:
        """
        Convert a list of token IDs back to a string.
        """
        if self.charwise:
            return "".join(self.vocab[tok] for tok in tokens)
        return self.model.decode(tokens)

    def __len__(self) -> int:
        """
        Get the total number of tokens in the vocabulary.
        """
        return len(self.vocab) if self.charwise else len(self.model)


class RNNTGreedyDecoding:
    def __init__(
        self,
        vocabulary: list[str],
        model_path: str | None = None,
        max_symbols_per_step: int = 10,
    ) -> None:
        """
        Class for performing greedy decoding of RNN-T outputs.
        """
        self.tokenizer = Tokenizer(vocabulary, model_path)
        self.blank_id = len(self.tokenizer)
        self.max_symbols = max_symbols_per_step

    def _greedy_decode(self, head: RNNTHead, x: Tensor, seqlen: Tensor) -> str:
        """
        Internal helper function for performing greedy decoding on a single sequence.
        """
        hyp: list[int] = []
        dec_state: Tensor | None = None
        last_label: Tensor | None = None
        for t in range(seqlen):
            f = x[t, :, :].unsqueeze(1)
            not_blank = True
            new_symbols = 0
            while not_blank and new_symbols < self.max_symbols:
                g, hidden = head.decoder.predict(last_label, dec_state)
                k = head.joint.joint(f, g)[0, 0, 0, :].argmax(0).item()
                if k == self.blank_id:
                    not_blank = False
                else:
                    hyp.append(k)
                    dec_state = hidden
                    last_label = torch.tensor([[hyp[-1]]]).to(x.device)
                    new_symbols += 1

        return self.tokenizer.decode(hyp)

    @torch.inference_mode()
    def decode(self, head: RNNTHead, encoded: Tensor, enc_len: Tensor) -> list[str]:
        """
        Decode the output of an RNN-T model into a list of hypotheses.
        """
        b = encoded.shape[0]
        pred_texts = []
        encoded = encoded.transpose(1, 2)
        for i in range(b):
            inseq = encoded[i, :, :].unsqueeze(1)
            pred_texts.append(self._greedy_decode(head, inseq, enc_len[i]))
        return pred_texts
