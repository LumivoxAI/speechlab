import re

from torch import inference_mode
from runorm import RUNorm

from ...transport.model import BaseModel


class RuNormModel(RUNorm, BaseModel):
    def __init__(self) -> None:
        super().__init__()

        self.re_normalization = re.compile(r"[^a-zA-Z0-9\sа-яА-ЯёЁ.,!?:;" "''(){}[]«»„“”-]")

    def load(self, model_size: str, device: str, workdir: str) -> None:
        from transformers import (
            AutoTokenizer,
            BertForTokenClassification,
            T5ForConditionalGeneration,
            pipeline,
        )

        self.workdir = workdir

        self.model_size = model_size
        self.abbr_tokenizer = AutoTokenizer.from_pretrained(
            self.paths[model_size], cache_dir=self.workdir
        )
        self.abbr_model = T5ForConditionalGeneration.from_pretrained(
            self.paths[model_size], cache_dir=self.workdir
        )
        self.angl_tokenizer = AutoTokenizer.from_pretrained(
            self.paths["kirillizator"], cache_dir=self.workdir
        )
        self.angl_model = T5ForConditionalGeneration.from_pretrained(
            self.paths["kirillizator"], cache_dir=self.workdir
        )
        self.tagger_model = BertForTokenClassification.from_pretrained(
            self.paths["tagger"], cache_dir=self.workdir
        )
        self.tagger_tokenizer = AutoTokenizer.from_pretrained(
            self.paths["tagger"], cache_dir=self.workdir
        )
        self.tagger = pipeline(
            "ner",
            model=self.tagger_model,
            tokenizer=self.tagger_tokenizer,
            aggregation_strategy="average",
            device=device,
        )
        self.abbr_model.to(device)
        self.angl_model.to(device)
        self.abbr_model.eval()
        self.angl_model.eval()

    @inference_mode()
    def preprocess(self, text: str) -> str:
        return self.norm(text)
