import logging
import time
import warnings

from src.models.base import ModelRunner


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class TransformersRunner(ModelRunner):
    """Modelo local de Hugging Face (GPT-2 base) en CPU.

    do_sample=False es decodificación greedy aunque se pase temperature: el valor se envía igual a
    generate() para que quede registrado que se ignora (y el aviso de la librería, en `notices`).
    """

    def __init__(
        self, tokenizer, model, *, do_sample: bool = False, seed: int | None = None, max_new_tokens: int = 60,
        record_ids: bool = False,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.do_sample = do_sample
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.record_ids = record_ids

    @classmethod
    def from_pretrained(cls, model_name: str, **options) -> "TransformersRunner":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(model_name), AutoModelForCausalLM.from_pretrained(model_name).eval(), **options)

    def clone(self, **options) -> "TransformersRunner":
        """Otro runner sobre el mismo modelo ya cargado, con opciones distintas."""
        merged = {"do_sample": self.do_sample, "seed": self.seed, "max_new_tokens": self.max_new_tokens, "record_ids": self.record_ids, **options}
        return TransformersRunner(self.tokenizer, self.model, **merged)

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        effort: str | None = None,
        structured_schema: dict | None = None,
    ) -> dict:
        if effort is not None or structured_schema is not None:
            raise NotImplementedError("TransformersRunner no soporta effort ni structured_schema.")
        import torch

        if self.seed is not None:
            torch.manual_seed(self.seed)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        kwargs = {"do_sample": self.do_sample}
        if self.record_ids:
            # Los logits crudos permiten comprobar cada paso contra la distribución, no solo comparar textos.
            kwargs.update(return_dict_in_generate=True, output_logits=True)
        for name, value in (("temperature", temperature), ("top_p", top_p), ("top_k", top_k)):
            if value is not None:
                kwargs[name] = value

        capture = _LogCapture()
        hf_logger = logging.getLogger("transformers")
        hf_logger.addHandler(capture)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                start = time.perf_counter()
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id, **kwargs
                    )
                latency = time.perf_counter() - start
        finally:
            hf_logger.removeHandler(capture)

        prompt_length = inputs["input_ids"].shape[1]
        extras = {}
        if self.record_ids:
            sequences, step_logits = output.sequences, output.logits
            new_ids = sequences[0, prompt_length:]
            extras = {
                "output_ids": [int(i) for i in new_ids],
                "matches_argmax": all(int(torch.argmax(step[0])) == int(new_ids[i]) for i, step in enumerate(step_logits)),
            }
        else:
            new_ids = output[0, prompt_length:]
        return {
            "text": self.tokenizer.decode(new_ids, skip_special_tokens=True),
            "input_tokens": int(prompt_length),
            "output_tokens": int(new_ids.shape[0]),
            "reasoning_tokens": None,
            "latency_seconds": latency,
            "raw_response": {},
            "notices": [str(w.message) for w in caught] + capture.messages,
            "extras": extras,
        }
