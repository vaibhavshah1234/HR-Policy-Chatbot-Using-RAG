from langchain.llms.base import LLM
from ctransformers import AutoModelForCausalLM
from pydantic import BaseModel, Field
from typing import Any
from app.config import MODEL_PATH

class CTransformerLLM(LLM, BaseModel):
    model: Any = Field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "model", AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            model_type="mistral",
            max_new_tokens=512,
            temperature=0.7,
            context_length=4096
        ))

    def _call(self, prompt: str, stop=None) -> str:
        return self.model(prompt)

    @property
    def _llm_type(self) -> str:
        return "ctransformers"
