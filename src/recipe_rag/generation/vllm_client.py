from __future__ import annotations


class VLLMClient:
    def __init__(self, model: str, **kwargs):
        self.model_name = model
        self.kwargs = kwargs
        try:
            from vllm import LLM, SamplingParams

            self._SamplingParams = SamplingParams
            self._llm = LLM(model=model, **kwargs)
        except Exception:
            self._SamplingParams = None
            self._llm = None

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        if self._llm is None:
            raise RuntimeError("vLLM is not available. Install recipe-rag-assistant[serve] and provide model weights.")
        params = self._SamplingParams(max_tokens=max_tokens, temperature=temperature)
        outputs = self._llm.generate([prompt], params)
        return outputs[0].outputs[0].text
