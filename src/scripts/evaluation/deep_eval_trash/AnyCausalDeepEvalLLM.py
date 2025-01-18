from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM

class AnyCausalDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(
        self,
        name:str
    ):
        self.name = name
        self.model = AutoModelForCausalLM.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        device = "cuda" # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)
        # Changing generate because we only want first token anyways
        generated_ids = model.generate(**model_inputs, max_new_tokens=1, do_sample=False, temperature=0.0)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)


    # I commented out the following because I don't know the shape that the prompts are supposed to be, so truncation and max_length are dubious. That's unfortunate because this sacrifices a lot of speed. 
    # Dan Hendrycks' code (evaluation/MMLU/test) did one-at-a-time, too.

    # This is optional.
    # def batch_generate(self, promtps: list[str]) -> list[str]:
    #     model = self.load_model()
    #     device = "cuda" # the device to load the model onto

    #     model_inputs = self.tokenizer(promtps, return_tensors="pt").to(device)
    #     model.to(device)

    #     generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
    #     return self.tokenizer.batch_decode(generated_ids)

    def get_model_name(self):
        return f"AnyDeepEval: {self.name}"