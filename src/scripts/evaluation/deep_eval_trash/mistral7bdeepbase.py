from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM

class Mistral7B(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()

        device = "cuda" # the device to load the model onto

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
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
        return "Mistral 7B"

def get_testable_mistral():
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    mistral_7b = Mistral7B(model=model, tokenizer=tokenizer)
    return mistral_7b

if __name__ == '__main__':
    from huggingface_hub import login

    login(token="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")

    mistral_7b = get_testable_mistral()
    print(mistral_7b.generate("Write me a joke"))