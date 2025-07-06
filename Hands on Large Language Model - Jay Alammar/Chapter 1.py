from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

# load model and tokenizer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map = "cuda",
    torch_dtype = "auto",
    trust_remote_code = True
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# create pipeline

generator = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    return_full_text = False,
    max_new_tokens = 500,
    do_sample = False
)

# The prompt

messages = [
    {"role":"user","content":"Create a funny joke about data scientist"}
]

# Generate output

output = generator(messages)
print(output[0]["generated_text"])