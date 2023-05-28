# pip install -q transformers==4.29.2
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/tiny_starcoder_py"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Sane hyper-parameters
params = {
    'max_new_tokens': 128,
    'temperature': 0.2,
    'top_k': 50,
    'top_p': 0.1,
    'repetition_penalty': 1.17
}

# Prompt Style 1: Function Signature
inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs, pad_token_id=tokenizer.eos_token_id, **params)
print()
print(tokenizer.decode(outputs[0]))
print()

# Prompt Style 2: A comment
inputs = tokenizer.encode("# a python function that says hello\n", return_tensors="pt").to(device)
outputs = model.generate(inputs, pad_token_id=tokenizer.eos_token_id, **params)
print()
print(tokenizer.decode(outputs[0]))
print()

# Prompt Style 3: A docstring
inputs = tokenizer.encode("\"\"\" a python function that says hello \"\"\"\n", return_tensors="pt").to(device)
outputs = model.generate(inputs, pad_token_id=tokenizer.eos_token_id, **params)
print()
print(tokenizer.decode(outputs[0]))
print()

# Prompt Style 4: [ADVANCED] Fill in the middle
input_text = "<fim_prefix>def print_one_two_three():\n    print('one')\n    <fim_suffix>\n    print('three')<fim_middle>"
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, pad_token_id=tokenizer.eos_token_id, **params)
print(tokenizer.decode(outputs[0]))