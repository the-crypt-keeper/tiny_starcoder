# pip install -q transformers==4.29.2
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from jinja2 import Template

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

input_template = Template("""<fim_prefix>def {{Signature}}:
    '''a function {{Input}} that returns {{Output}}{% if Fact %} given {{Fact}}{% endif %}'''
    <fim_suffix>

# another function
<fim_middle>""")

output_template = Template("""def {{Signature}}:
    '''a function {{Input}} that computes {{Output}}'''
    {{Answer}}""")

# Load interview
interview = yaml.safe_load(open("tiny-interview.yml"))
for name, challenge in interview.items():
    
    challenge['name'] = name
    input = input_template.render(**challenge)

    print(input)
    
    inputs = tokenizer.encode(input, return_tensors="pt").to(device)
    outputs = model.generate(inputs, pad_token_id=tokenizer.eos_token_id, **params)

    result = tokenizer.decode(outputs[0])
    result = result.replace(input, '').replace('<|endoftext|>','')
    
    output = output_template.render(**challenge, Answer=result)

    print()
    print(output)
    print()

    with open(f"{name}.txt", "w") as f:
        f.write(output)
