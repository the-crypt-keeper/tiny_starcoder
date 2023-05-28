# tiny_starcoder

A 159M parameter mode that can write Python? LFG! :rocket:

This repository contains working python example code for interacting with https://huggingface.co/bigcode/tiny_starcoder_py

`pip install transformers==4.29.2` or you will get errors about this model type not being recognized.

Set `device = "cpu"` if you do not have a CUDA capable GPU.

## Prompt Style 1: Function Signature

This is not a multi-billion parameter chat model, it's a tiny specialized code generation model and has to be prompted correctly.

The simplest possible prompt is a function signature:

```def print_hello_world():```

This returns:

```
def print_hello_world():
    """Prints hello world"""

    print("Hello World!")


if __name__ == "__main__":
    main()
<|endoftext|>
```

Notes:

* <|endoftext|> is a special character for this model
* It frequently puts in nonsense main or test code after what has been requested

## Prompt Style 2: A comment

Another possible prompt style is a comment that describes what this function does:

```# a python function that says hello```

This returns:

```# a python function that says hello
def say_hello():
    print("Hello World!")

if __name__ == "__main__":
    say_hello()<|endoftext|>    
```

Note:
*  This style can be combined with Style 1, you can give both a comment and a function signature.

## Prompt Style 3: A Docstring

This is python after all, so this model understands docstrings:

```""" a python function that says hello """```

This returns:

```
""" a python function that says hello """
def say_hello():
    print("Hello World!")

<|endoftext|>
```

Note:

* This prompt style did not elicit the post-generation garbage, this is worth exploring further

## Prompt Style 4: Fill in the middle

```
<fim_prefix>def print_one_two_three():
    print('one')
    <fim_suffix>
    print('three')<fim_middle>
```

Returns:

```
<fim_prefix>def print_one_two_three():
    print('one')
    <fim_suffix>
    print('three')<fim_middle>print('two')<|endoftext|>
```

The model filled in the 'two' in the middle.

Note:
* The example on the model page uses the wrong special tokens.  The example in this repository is correct.