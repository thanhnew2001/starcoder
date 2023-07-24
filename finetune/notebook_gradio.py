
from huggingface_hub import notebook_login
notebook_login()

import torch
import transformers
from transformers import GenerationConfig, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("starcoder-merged")

model = AutoModelForCausalLM.from_pretrained("starcoder-merged",
                                              load_in_8bit=True,
                                              device_map='auto',
                                              torch_dtype=torch.float16,
                                            use_auth_token= False
                                              )

tokenizer.eos_token, tokenizer.pad_token

tokenizer.pad_token_id = 0


import textwrap

def generate_response(input_prompt):
    system_prompt = "<|system|>\nBelow is a conversation between a human user and a helpful AI coding assistant.<|end|>\n"

    user_prompt = f"<|user|>\n{input_prompt}<|end|>\n"

    assistant_prompt = "<|assistant|>"

    full_prompt = system_prompt + user_prompt + assistant_prompt

    inputs = tokenizer.encode(full_prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(inputs,
                            eos_token_id = 0,
                            pad_token_id = 0,
                            max_length=256,
                            early_stopping=True)
    output =  tokenizer.decode(outputs[0])
    output = output[len(full_prompt):]
    if "<|end|>" in output:
        cutoff = output.find("<|end|>")
        output = output[:cutoff]
    print(input_prompt+'\n')
    # wrapped_text = textwrap.fill(output, width=100)
    # print(wrapped_text +'\n\n')
    print(output)
    return output

##converting to gradio web

import gradio as gr
demo = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Name Here..."),
    outputs="text",
)

demo.launch(server_name='0.0.0.0', share=True)



#generate_response("Write a python function that reverses each word in an input string")

#generate_response("What is Flask?")

#generate_response("def scrape_url(url):")

#generate_response("write a JS function that sorts a list alphabetically")
