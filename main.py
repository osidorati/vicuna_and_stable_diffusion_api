import os
from os import getcwd

import re
import demoji

import torch

import uuid

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse

from config import Config

from diffusers import StableDiffusionPipeline

from utils.conversation import conv_templates, SeparatorStyle

from utils.vicuna_utils import load_model, generate_stream
from utils.diffusion_utils import generate_image

config = Config()
print("Using config: ", config)

revision = "fp16" if config.low_vram_mode else None
revision = None
torch_dtype = torch.float16


class TextRequest(BaseModel):
    topic: str
    description: str


class TextRequest2(BaseModel):
    topic: str
    description: str
    num: int


class TextResponse(BaseModel):
    result: str


class DictResponse(BaseModel):
    result: List[str]


app = FastAPI()

conv_template = 'vicuna_v1.1'

model, tokenizer = load_model(config.model_vicuna, config.device,
                              config.num_gpus, config.load_8bit)

pipe = StableDiffusionPipeline.from_pretrained(config.model_sd, torch_dtype=torch_dtype, revision=None)
pipe = pipe.to("cuda")


def clean(res):
    print(res)
    # clean symbols / [] ""
    res = re.sub('["|/|\]|\[]', "", res)
    # delete paragraphs
    res = res.replace('\n', ' ')
    # delete emoji
    res = demoji.replace(res, '')
    # delete hashtags
    res = re.sub("#\S+", "", res)
    # delete bouble whitespacing
    res = " ".join(res.split())
    return res


@app.post("/predict", response_model=TextResponse)
def predict(text_request: TextRequest):
    print(text_request.description)
    base_prompt3 = "You're a tiktok blogger. Write a text for vidie on this theme: {}. description: {}.".format(
        text_request.topic, text_request.description)
    # base_prompt2 = "You're a tiktok blogger. What do you say in the video on this theme: {}. description: {}.".format(text_request.topic, text_request.description)
    # base_prompt = "write solid text for tiktok video on this theme: {}. description: {}. don't use square brackets in your answer. don't use quotes in your answer".format(text_request.topic, text_request.description)
    conv = conv_templates[conv_template].copy()
    conv.append_message(conv.roles[0], base_prompt3)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    params = {
        "model": config.model_vicuna,
        "prompt": prompt,
        "temperature": config.temperature,
        "max_new_tokens": config.max_new_tokens,
        "stop": conv.sep2,
    }

    pre = 0
    for outputs in generate_stream(tokenizer, model, params, config.device):
        outputs = outputs[len(prompt) + 1:].strip()
        outputs = outputs.split(" ")
        now = len(outputs)
        if now - 1 > pre:
            pre = now - 1

    data = clean(" ".join(outputs))

    return TextResponse(result=data)


@app.post("/create_img", response_model=DictResponse)
def create_image(text_request: TextRequest2):
    prompt = 'picture {} {} 8k textures, extreme detail, high sharpness.'.format(text_request.topic,
                                                                                 text_request.description)

    img = generate_image(pipe, prompt, text_request.num)
    filenames = []
    for i in img:
        filename = str(uuid.uuid4().int)[:8] + '.jpg'
        i.save(config.sub_dir + filename)
        print(config.full_filename + filename)
        filenames.append(config.full_filename + filename)
    print(filenames)

    return DictResponse(result=filenames)


@app.get('/images/{name_file}')
def imaging(name_file: str):
    print(getcwd() + "/img/" + name_file)
    return FileResponse(path=getcwd() + '/' + config.sub_dir + name_file)
