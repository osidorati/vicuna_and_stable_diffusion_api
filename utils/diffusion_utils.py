import os
import random
import torch
from torch import autocast

from config import Config

config = Config()


def dummy_checker(images, **kwargs): return images, False


def generate_image(pipe, prompt, num, seed=None, height=config.height, width=config.width, num_inference_steps=config.num_inferenсe_steps, strength=config.strength, guidance_scale=config.guidance_scale):
    if not config.safety_checker:
        pipe.safety_checker = dummy_checker

    seed = seed if seed is not None else random.randint(1, 1000000)
    generator = torch.cuda.manual_seed_all(seed)
    pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    #neg = 'ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face.'
    #neg = '[extra nipples, bad anatomy, blurry, fuzzy, extra arms, extra fingers, poorly drawn hands, disfigured, tiling, deformed, mutated, out of frame, cloned face, ugly, disfigured, bad proportion, out of frame, b&w, painting, drawing, watermark, logo, text, signature, icon, monochrome, blurry, ugly, cartoon, 3d, bad_prompt, long neck, totem pole, multiple heads, multiple jaws, disfigured, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, username, artist name, ancient, character, frame, child, asian, cartoon, animation, grayscale 3d, disfigured, bad art, deformed, poorly drawn, extra limbs, strange colours, boring, sketch, lackluster, repetitive, cropped'
    #neg = '[split image, out of frame, amputee, mutated, mutation, deformed, severed, dismembered, corpse, photograph, poorly drawn, bad anatomy, blur, blurry, lowres, bad hands, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, ugly, symbol, hieroglyph,, extra fingers,  six fingers per hand, four fingers per hand, disfigured hand, monochrome, missing limb, disembodied limb, linked limb, connected limb, interconnected limb,  broken finger, broken hand, broken wrist, broken leg, split limbs, no thumb, missing hand, missing arms, missing legs, fused finger, fused digit, missing digit, bad digit, extra knee, extra elbow, storyboard, split arms, split hands, split fingers, twisted fingers, disfigured butt]'
    #neg = 'split image, out of frame, amputee, mutated, mutation, deformed, severed, dismembered, corpse, photograph, poorly drawn, bad anatomy, blur, blurry, lowres, bad hands, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, artist name, ugly, symbol, hieroglyph,, extra fingers,  six fingers per hand, four fingers per hand, disfigured hand, monochrome, missing limb, disembodied limb, linked limb, connected limb, interconnected limb,  broken finger, broken hand, broken wrist, broken leg, split limbs, no thumb, missing hand, missing arms, missing legs, fused finger, fused digit, missing digit, bad digit, extra knee, extra elbow, storyboard, split arms, split hands, split fingers, twisted fingers, disfigured butt'
    neg = '(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra lim, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation.'
    images = []
    while num != 0:
        with autocast("cuda"):
            image = pipe(prompt=[prompt], negative_prompt=[neg], generator=generator, height=height, width=width, guidance_scale=guidance_scale,
                                    num_inference_steps=num_inference_steps)["images"][0]
            num -= 1
            images.append(image)
    return images