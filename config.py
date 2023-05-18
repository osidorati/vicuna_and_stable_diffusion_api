import os
from dotenv import load_dotenv

load_dotenv('.env')

class Config:
    def __init__(self) -> None:
        self.device = os.getenv("DEVICE", "cuda")
        self.model_vicuna = os.getenv("MODEL_VICUNA", "helloollel/vicuna-7b")
        self.num_gpus = os.getenv("NUM_GPUS", 1)
        self.load_8bit = True if os.getenv("LOAD_8BIT") else False
        self.max_new_tokens = os.getenv("MAX_NEW_TOKENS")
        self.temperature = os.getenv("TEMPERATURE")

        self.model_sd = os.getenv('MODEL_SD', 'stablediffusionapi/deliberate-v2')
        self.low_vram_mode = True if os.getenv('LOW_VRAM') else False
        self.safety_checker = True if os.getenv('SAFETY_CHECKER') else False
        self.height = int(os.getenv('HEIGHT', '1024'))
        self.width = int(os.getenv('WIDTH', '1024'))
        self.num_inferenсe_steps = int(os.getenv('NUM_INFERENCE_STEPS', '30'))
        self.strength = float(os.getenv('STRENTH', '0.75'))
        self.guidance_scale = float(os.getenv('GUIDANCE_SCALE', '7.5'))
        self.full_filename = os.getenv('FULL_FILENAME')
        self.sub_dir = os.getenv('SUB_DIR', 'img/')


    def __str__(self) -> str:
        return str(self.__dict__)