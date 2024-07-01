# pip install git+https://github.com/huggingface/transformers
import torch
import requests
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

import warnings
warnings.filterwarnings(action='ignore')

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
image = [Image.open(requests.get(url, stream=True).raw).convert("RGB")]
prompt = ["What is unusual about this image?"]
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
)
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)
