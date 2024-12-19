import gradio

import tensorflow as tf
import transformers
from transformers import  TFViTForImageClassification
import tf_keras
import numpy
model_path = "./model69/"
model = TFViTForImageClassification.from_pretrained(model_path)
#config = transformers.ViTConfig.from_pretrained(model_path)
image_processor = transformers.ViTImageProcessor(
    image_size=64,
    do_normalize=True,
    do_resize=True,
    size=64,
)
import json
mapping = json.load(open("folder_names.json"))



def classify_image(image):
    inputs = image_processor(images=image, return_tensors="tf")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = int(tf.math.argmax(logits, axis=-1))
    
    
    return mapping[str(predicted_class_idx)]

interface = gradio.Interface(
    fn=classify_image,
    inputs=gradio.Image(type="numpy"),
    outputs="label",
    title="Image Classification",
    description="Upload an image to classify"
)

interface.launch()