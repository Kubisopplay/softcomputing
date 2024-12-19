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
    
    
    return str(predicted_class_idx)


from dataset_creation import create_dataset, augment_dataset, process_dataset

dataset = process_dataset(create_dataset())


all_images = []
for line in dataset['train']:
    image = line["image"]
    label = line["label"]
    print(image.filename)
    temp = {}
    temp['image'] = image.filename
    temp["actual_label"] = image.filename.split("/")[-2]
    temp["label"] = label
    temp["recognized"] = classify_image(image)
    temp["mapped_label"] = mapping[temp["recognized"]]
    all_images.append(temp)

import json
with open('recognized.json', 'w') as fp:
    json.dump(all_images, fp)
    