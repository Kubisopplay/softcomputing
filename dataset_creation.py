from datasets import load_dataset
import tf_keras as keras
import tensorflow as tf
from transformers import ViTImageProcessor, ViTImageProcessorFast, ViTForImageClassification


def create_dataset():
    #dataset = load_dataset('imagefolder', data_dir='train/GTSRB/Final_Training')
    datafiles = { "train": "train_images/train/**" }
    dataset = load_dataset('imagefolder', data_dir="train_images", drop_labels=False)
    #dataset = dataset.train_test_split(test_size=0.95)["train"]
    #dataset["train"] = dataset
    print("Dataset loaded")
    return dataset

def augment_dataset(dataset):
    
    data_augumentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ])
    dataset_augmented = dataset.map(lambda x: { 
        'image': data_augumentation(tf.convert_to_tensor(x['image'])), 
        'label': x['label'] 
    })
    return dataset_augmented


def mapper(batch, processor):
    images = [img for img in batch["image"]]
    processed_images = processor(images=images, return_tensors="tf")["pixel_values"]
    processed_images = tf.transpose(processed_images, perm=[0, 3, 1, 2])
    return { "image": processed_images, "label": batch["label"] }
def transpose_image(image):
    return tf.transpose(image, perm=[1, 2, 0])
def process_dataset(dataset):
    feature_extractor = ViTImageProcessor(
        image_size=64,
        do_normalize=True,
        do_resize=True,
        size=64,
    )
    dataset = dataset.map(mapper, fn_kwargs={"processor": feature_extractor}, batched=True)
    print("Dataset processed")
    return dataset

if __name__ == '__main__':
    dataset = create_dataset()
    print(dataset["train"][3500]["image"].filename)
    print(dataset["train"][3500]["label"])
    #dataset = augment_dataset(dataset)
    dataset = process_dataset(dataset)
    
    #most horrible workaround ever, kill the person who wrote this (me)
    
    label_to_id = {}
    
    for i in range(len(dataset["train"])):
        image = dataset["train"][i]["image"]
        label = dataset["train"][i]["label"]
        path = image.filename.split("/")
        dir = path[-2]
        if dir not in label_to_id:
            label_to_id[dir] = label
            print(label," ",dir)
        else:
            if label != label_to_id[dir]:
                print("Kurwa")
                print(label)
                print(label_to_id[dir])
                print(image.filename)
        
    import json
    with open('label_to_id.json', 'w') as fp:
        json.dump(label_to_id, fp)