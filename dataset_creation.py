from datasets import load_dataset
import keras
import tensorflow as tf
from transformers import ViTImageProcessor, ViTImageProcessorFast, ViTForImageClassification


def create_dataset():
    #dataset = load_dataset('imagefolder', data_dir=f'train/GTSRB/Final_Training')
    dataset = load_dataset('imagefolder', data_dir=f'train_cropped')
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

def process_dataset(dataset):
    
    feature_extractor = ViTImageProcessor(
        image_size=64,
        do_normalize=False,
        do_resize=False,
        do_rescale=False,
        )
    dataset = dataset.map(lambda x: {
        'image': feature_extractor.preprocess(x['image'], return_tensors="tf")['pixel_values'],
        'label': x['label']
    }, batched=True)
    #dataset.rename_column_("image", "images")
    #dataset.set_format("tensorflow")
    return dataset
if __name__ == '__main__':
    dataset = create_dataset()
    #dataset = augment_dataset(dataset)
    dataset = process_dataset(dataset)
    
    print(dataset)
