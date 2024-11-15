from datasets import load_dataset
import keras
import tensorflow as tf
from transformers import ViTImageProcessor, ViTImageProcessorFast, ViTForImageClassification


def create_dataset():
    #dataset = load_dataset('imagefolder', data_dir='train/GTSRB/Final_Training')
    dataset = load_dataset('imagefolder', data_dir='train_cropped',num_proc=64)
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

def process_dataset(dataset):
    feature_extractor = ViTImageProcessor(
        image_size=64,
        do_normalize=True,
        do_resize=True,
        size=64,
    )
    dataset = dataset.map(lambda x: 
        x.update(feature_extractor(x['image']),)
    , batched=True)
    print("Dataset processed")
    return dataset

if __name__ == '__main__':
    dataset = create_dataset()
    #dataset = augment_dataset(dataset)
    dataset = process_dataset(dataset)
    
    print(dataset)
