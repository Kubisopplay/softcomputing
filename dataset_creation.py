from datasets import load_dataset
import keras
from transformers import ViTImageProcessor, ViTForImageClassification

feature_extractor = ViTImageProcessor()

def create_dataset():
    #dataset = load_dataset('imagefolder', data_dir=f'train/GTSRB/Final_Training')
    dataset = load_dataset('imagefolder', data_dir=f'train_cropped')
    return dataset

def augment_dataset(dataset):
    
    data_augumentation = keras.Sequential([
        keras.layers.preprocessing.RandomFlip("horizontal"),
        keras.layers.preprocessing.RandomRotation(0.1),
        keras.layers.preprocessing.RandomZoom(0.1),
    ])
    dataset_augmented = dataset.map(lambda x: { 'image': data_augumentation(x['image']), 'label': x['label'] })
    return dataset_augmented

def process(dataset):
    dataset_processed = dataset.copy()
    dataset_processed.update({
        'image': feature_extractor(dataset['image'], return_tensors='pt')['pixel_values']
    })
    return dataset_processed

if __name__ == '__main__':
    dataset = create_dataset()
    dataset = augment_dataset(dataset)
    dataset = dataset.map(process)
    print(dataset)
