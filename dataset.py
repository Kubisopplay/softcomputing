



from dataset_creation import create_dataset, augment_dataset, process_dataset
import matplotlib.pyplot as plt

dataset = process_dataset(create_dataset())







num = 1
fig = plt.figure(figsize=(20, 14)) 
for image, label in iter(dataset['train']): 
    if num > 4: 
        break
    img = image.numpy() 
    fig.add_subplot(2, 2, num) 
    plt.imshow(img) 
    plt.axis('off') 
    plt.title((label)) 
    num += 1
plt.tight_layout() 
plt.show() 