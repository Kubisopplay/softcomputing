import transformers
from transformers import  TFViTForImageClassification
import tf_keras
import tensorflow as tf
from dataset_creation import create_dataset, augment_dataset, process_dataset

import json
if __name__ == "__main__":

    continue_training = False
    
    continue_from = 0
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    dataset = process_dataset(create_dataset())
    num_classes = len(set(dataset['train']['label']))
    print(num_classes)
    if not continue_training:
        config = transformers.ViTConfig(
            image_size=64,
            num_channels=3, 
            num_labels =num_classes,
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=16,
            intermediate_size=2048,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            is_encoder_decoder=False,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            )

        model = TFViTForImageClassification(config=config)


    optimizer = tf_keras.optimizers.Adadelta()

    loss = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    metrics=[ 
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name="top-3-accuracy"),
    ]

    
    def convert_to_tf_dataset(dataset):
        def gen():
            for example in dataset:
                image = example['image']  # Assuming image shape (64, 64, 3)
                # Transpose to (3, 64, 64)
                image = tf.transpose(image, perm=[2, 0, 1])
                label = example['label']
                yield image, label

        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(3, 64, 64), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int64),
            )
        )


    num_samples = len(dataset['train'])  # Total number of samples in the training set
    batch_size = 32  # Your current batch size
    steps_per_epoch = num_samples // batch_size


    
    dataset= dataset['train'].train_test_split(test_size=0.1)

    #train_dataset = convert_to_tf_dataset(dataset["train"]).batch(batch_size)
    #test_dataset = convert_to_tf_dataset(dataset["test"]).batch(batch_size)

    import os
    #train_dataset = dataset["train"].to_tf_dataset(batch_size=batch_size,columns=["image"], label_cols=["label"])
    #test_dataset = dataset["test"].to_tf_dataset(batch_size=batch_size, columns=["image"],label_cols=["label"])

    train_dataset = dataset["train"].with_format("tf")
    test_dataset = dataset["test"].with_format("tf")
    #tf.config.run_functions_eagerly(True)

    
    if continue_training:
        config = transformers.ViTConfig.from_pretrained("model" + str(continue_from))
        model = TFViTForImageClassification.from_pretrained("model" + str(continue_from), config=config)
    model.compile(loss=loss,metrics=metrics, optimizer=optimizer)

    results = []
    epochs = 25
    for i in range(continue_from,epochs+continue_from):
        model.fit(train_dataset, epochs=1, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
        results.append(model.evaluate(test_dataset, batch_size=batch_size))
        model.save_pretrained('model'+str(i+continue_from))
        with open("results.json", "w", encoding="UTF-8") as file:
            file.write(json.dumps(results)) 
   
    print(results)  
    
    


