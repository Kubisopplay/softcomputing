import transformers
from transformers import  ViTForImageClassification
import keras
from dataset_creation import create_dataset, augment_dataset, process_dataset
import tensorflow as tf

dataset = create_dataset()

config = transformers.ViTConfig(
    image_size=64,
    num_classes=43,
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=12,
    intermediate_size=3072,
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

model = ViTForImageClassification(config=config)

