import gradio as gr
import tensorflow as tf
import numpy as np


model = model = tf.keras.models.load_model("model_10_transfer_finetuning.keras")

IMG_SIZE = [128, 128]

class_names = ['Dog', 'Horse', 'Elephant', 'Butterfly', 'Chicken', 'Cat', 'Cow', 'Sheep', 'Spider', 'Squirrel']

def predict_animal(gradio_image):
    image = tf.convert_to_tensor(gradio_image, dtype=tf.float32)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)

    predictions = model.predict(image)[0]
    return {class_names[i]: float(predictions[i]) for i in range(len(class_names))}

demo = gr.Interface(
    fn=predict_animal,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    title="What's that animal?!"
)

demo.launch(share=True, debug=True)
