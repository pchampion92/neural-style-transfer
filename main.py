import os.path

import tensorflow as tf
from matplotlib.pyplot import imshow, show

from constants import IMG_SIZE, LAYERS_LIST
from utils import load_content_style_image, load_vgg_CNN, initialize_generated_image, generate_selected_model, \
    tensor_to_image, \
    compute_content_cost, compute_style_cost, compute_cost


def run_simulation(path_image_content, path_image_style, path_image_output):
    # Load content and style images
    content_image, style_image = load_content_style_image(path_image_content, path_image_style, image_size=IMG_SIZE)

    # Load CNN to compute activation layers
    vgg_model = load_vgg_CNN(input_shape=content_image.shape[1:])
    vgg_model_selected = generate_selected_model(vgg_model, layers_list=LAYERS_LIST)

    # Compute content and style output
    # TODO preprocess (why?)
    content_output = vgg_model_selected(tf.image.convert_image_dtype(content_image, tf.float32))
    style_output = vgg_model_selected(tf.image.convert_image_dtype(style_image, tf.float32))

    # Initialize generated image
    generated_image = initialize_generated_image(content_image)
    generated_image = tf.Variable(generated_image) #N.B.: variable need to be instantiated in the same script (weird)
    image_result = tensor_to_image(generated_image)
    image_result.save(os.path.join(path_image_output, f"output_0.jpg"))

    # Neural Style Transfer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    for epochs in range(11):
        with tf.GradientTape() as tape:
            generated_output = vgg_model_selected(generated_image)
            content_cost = compute_content_cost(content_output, generated_output)
            style_cost = compute_style_cost(style_output, generated_output, style_layers_weights=LAYERS_LIST)
            cost = compute_cost(content_cost, style_cost)
        gradient = tape.gradient(cost, generated_image)
        optimizer.apply_gradients([(gradient, generated_image)])
        generated_image.assign(
            tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
        )
        if epochs % 5 == 0:
            image_result = tensor_to_image(generated_image)
            image_result.save(os.path.join(path_image_output, f"output_{epochs}.jpg"))
            imshow(image_result)
            show()


if __name__ == '__main__':
    run_simulation(
        path_image_content=r'input/content.jpg',
        path_image_style=r'input/style.jpg',
        path_image_output='output'
    )
    # from PIL import Image
    # import numpy as np
    #
    # content = np.array(Image.open(r'input/content.jpg'))
    # style = np.array(Image.open(r'input/style.jpg'))
    #
    # from utils import load_content_style_image
    # content,style=load_content_style_image(
    #     path_image_content=r'input/content.jpg',
    #     path_image_style=r'input/style.jpg',
    #     image_size=IMG_SIZE
    # )
