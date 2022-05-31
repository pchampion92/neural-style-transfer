import numpy as np
import tensorflow as tf
from PIL import Image


def load_image(image_path):
    return np.array(Image.open(image_path)) / 255.0


def test_image_rgb(tensor):
    if np.ndim(tensor) == 3:
        assert tensor.shape[2] == 3, "Not three channels"
    else:
        raise ValueError('Image is not RGB')


def load_content_style_image(path_image_content, path_image_style, image_size):
    content_image = load_image(path_image_content)
    test_image_rgb(content_image)
    style_image = load_image(path_image_style)
    test_image_rgb(style_image)
    # Reduce content image to decrease parameters to train
    content_image = tf.image.resize(content_image, [image_size, image_size],
                                    preserve_aspect_ratio=True)  # resize preserving the aspect ratio respecting either dimensions
    # Resize style image to fit the content image
    style_image = tf.image.resize(style_image, content_image.shape[:-1])
    # Add batch dimension
    content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))
    style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))
    return content_image, style_image


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


def load_vgg_CNN(input_shape):
    """
    Set-up the CNN used to compute activations
    :param input_shape:
    :return:
    """
    vgg_model = tf.keras.applications.VGG19(
        include_top=False,
        input_shape=input_shape,
        weights="imagenet"  # weights='pretrained_model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
    )
    vgg_model.trainable = False
    return vgg_model


def initialize_generated_image(content_image):
    """
    Initialize the content image with an image close to the content image
    :param content_image: tensor of size (n_H,n_W,n_C)
    :return:tensor of size (n_H,n_W,n_C)
    """
    generated_image = tf.image.convert_image_dtype(content_image, tf.float32)
    noise = tf.random.uniform(tf.shape(generated_image), -0.05, 0.05)
    generated_image = tf.add(generated_image, noise)
    generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
    return generated_image


def generate_selected_model(model, layers_list):
    outputs = []
    layers_name = [l.name for l in model.layers]
    for layer_name, _ in layers_list:
        if not (layer_name in layers_name):
            raise ValueError(f'Layer {layer_name} is not a valid layer of the selected model.')
        else:
            outputs.append(model.get_layer(layer_name).output)
    selected_model = tf.keras.Model([model.input], outputs)
    return selected_model


def compute_content_cost(content_output, generated_output):
    """
    Compute the content cost
    :param content_output: tensor of dimension (1,n_H,n_W,n_C), hidden layer activations representing the content of the content image
    :param generated_output: tensor of dimension (1,n_H,n_W,n_C), hidden layer activations representing content of the generated image
    :return:
    """
    # Get the last output of the the CNN used
    a_C = content_output[-1]
    a_G = generated_output[-1]

    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G
    a_C_unrolled = tf.reshape(a_C, shape=[1, -1, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[1, -1, n_C])

    # compute the cost with tensorflow
    J_content = tf.reduce_sum((a_C_unrolled - a_G_unrolled) ** 2) / (4 * n_H * n_W * n_C)

    return J_content


def gram_matrix(A):
    return tf.matmul(A, tf.transpose(A))


def compute_layer_style_cost(style_output_layer, generated_output_layer):
    """
    Compute the style cost for a given layer
    :param style_output: tensor of dimension (1,n_H,n_W,n_C), hidden layer activation representing the content of the style image.
    :param generated_output: tensor of dimension (1,n_H,n_W,n_C) hidden layer activation representing the content of the generated image.
    """
    _, n_H, n_W, n_C = style_output_layer.get_shape().as_list()
    a_S = tf.transpose(tf.reshape(style_output_layer, shape=[-1, n_C]))
    a_G = tf.transpose(tf.reshape(generated_output_layer, shape=[-1, n_C]))

    style_matrix_S = gram_matrix(a_S)
    style_matrix_G = gram_matrix(a_G)

    J_style_layer = tf.reduce_sum((style_matrix_S - style_matrix_G) ** 2) / ((2 * n_H * n_W * n_C) ** 2)
    return J_style_layer


def compute_style_cost(style_output, generated_output, style_layers_weights):
    """
    Compute the style cost over all layers with specified weights dictionnary
    :param style_output:
    :param generated_output:
    :param style_layers_weights:
    :return:
    """
    J_style = 0
    for i, lay_weights in enumerate(
            style_layers_weights[:-1]):  # N.B. compute the style loss on inners layers (exclude last layer)
        style_output_layer = style_output[i]
        generated_output_layer = generated_output[i]
        J_style_layer = compute_layer_style_cost(style_output_layer, generated_output_layer)
        J_style += lay_weights[1] * J_style_layer
    return J_style


def compute_cost(content_cost, style_cost, alpha=10, beta=40):
    """
    Compute total cost
    :param content_cost:
    :param style_cost:
    :param alpha:
    :param beta:
    :return:
    """
    return alpha * content_cost + beta * style_cost
