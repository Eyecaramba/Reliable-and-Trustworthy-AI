import random
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
from keras_preprocessing import image


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32,32))
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)  # final input shape = (1,224,224,3)
    return input_img_data


# def deprocess_image(x):
#     x = x.reshape((32,32, 3))
#     # Remove zero-center by mean pixel
#     x[:, :, 0] += 103.939
#     x[:, :, 1] += 116.779
#     x[:, :, 2] += 123.68
#     # 'BGR'->'RGB'
#     x = x[:, :, ::-1]
#     x = np.clip(x, 0, 255).astype('uint8')
#     return x
def deprocess_image(x):
    x = x.copy()
    x *= 255.0
    x = np.clip(x, 0, 255).astype('uint8')
    return x


CIFAR10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def decode_label(pred):
    """
    CIFAR-10 모델의 예측 결과(pred)를 받아 가장 높은 확률의 클래스 이름을 반환합니다.
    """
    # 1. 10개 점수 중 가장 높은 값의 인덱스(0~9)를 찾습니다.
    pred_index = np.argmax(pred[0])
    # 2. 해당 인덱스에 맞는 클래스 이름을 리스트에서 찾아 반환합니다.
    return CIFAR10_LABELS[pred_index]


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = 1e4 * np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(10, 10)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


def init_coverage_tables(model1, model2):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)

    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)

    return model_layer_dict1, model_layer_dict2


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index

    import random
import tensorflow as tf

def neuron_to_cover_V2(model_layer_dict, model=None, verbose=True):
    """
    Selects a neuron (layer, index) that has not been covered yet for neuron coverage.
    If all neurons are covered, randomly selects a neuron.
    Provides detailed output about the selection process and validates layer compatibility.
    
    Args:
        model_layer_dict: Dictionary mapping (layer_name, index) to coverage status (True/False).
        model: Keras model to validate layer compatibility (optional).
        verbose: Whether to print detailed selection information (default: True).
    
    Returns:
        tuple: (layer_name, index) of the selected neuron.
    """
    # Filter for valid layers (e.g., Conv2D, Dense) if model is provided
    valid_layers = []
    if model is not None:
        for layer_name, _ in model_layer_dict.keys():
            try:
                layer = model.get_layer(layer_name)
                # Only include layers with numerical outputs (e.g., Conv2D, Dense)
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense, tf.keras.layers.BatchNormalization)):
                    valid_layers.append(layer_name)
            except ValueError:
                continue  # Skip invalid layer names
    else:
        valid_layers = [layer_name for layer_name, _ in model_layer_dict.keys()]

    # Find not covered neurons among valid layers
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() 
                   if not v and layer_name in valid_layers]
    
    if not_covered:
        layer_name, index = random.choice(not_covered)
        status = "not covered"
    else:
        # If all neurons are covered, select a random neuron from valid layers
        valid_keys = [(layer_name, index) for layer_name, index in model_layer_dict.keys() 
                      if layer_name in valid_layers]
        if not valid_keys:
            raise ValueError("No valid layers found for neuron coverage.")
        layer_name, index = random.choice(valid_keys)
        status = "covered (all neurons covered, random selection)"
    
    # Print detailed information if verbose
    if verbose:
        print(f"Selected neuron: layer_name='{layer_name}', index={index}, status={status}")
        if model is not None:
            try:
                layer = model.get_layer(layer_name)
                output_shape = layer.output_shape
                print(f"Layer details: type={type(layer).__name__}, output_shape={output_shape}")
            except ValueError:
                print(f"Warning: Could not retrieve layer '{layer_name}' from model.")
    
    return layer_name, index


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True


def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False
