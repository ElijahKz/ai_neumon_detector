import numpy as np
import cv2
import backend.Inference.ModelInference as inference
from tensorflow.keras import backend as K
import tensorflow as tf
from backend.Preprocessing import ImgPreprocessing 

"""
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)

"""

from skimage import transform
from PIL import ImageTk, Image


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path='static/cam.jpg', alpha=0.4):
    # Load the original image
    #img = keras.preprocessing.image.load_img(img_path)
    #img = keras.preprocessing.image.img_to_array(img)
    
    img = ImgPreprocessing.preprocess(img_path)
    """img_array_export = np.asarray(img)
    img2show = Image.fromarray(img_array_export)
    cv2.imwrite('static/img.jpg', img2show)"""


    
    #img = inference.preprocess(img)
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap =  tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    """superimposed_img = np.array(superimposed_img, dtype='uint8')
    superimposed_img = cv2.resize(superimposed_img, (512, 512))"""
    
    #superimposed_img = transform.resize(superimposed_img, (superimposed_img.shape[1], superimposed_img.shape[2]))
    #print(superimposed_img.shape[1])
    #print(superimposed_img.shape[2])
    superimposed_img = superimposed_img.reshape((superimposed_img.shape[1],superimposed_img.shape[2], 3))
    #arr_ = np.squeeze(superimposed_img)
    
    #superimposed_img = np.expand_dims(superimposed_img, axis=0)
    #superimposed_img = np.expand_dims(superimposed_img, axis=0)
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    #superimposed_img = np.squeeze(superimposed_img)
    
    # Save the superimposed image
    superimposed_img.save(cam_path)



def get_heatMap(img_path):
    
    
    img = ImgPreprocessing.preprocess(img_path)
    model = inference.load_prediction_model()

    
    preds = model.predict(img)
    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]
    last_conv_layer = model.get_layer("conv10_thisone")
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate(iter(img))

    for filters in range(64):
        conv_layer_output_value[:, :, filters] *= pooled_grads_value[filters]
    # creating the heatmap
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU
    heatmap /= np.max(heatmap)  # normalize
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img = np.uint8(255 * img)
    img2 = cv2.resize(img, (512, 512))
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2)
    superimposed_img = superimposed_img.astype(np.uint8)
    return superimposed_img[:, :, ::-1]