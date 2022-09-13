from keras.models import load_model
from backend.Preprocessing import ImgPreprocessing 
from backend.HeatMapping import HeatMap
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras.models import Model
#https://keras.io/examples/vision/grad_cam/
#tf.compat.v1.disable_eager_execution()
#tf.compat.v1.experimental.output_all_intermediates(True)



def load_prediction_model():
    model_cnn = load_model('./backend/inference/WilhemNet_86.h5')
    model_cnn.make_predict_function()
    return model_cnn
###------------------------------------------------------------------------------------------------------------
### Lectura de la ruta de la imágen
###------------------------------------------------------------------------------------------------------------


def GradCam(model, img_array, layer_name, eps=1e-8):
    '''
    Creates a grad-cam heatmap given a model and a layer name contained with that model
    

    Args:
      model: tf model
      img_array: (img_width x img_width) numpy array
      layer_name: str


    Returns 
      uint8 numpy array with shape (img_height, img_width)

    '''

    gradModel = Model(
			inputs=[model.inputs],
			outputs=[model.get_layer(layer_name).output,
				model.output])
    
    with tf.GradientTape() as tape:
			# cast the image tensor to a float-32 data type, pass the
			# image through the gradient model, and grab the loss
			# associated with the specific class index
      inputs = tf.cast(img_array, tf.float32)
      (convOutputs, predictions) = gradModel(inputs)
      loss = predictions[:, 0]
		# use automatic differentiation to compute the gradients
    grads = tape.gradient(loss, convOutputs)
    
    # compute the guided gradients
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
		# the convolution and guided gradients have a batch dimension
		# (which we don't need) so let's grab the volume itself and
		# discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]
    # compute the average of the gradient values, and using them
		# as weights, compute the ponderation of the filters with
		# respect to the weights
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
  
    # grab the spatial dimensions of the input image and resize
		# the output class activation map to match the input image
		# dimensions
    (w, h) = (512, 512)
    heatmap = cv2.resize(cam.numpy(), (w, h))
		# normalize the heatmap such that all values lie in the range
		# [0, 1], scale the resulting values to the range [0, 255],
		# and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    # heatmap = (heatmap * 255).astype("uint8")
		# return the resulting heatmap to the calling function
    return heatmap


def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x-b)))

def superimpose(img_bgr, cam, thresh, emphasize=False):
    
    '''
    Superimposes a grad-cam heatmap onto an image for model interpretation and visualization.
    

    Args:
      image: (img_width x img_height x 3) numpy array
      grad-cam heatmap: (img_width x img_width) numpy array
      threshold: float
      emphasize: boolean

    Returns 
      uint8 numpy array with shape (img_height, img_width, 3)

    '''
    heatmap = cv2.resize(cam, (512, 512))
    
    if emphasize:
        heatmap = sigmoid(heatmap, 50, thresh, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    hif = .8
    superimposed_img = heatmap * hif + img_bgr
    
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255  
    
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)    
  
    
    superimposed_img_rgb = tf.keras.preprocessing.image.img_to_array(superimposed_img_rgb)
    print('superimposed_img', superimposed_img_rgb.shape)
    #superimposed_img_rgb.save('static/heatmap.jpeg')
    cv2.imwrite('static/heatmap.jpeg', superimposed_img_rgb)
    return superimposed_img_rgb

# Predecir si la imagen corresponde a bacterial, virus, normal
def predict_label(img_path):
    #   1. call function to pre-process image: it returns image in batch format
    batch_array_img = ImgPreprocessing.preprocess(img_path)
    #   2. call function to load model and predict: it returns predicted class and probability
      
    model = load_prediction_model()
    
    # model_cnn = tf.keras.models.load_model('conv_MLP_84.h5')
    prediction = np.argmax(model.predict(batch_array_img))
    proba = np.max(model.predict(batch_array_img)) * 100
    label = ""
    if prediction == 0:
        label = "bacteriana"
    if prediction == 1:
        label = "normal"
    if prediction == 2:
        label = "viral"
    #   3. call function to generate Grad-CAM: it returns an image with a superimposed heatmap
    
    #heatmap = HeatMap.get_heatMap(img_path)
    #heatmap = get_heatMap(img_path)
    #preprocess_input = keras.applications.xception.preprocess_input
    #img_array = preprocess_input(HeatMap.get_img_array(img_path, size=(512,512)))
    #model.layers[-1].activation = None
    #heatmap = HeatMap.make_gradcam_heatmap(batch_array_img, model, "conv10_thisone")    
    #HeatMap.save_and_display_gradcam(img_path, heatmap)

    img = ImgPreprocessing.preprocess(img_path)
    
    layer_name = 'conv10_thisone'
    grad_cam= GradCam(model,img,layer_name)
    
    grad_cam_superimposed = superimpose(img, grad_cam, 0.5, emphasize=True)
    

    return label 
    
