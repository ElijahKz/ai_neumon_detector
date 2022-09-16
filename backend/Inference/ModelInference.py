from keras.models import load_model
from backend.Preprocessing import ImgPreprocessing 
from backend.HeatMapping import HeatMap
import numpy as np
import cv2



#https://keras.io/examples/vision/grad_cam/




def load_prediction_model():
    model_cnn = load_model('./backend/inference/WilhemNet_86.h5')
    model_cnn.make_predict_function()
    return model_cnn
###------------------------------------------------------------------------------------------------------------
### Lectura de la ruta de la imágen
###------------------------------------------------------------------------------------------------------------

# Predecir si la imagen corresponde a bacterial, virus, normal
def predict_label(img_path):
    #   1. call function to pre-process image: it returns image in batch format
    batch_array_img = ImgPreprocessing.preprocess(img_path)
    #   2. call function to load model and predict: it returns predicted class and probability      
    model = load_prediction_model()    
    # model_cnn = tf.keras.models.load_model('conv_MLP_84.h5')
    prediction = np.argmax(model.predict(batch_array_img))
    proba = round(np.max(model.predict(batch_array_img)) * 100, 2)
    label = ""
    if prediction == 0:
        label = "bacteriana"
    if prediction == 1:
        label = "normal"
    if prediction == 2:
        label = "viral"
    #   3. call function to generate Grad-CAM: it returns an image with a superimposed heatmap
  
    X = HeatMap.getting_arrimg_for_gradcam(img_path, 512)
    layer_name = 'conv10_thisone'
    img = X[0]
    grad_cam= HeatMap.GradCam(model,np.expand_dims(img, axis=0),layer_name)    
    grad_cam_superimposed = HeatMap.superimpose(img, grad_cam, 0.5, emphasize=True)    

    return label , grad_cam_superimposed, proba
    
