import tensorflow as tf
import numpy as np

def func(image_path):
    class_name=[
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
    ]

    model=tf.keras.models.load_model("Trained_Model.keras")

    img=tf.keras.preprocessing.image.load_img(image_path,target_size=(128,128))
    img_arr=tf.keras.preprocessing.image.img_to_array(img)
    img_arr=np.array([img_arr])
    predicted_class=class_name[np.argmax(model.predict(img_arr))]
    return predicted_class


print(func("test/le_mo.jpg"))
