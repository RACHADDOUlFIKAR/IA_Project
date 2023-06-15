
from flask import Flask ,render_template,request
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

model = load_model('keras_model.h5')
app= Flask(__name__)
@app.route('/',methods=['GET'])
def hellow():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    imagefile=request.files['image']
    imagedir="./static/images/"+imagefile.filename

    imagefile.save(imagedir)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    image = Image.open(imagedir)
    #image.show()
    

    
    
    size = (224, 224)
    image = ImageOps.fit(image, size , Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
   
    
    data[0] = normalized_image_array
    prediction = model.predict(data)
    val=max(prediction)
    
    return render_template('index.html',prediction=[val],imge=imagedir)
    
if __name__ == '__main__':
    app.run(port=2000,debug=True)    
