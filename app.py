from flask import Flask,request,render_template,jsonify
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from io import BytesIO
import base64

CR_Model=pickle.load(open('CropRecommendation.pkl','rb'))
CY_Model=pickle.load(open('CropYield.pkl','rb'))
MM_Model=pickle.load(open('MinMaxScaler.pkl','rb'))
SS_Model=pickle.load(open('StandardScaler.pkl','rb'))
preprocessor=pickle.load(open('preprocessor.pkl','rb'))

model=load_model('Crop_Disease_Prediction_Model.h5')

crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

crop_condition = ['Apple___Black_rot','Apple___healthy', 
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 
            'Grape___Black_rot','Grape___healthy', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
            'Tomato___healthy']

app=Flask(__name__)

app.static_folder='static'

@app.route("/")
def homePage():
    return render_template("home.html") 

@app.route("/model/Crop Recommendation",methods=['GET'])
def CR():
    return render_template("Crop_Recommend.html",result="")

@app.route("/model/Crop Recommendation/Result",methods=['POST'])
def CRR():
    data=request.json
    if data:
        try:
            N = int(data['N'])
            P = int(data['P'])
            K = int(data['K'])
            temp = float(data['temp'])
            humidity = float(data['humidity'])
            ph = float(data['ph'])
            rain = float(data['rain'])

            feature_list = [N, P, K, temp, humidity, ph, rain]
            single_pred = np.array(feature_list).reshape(1, -1)

            scaled_features = MM_Model.transform(single_pred)
            final_features = SS_Model.transform(scaled_features)
            prediction = CR_Model.predict(final_features)

            if prediction[0] in crop_dict:
                crop = crop_dict[prediction[0]]
                result = crop
            else:
                result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
            return jsonify(result=result)
        except:
            return jsonify(result="There is some error. Please try again")            

    return jsonify(result="There is some error. Please try again")



@app.route("/model/Crop Yield Prediction",methods=['GET'])
def CY():
    return render_template("Crop_Yield.html")

@app.route("/model/Crop Yield Prediction/Result",methods=['POST'])
def CYR():
    data=request.json
    print(data)
    if data:
        try:
            area=data['area']
            item=data['item']
            year=int(data['year'])
            temp=float(data['temp'])
            rain=float(data['rain'])
            pesticide=float(data['pesticide'])

            features = np.array([[year,rain,pesticide,temp,area,item]],dtype=object)
            transformed_features = preprocessor.transform(features)
            prediction = CY_Model.predict(transformed_features).reshape(1,-1)
            print(prediction)
            return jsonify(result=prediction[0][0])
        except:
            return jsonify(result="There is some error please try again")

    
    return jsonify(result="There is some error please try again")



@app.route("/model/Crop Disease Detection",methods=['GET'])
def CD():
    return render_template("Crop_Disease.html")

@app.route("/model/Crop Disease Detection/Result",methods=['POST'])
def CDR():

    if 'plant' in request.files:
        try:
            img_data=request.files['plant']

            img_data=BytesIO(img_data.read())
            image=tf.keras.preprocessing.image.load_img(img_data,target_size=(128,128))

            img_arr=tf.keras.preprocessing.image.img_to_array(image)
            img_arr=np.array([img_arr])

            pred=model.predict(img_arr)
            result=np.argmax(pred)
            
            img_data.seek(0)
            base64_img=base64.b64encode(img_data.read()).decode('utf-8')

            return jsonify(result=crop_condition[int(result)],image=base64_img)
        except:
            return jsonify(result="There is some error, Please try again")
    else:
        return jsonify(result="Image not uploaded successfully")


if __name__=='__main__':
    app.run()