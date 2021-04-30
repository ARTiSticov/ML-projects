import numpy as np
from flask import Flask, request, render_template
import pickle



app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
	return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
   
    

   int_features = [[int(x) for x in request.form.values()]]
   print(int_features)
   data1 = np.array(int_features)
  
 
   
   print('DATA: ', data1)
   
   from model import classifier
   prediction = classifier.predict(data1)
   
   print('----------------')
   print('PREDiCTION: ', prediction)
   
   output = np.array2string(prediction)
   
   
   return render_template('index.html', prediction_text='Breast Cancer Prediction: {}'.format(output))
   
            
if __name__ == "__main__":
	app.run(debug=True)