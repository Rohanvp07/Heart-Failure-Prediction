from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    	gender = request.form['Gender']
    	highbp = request.form['highbp']
    	age = request.form['age']
    	creatinine_phosphokinase = request.form['Creatinine']
    	ejection_fraction = request.form['Ejection']
    	platelets = request.form['Platelets']
    	serum_creatinine = request.form['Serum_Creatinine']
    	serum_sodium = request.form['Serum_Sodium']
    	time = request.form['Time']
    	
    age = scaler.transform(np.array(float(age)).reshape(-1,1))
    creatinine_phosphokinase = scaler.transform(np.array(float(creatinine_phosphokinase)).reshape(-1,1))
    ejection_fraction = scaler.transform(np.array(float(ejection_fraction)).reshape(-1,1))
    platelets = scaler.transform(np.array(float(platelets)).reshape(-1,1))
    serum_creatinine = scaler.transform(np.array(float(serum_creatinine)).reshape(-1,1))
    serum_sodium = scaler.transform(np.array(float(serum_sodium)).reshape(-1,1))
    time = scaler.transform(np.array(float(time)).reshape(-1,1))

    if gender=="Male":
        sex=1
    elif gender=="Female":
        sex=0

    if highbp=="Yes":
        highbp=1
    elif highbp=="No":
        highbp=0

            
    test=np.array([[age,creatinine_phosphokinase,ejection_fraction,highbp,platelets,
                        serum_creatinine,serum_sodium,sex,time]])

    my_prediction = model.predict(test)[0]
    return render_template('index.html', msg=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)


