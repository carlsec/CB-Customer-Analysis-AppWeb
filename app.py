#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from flask import Flask, request, render_template, make_response
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('clf_gb-model.pkl')
d = joblib.load('d-model.pkl')

@app.route('/')
def home():
    return render_template('template.html')

@app.route('/predict',methods=['POST'])
def predict():
	customers = request.form['customers']
	protests = np.array(request.form['protests'])
	amountdue = np.array(request.form['amountdue'])
	amountrent = np.array(request.form['amountrent'])
	city = np.array(request.form['city'])
	commerce = np.array(request.form['commerce'])
	consumes = np.array(request.form['consumes'])
	equipment = np.array(request.form['equipment'])
	amountrent = amountrent.astype(float)
	protests = protests.astype(int)
	amountdue = amountdue.astype(float)
	consumes = consumes.astype(int)
	di = {'protests': protests, 'amountdue': amountdue, 'amountrent': amountrent, 'city': city, 'commerce': commerce, 'consumes': consumes, 'equipment': equipment}
	df = pd.DataFrame(di,  index=[0])
	df_cat = df[['city','commerce','equipment']]
	df_cat = df_cat.apply(lambda x: d[x.name].fit_transform(x))
	df_num = df[['amountrent', 'protests', 'amountdue', 'consumes']]
	df = pd.concat([df_num,df_cat], axis = 1)
	result = model.predict(df)

	if result == 1:
		prediction = f"O cliente {customers} se enquadra nos clientes bons pagadores!"
	else:
	    prediction = f"O cliente {customers} se enquadra nos clientes maus pagadores!"	

	return render_template('template.html', prediction=f'{prediction}', Cliente = customers)

if __name__ == "__main__":
    app.run(debug=True)