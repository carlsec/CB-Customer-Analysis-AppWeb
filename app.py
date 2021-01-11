#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from flask import Flask, request, render_template, make_response
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model_gb = joblib.load('clf_gb-model.pkl')
model_lgbm = joblib.load('clf_lgbm-model.pkl')
d = joblib.load('d-model.pkl')

@app.route('/')
def home():
    return render_template('template.html')

@app.route('/predict',methods=['POST'])
def predict():
	customers = request.form['customers']
	protests = np.array(request.form['protests'])
	amountdue = np.array(request.form['amount due'])
	amountrent = np.array(request.form['rent amount'])
	city = np.array(request.form['city'])
	commerce = np.array(request.form['commerce'])
	consumes = np.array(request.form['consumes'])
	equipment = np.array(request.form['equipment'])
	amountrent = amountrent.astype(float)
	amountrent = (amountrent * 1000)/50
	protests = protests.astype(int)
	protests = (protests * 1000)/50
	amountdue = amountdue.astype(float)
	amountdue = (amountdue * 1000)/50
	consumes = consumes.astype(int)
	di = {'protests': protests, 'amount due': amountdue, 'rent amount': amountrent, 'city': city, 'commerce': commerce, 'consumes': consumes, 'equipment': equipment}
	df = pd.DataFrame(di,  index=[0])
	df_cat = df[['city', 'commerce', 'equipment']]
	df_cat = df_cat.apply(lambda x: d[x.name].fit_transform(x))
	df_num = df[['protests', 'amount due', 'rent amount','consumes']]
	df = pd.concat([df_num,df_cat], axis = 1)
	result_gb = model_gb.predict_proba(df)[:, 1]
	result_lgbm = model_lgbm.predict_proba(df)[:, 1]
	result = (result_gb*0.5) + (result_lgbm*0.5)
	result = (result[0]).astype(float) * 100
	result = round(result, 2)

	if result > 50:
		prediction = f"O cliente {customers} se enquadra nos clientes bons pagadores! com {result}% de chance de ser um bom pagador."
	else:
	    prediction = f"O cliente {customers} se enquadra nos clientes maus pagadores! com {result}% de chance de ser um bom pagador."	

	return render_template('template-pred.html', prediction=f'{prediction}')

if __name__ == "__main__":
    app.run(debug=True)