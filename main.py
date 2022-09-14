# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 16:00:42 2022

@author: Tina
"""

import logging
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

#from flasgger import Swagger

with open('PoA_v5.0.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)
#swagger = Swagger(app)

@app.route('/')
def home():
    return '''
    how are you today?
    <hr/>
    <a href=/form>Prediction Form</a>
    '''

@app.route('/form')
def form():
    return '''
<style>
body{
    margin: 20px 20%;
    border: 1px solid silver;
    padding: 10px;
}
ul{
    border: 5px solid silver;
    padding: 10px;
    list-style: none;
}
li{
    padding: 5px 0px;
}
li:hover{
    background: silver;
}
label{
    float: left;
    width: 300px;
}
label:after{
    content: " : ";
}
iframe{
    width: 100%;
    height: 50px;
}
</style>
<body>
<h1>Probability of Approval</h1>
<hr/>
<form action=/predict target=predict>
    <ul>
    <li>
        <label>BACKEND_RATIO_PCT</label>
        <input type=number step=any name=BACKEND_RATIO_PCT />
    </li>
    <li>
        <label>CRED_SCORE_NUM</label>
        <input type=number step=any name=CRED_SCORE_NUM />
    </li>
    <li>
        <label>DTI</label>
        <input type=number step=any name=DTI />
    </li>
    <li>
        <label>FRONTEND_RATIO_PCT</label>
        <input name=FRONTEND_RATIO_PCT />
    </li>
    <li>
        <label>HOME_VALUE</label>
        <input name=HOME_VALUE />
    </li>
    <li>
        <label>LATE_CHRG_DELINQ_12_MTH_NUM</label>
        <input name=LATE_CHRG_DELINQ_12_MTH_NUM />
    </li>
    <li>
        <label>LATE_NON_MTG_PYMTS</label>
        <input name=LATE_NON_MTG_PYMTS />
    </li>
    <li>
        <label>LOAN_AMT</label>
        <input name=LOAN_AMT />
    </li>
    <li>
        <label>LTV</label>
        <input name=LTV />
    </li>
    <li>
        <label>MDO</label>
        <input name=MDO />
    </li>
    <li>
        <label>MTHLY_HOUSING_EXPN_AMT</label>
        <input name=MTHLY_HOUSING_EXPN_AMT />
    </li>
    <li>
        <label>PITI</label>
        <input name=PITI />
    </li>
    <li>
        <label>TOTAL_LOAN_COST</label>
        <input name=TOTAL_LOAN_COST />
    </li>
    <li>
        <label>BANKRUPTCY</label>
        <select name="BANKRUPTCY">
          <option value="">NA</option>
          <option value="Y">Yes</option>
          <option value="N">No</option>
        </select>
    </li>
    <li>
        <label>FORECLOSURE_FLAG</label>
        <select name="FORECLOSURE_FLAG">
          <option value="">NA</option>
          <option value="Y">Yes</option>
          <option value="N">No</option>
        </select>
    </li>
    </ul>
    <input type=submit />
</form>
<hr/>
<iframe name=predict></iframe>
<script>
var form = document.querySelector('form');
form.addEventListener('change', function(){
    var data = new FormData(form);
    var params = new URLSearchParams(data).toString();
    console.log(params);
});
</script>
</body>
    '''

@app.route('/predict')
def predict():
    BACKEND_RATIO_PCT = request.args.get("BACKEND_RATIO_PCT")
    CRED_SCORE_NUM = request.args.get("CRED_SCORE_NUM")
    DTI = request.args.get("DTI")
    FRONTEND_RATIO_PCT = request.args.get("FRONTEND_RATIO_PCT")
    HOME_VALUE = request.args.get("HOME_VALUE")
    LATE_CHRG_DELINQ_12_MTH_NUM = request.args.get("LATE_CHRG_DELINQ_12_MTH_NUM") 
    LATE_NON_MTG_PYMTS = request.args.get("LATE_NON_MTG_PYMTS")
    LOAN_AMT = request.args.get("LOAN_AMT")
    LTV = request.args.get("LTV")
    MDO = request.args.get("MDO")
    MTHLY_HOUSING_EXPN_AMT = request.args.get("MTHLY_HOUSING_EXPN_AMT")
    PITI = request.args.get("PITI")
    TOTAL_LOAN_COST = request.args.get("TOTAL_LOAN_COST")
    BANKRUPTCY = request.args.get("BANKRUPTCY")
    FORECLOSURE_FLAG = request.args.get("FORECLOSURE_FLAG")

    features_numerical = [BACKEND_RATIO_PCT, CRED_SCORE_NUM, DTI, FRONTEND_RATIO_PCT,
                          HOME_VALUE, LATE_CHRG_DELINQ_12_MTH_NUM, LATE_NON_MTG_PYMTS, LOAN_AMT,
                          LTV, MDO, MTHLY_HOUSING_EXPN_AMT, PITI,
                          TOTAL_LOAN_COST]
    features_categorical = [BANKRUPTCY, FORECLOSURE_FLAG]

    features_numerical = [float(feature) if feature else np.nan for feature in features_numerical]
    features_categorical = [feature or np.nan for feature in features_categorical]
    features = features_numerical + features_categorical
    features = np.array([features], dtype=object)

    label = model.predict_proba(features)[0][1]

    return jsonify({
        'PoA': label,
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
