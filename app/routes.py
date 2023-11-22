import re
from flask import render_template, url_for
from app.forms import (FormSelectModel, FormKNN,
                       FormMLP, FormRF, FormSVM)
from app.operators import fit_knn, fit_mlp, fit_rf, fit_svm

from app import app

@app.route('/', methods=['GET', 'POST'])
def homepage():
    _formSelectModel = FormSelectModel()
    _formKNN = FormKNN()
    _formMLP = FormMLP()
    _formRF = FormRF()
    _formSVM = FormSVM()

    if _formKNN.validate_on_submit():
        print('fitting knn')
        fit_knn(_formKNN.data['n_neighbors'])
    elif _formMLP.validate_on_submit():
        str_hidden = _formMLP.data['hidden_sizes']
        hidden_sizes = re.sub('[(,)]', '', str_hidden).split()
        fit_mlp(hidden_sizes,
                _formMLP.data['activation'],
                _formMLP.data['lr'])
    elif _formRF.validate_on_submit():
        fit_rf(_formRF.data['n_estimators'],
               _formRF.data['max_depth'])
    elif _formSVM.validate_on_submit():
        fit_svm(_formSVM.data['C'],
                _formSVM.data['kernel'],
                _formRF.data['degree'])
    

    return render_template('home.html', form=_formSelectModel,
                           knn=_formKNN, mlp=_formMLP,
                           rf=_formRF, svm=_formSVM)
