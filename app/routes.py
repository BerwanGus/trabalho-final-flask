import re
import json
from flask import render_template, url_for, redirect, request
from app.forms import (FormSelectModel, FormKNN,
                       FormMLP, FormRF, FormSVM)
from app.operators import fit_knn, fit_mlp, fit_rf, fit_svm
from app.models import Round
from app import app, session

@app.route('/', methods=['GET', 'POST'])
def homepage():
    _formSelectModel = FormSelectModel()

    return render_template('home.html')

@app.route('/confusion_matrix', methods=['GET'])
def confusion_matrix():
    return render_template('cm.html')

@app.route('/train/<model>', methods=['GET', 'POST'])
def train(model):
    _formKNN = FormKNN()
    _formMLP = FormMLP()
    _formRF = FormRF()
    _formSVM = FormSVM()

    if model == 'knn':
        if _formKNN.is_submitted():
            params = {'n_neighbors': _formKNN.data['n_neighbors']}
            metrics = fit_knn(**params)

            round = Round('knn', json.dumps(params),
                        *metrics)
            
            session.add(round)
            session.commit()

            return redirect(url_for('dashboard', model=model))
        return render_template('knn.html', knn=_formKNN)

    if model == 'mlp':
        if _formMLP.is_submitted():
            str_hidden = _formMLP.data['hidden_sizes']
            hidden_sizes = re.sub('[(,)]', '', str_hidden).split()
            hidden_sizes = tuple(int(x) for x in hidden_sizes)
            print(hidden_sizes)
            params = {'hidden_sizes': hidden_sizes,
                    'activation': _formMLP.data['activation'],
                    'lr':  _formMLP.data['lr']}
            metrics = fit_mlp(**params)

            round = Round('mlp', json.dumps(params),
                        *metrics)

            session.add(round)
            session.commit()

            return redirect(url_for('dashboard', model=model))
        return render_template('mlp.html', mlp=_formMLP)
        
    if model == 'rf':
        if _formRF.is_submitted():
            print('AAAAAAAAAAAAAAA')
            params = {'n_estimators': _formRF.data['n_estimators'],
                    'max_depth': _formRF.data['max_depth']}
            metrics = fit_rf(**params)

            round = Round('rf', json.dumps(params),
                        *metrics)

            session.add(round)
            session.commit()
            
            return redirect(url_for('dashboard', model=model))
        return render_template('rf.html', rf=_formRF)
    
    if model == 'svm':
        if _formSVM.is_submitted():
            params = {'C': _formSVM.data['C'],
                    'kernel': _formSVM.data['kernel'],
                    'poly_degree': _formSVM.data['degree']}
            metrics = fit_svm(**params)

            round = Round('svm', json.dumps(params),
                        *metrics)        

            session.add(round)
            session.commit()
            
            return redirect(url_for('dashboard', model=model))
        return render_template('svm.html', svm=_formSVM)

@app.route('/dashboard/<model>', methods=['GET', 'POST'])
def dashboard(model):
    query = session.query(Round).filter_by(estimator=model).all()
    
    if len(query) > 0:
        _round = query[-1]
        labels = ['acc', 'macro_pre', 'macro_rec', 'macro_f1', 'micro_f1']
        metrics = [_round.acc, _round.macro_pre, _round.macro_rec,
                _round.macro_f1, _round.micro_f1]
        return render_template('dashboard.html', round=_round, labels=labels, metrics=metrics)

    return redirect(url_for('homepage'))