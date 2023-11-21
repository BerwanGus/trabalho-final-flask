from flask import render_template, url_for
from app.forms import SelectModel

from app import app

@app.route('/', methods=['GET', 'POST'])
def homepage():
    _formSelectModel = SelectModel()
    return render_template('home.html', form=_formSelectModel, model=_formSelectModel.data)

@app.route('/model/knn', methods=['GET', 'POST'])
def knn():
    return render_template('knn.html')

@app.route('/model/mlp', methods=['GET', 'POST'])
def mlp():
    return render_template('mlp.html')

@app.route('/model/rf', methods=['GET', 'POST'])
def rf():
    return render_template('rf.html')

@app.route('/model/svm', methods=['GET', 'POST'])
def svm():
    return render_template('svm.html')