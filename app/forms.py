from wtforms import Form, SelectField

class SelectModel(Form):
    models = ['knn', 'mlp', 'rf', 'svm']
    model = SelectField('Selecione o algoritmo desejado', choices=models)