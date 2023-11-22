from flask_wtf import FlaskForm
from wtforms import SelectField, IntegerField, SubmitField, FloatField
from wtforms.validators import NumberRange, InputRequired


class FormSelectModel(FlaskForm):
    models = ['knn', 'mlp', 'rf', 'svm']
    model = SelectField('Selecione o algoritmo desejado', choices=models,
                        default='knn')
    submit = SubmitField('Form Submit')

class FormKNN(FlaskForm):
    n_neighbors = IntegerField('Number of neighbors', [NumberRange(min=1), InputRequired()],
                               default=1, render_kw={'placeholder': 'N of neighbors (default=1)'})
    knnsubmit = SubmitField('Train estimator')

class FormMLP(FlaskForm):
    hidden_sizes = SelectField('Hidden layers configuration',
                               choices=['(64, 64)',
                                        '(128, 128)',
                                        '(256, 256)'],
                                default='(64, 64)')
    activation = SelectField('Activation of hidden layers',
                             choices=['identity', 'logistic',
                                      'tanh', 'relu'],
                             default='relu')
    lr = FloatField('Learning rate', [NumberRange(min=0.0), InputRequired()],
                    default=0.001, render_kw={'placeholder': 'lr (default=0.001)'})
    mlpsubmit = SubmitField('Train estimator')

class FormRF(FlaskForm):
    n_estimators = IntegerField('Number of estimators', [NumberRange(min=1), InputRequired()],
                                default=5, render_kw={'placeholder': 'N of estimators (default=5)'})
    max_depth = IntegerField('Max depth', [NumberRange(min=1)],
                             default=None, render_kw={'placeholder': 'Max depth (default=None)'})
    rfsubmit = SubmitField('Train estimator')

class FormSVM(FlaskForm):
    C = FloatField('Regularization parameter', [NumberRange(min=0.0), InputRequired()],
                                default=1.0, render_kw={'placeholder': 'C (default=1.0)'})
    kernel = SelectField('Kernel', choices=['linear', 'poly', 'rbf', 
                                            'sigmoid', 'precomputed'],
                             default='rbf')
    degree = IntegerField('Polynomial kernel degree', [NumberRange(min=0), InputRequired()],
                          default=3, render_kw={'placeholder': 'Degree (default=3)'})
    svmsubmit = SubmitField('Train estimator')

