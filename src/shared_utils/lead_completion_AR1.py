
from IPython import get_ipython

import pandas as pd
import mogptk as mo
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ecg = pd.read_csv('Z:/P_ECG/MIT_BIH_database/Corpus/mitbih/100.csv', header=0)

y1 = mo.LoadCSV('Z:/P_ECG/MIT_BIH_database/Corpus/mitbih/100.csv', x_col="'sample #'", y_col="'MLII'")
y2 = mo.LoadCSV('Z:/P_ECG/MIT_BIH_database/Corpus/mitbih/100.csv', x_col="'sample #'", y_col="'V5'")

dataset = mo.DataSet(y1, y2)
dataset.plot('full dataset', figsize=(10,8))

start_ = 200
end_ = 1000
test = 100

for channel in dataset:
    channel.filter(start_, end_)

for i, channel in enumerate(dataset):
    if i == 0:
        channel.remove_range(end_-test, end_)
        #channel.remove_randomly(pct=0.6)

dataset.plot('removed', figsize=(10,8))

class TransformWhiten(mo.TransformBase):
    """
    Transform the data so it has mean 0 and variance 1
    """
    def __init__(self):
        pass

    def set_data(self, data):
        # take only the non-removed observations
        self.mean = data.Y[data.mask].mean()
        self.std = data.Y[data.mask].std()

    def forward(self, y, x=None):
        return (y - self.mean) / self.std

    def backward(self, y, x=None):
        return (y * self.std) + self.mean

for channel in dataset:
    channel.transform(TransformWhiten())
    channel.transform(mo.TransformLog())

    dataset.plot('Normalized Dataset');


# create model
model = mo.MOSM(dataset, Q=3)

# initial estimation of parameters before training
model.init_parameters()

# train
model.train(maxiter=800, verbose=True)

start_ = 900
end_ = 1000
# set prediction range
for channel in dataset:
    channel.set_prediction_range(start=start_, end=end_, step=1)
print(start_, end_)
# plot predictions
model.predict()
fig, _ = model.plot_prediction(title='Trained model');


fig.savefig('signal_completion_V0.pdf')
