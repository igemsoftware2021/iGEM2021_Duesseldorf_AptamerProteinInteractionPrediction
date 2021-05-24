import os
import time
import datetime

import torch
import torch.nn as nn

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) Prepare data
# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights


# 0) Prepare data
bc = datasets.load_breast_cancer()  # sample dataset for detecting breast cancer
X, y = bc.data, bc.target

n_samples, n_features = X.shape

print(n_samples, n_features)

# split our sample data randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale our features
sc = StandardScaler()  # always recommended to do when dealing with logistic regression
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# convert it to torch tensors now
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# important variables
learning_rate = 0.01
number_of_epochs = 12000  # should be divisible by 20

# create logfile
logfile = f'Model started training: {datetime.datetime.now()} ' \
          f'\n Learning rate: {learning_rate} ' \
          f'\n Number of epochs: {number_of_epochs}'


# 1) Design our model
class AptaModel(nn.Module):
    def __init__(self, n_input_features):
        super(AptaModel, self).__init__()
        self.lin1 = nn.Linear(n_input_features, n_input_features)
        self.lin2 = nn.Linear(n_input_features, n_input_features)
        self.lin3 = nn.Linear(n_input_features, 42)
        self.lin4 = nn.Linear(42, 12)
        self.lin5 = nn.Linear(12, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        x = self.lin3(x)
        x = torch.sigmoid(x)
        x = self.lin4(x)
        x = torch.relu(x)
        x = self.lin5(x)
        x = torch.sigmoid(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num


# load previously trained model (if wanted)
if os.path.isfile('saved_model.pt'):
    model = torch.load('saved_model.pt')
else:
    model = AptaModel(n_features)

# 2) Construct loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Stochastic Gradient Descent

# 3) Training loop
# start learning

loss = ""
for epoch in range(number_of_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()  # this does the backpropagation and gradient calculation

    # update weight
    optimizer.step()  # updates weights, but we need to empty our gradients...
    optimizer.zero_grad()  # like this.. never forget this

    # log
    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')
    if (epoch + 1) % (number_of_epochs / 20) == 0:
        logfile += f'\n    epoch: {epoch + 1}, loss = {loss.item():.8f}'

logfile += f'\n\nModel finished training: {datetime.datetime.now()}'
logfile += f'\n(training accuracy: {1 - loss.item()})'

# test our model
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
    logfile += f'\ntested accuracy = {acc:.12f}'

timestamp = round(time.time() * 1000)

# save our model with timestamp
torch.save(model, f'./models/saved_model_{timestamp}.pt')

# save log
text_file = open(f'./logfiles/log_{timestamp}.log', "w")
text_file.write(logfile)
text_file.close()
