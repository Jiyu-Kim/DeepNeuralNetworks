import numpy as np
from utils_plot import _initialize, optimizer
import matplotlib.pyplot as plt

np.random.seed(428)

# ========================= EDIT HERE ========================
# 1. Choose DATA : Titanic / Digit
# 2. Adjust Hyperparameters
# 3. Choose Optimizer : SGD

# DATA
DATA_NAME = 'Titanic'

# HYPERPARAMETERS
batch_size = 779 
tlqkf = 5000
learning_rate = 0.1

# ============================================================
epsilon = 0.01 # not for SGD
gamma = 0.9 # not for SGD

# OPTIMIZER
OPTIMIZER = 'SGD'

assert DATA_NAME in ['Titanic', 'Digit', 'Basic_coordinates']
assert OPTIMIZER in ['SGD']



# TRAIN
#loss, history = model.train(train_x, train_y, num_epochs, batch_size, learning_rate, optim)
#print('Training Loss at the last epoch: %.2f' % loss)
#print(history['epoch_loss'])


"""
# VISUALIZATION
plt.title("full-batch training model acc")
plt.xlabel('epoch')
plt.ylabel('acc')
plt.plot(np.arange(100, num_epochs+1, 100), history['epoch_acc'], 'bo')
plt.xticks(np.arange(100,num_epochs+1,100))
#plt.plot(np.arange(30, 301, 30), history['epoch_acc'])
plt.show()
"""
acc_list = []
#tlqkf = 3000
for epoch in range(100, tlqkf+1, 100):
    # Load dataset, model and evaluation metric
    train_data, test_data, logistic_regression, metric = _initialize(DATA_NAME)
    train_x, train_y = train_data

    num_data, num_features = train_x.shape
    #print('# of Training data : ', num_data)

    # Make model & optimizer
    model = logistic_regression(num_features)
    optim = optimizer(OPTIMIZER, gamma=gamma, epsilon=epsilon)

    loss, history = model.train(train_x, train_y, epoch, batch_size, learning_rate, optim)
    test_x, test_y = test_data
    pred = model.forward(test_x)
    #ACC = metric(pred.reshape(-1, 1), test_y) 
    ACC = metric(pred, test_y)
    acc_list.append(ACC)   
    print(OPTIMIZER, ' ACC on Test Data : %.3f' % ACC)

#print(max(tlqkf))
plt. title("model acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.plot(np.arange(100, tlqkf+1, 100), acc_list)
#plt.xticks(np.arange(100,tlqkf+1,100))
plt.show()
"""
# EVALUATION
test_x, test_y = test_data
pred = model.forward(test_x)
#ACC = metric(pred.reshape(-1, 1), test_y) 
ACC = metric(pred, test_y)

print(OPTIMIZER, ' ACC on Test Data : %.3f' % ACC)
"""