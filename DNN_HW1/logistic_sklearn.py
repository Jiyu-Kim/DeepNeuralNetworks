from utils import _initialize, optimizer
import sklearn
from sklearn.linear_model import LogisticRegression

# 1. Choose DATA : Titanic / Digit
# 2. Adjust Hyperparameters
# ========================= EDIT HERE ========================
# DATA
DATA_NAME = 'Titanic'

# HYPERPARAMETERS
num_epochs = 300
# ============================================================

assert DATA_NAME in ['Titanic', 'Digit','Basic_coordinates']

# Load dataset, model and evaluation metric
train_data, test_data, _, metric = _initialize(DATA_NAME)
train_x, train_y = train_data

num_data, num_features = train_x.shape
print('# of Training data : ', num_data)
print("train_x.shape", train_x.shape)
ACC = 0.0
# ========================= EDIT HERE ========================
# Make model & optimizer
model = LogisticRegression(max_iter=num_epochs, random_state=0)

# TRAIN
model.fit(X=train_x, y=train_y)

# EVALUATION
test_x, test_y = test_data
predicted_y = model.predict(test_x)
predicted_y = predicted_y.reshape(-1,1)
ACC = metric(predicted_y, test_y)
# ============================================================

print('ACC on Test Data : %.2f ' % ACC)
