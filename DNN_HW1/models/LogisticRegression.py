import numpy as np

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.random.rand(self.num_features, 1)

    def train(self, x, y, epochs, batch_size, lr, optim):
        loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses
        # Weights are updated through the optimizer, not directly within 'train' function.

        # Tip : log computation may cause some error, so try to solve it by adding an epsilon(small value) within log term.
        epsilon = 1e-7
        # ========================= EDIT HERE ========================
        for epoch in range(epochs):

            epoch_loss = []
            wd = np.zeros_like(self.W)

            for j in range(x.shape[0]//batch_size):
                idx = j*batch_size
                x_batch = x[idx:idx+batch_size] 
                y_batch = y[idx:idx+batch_size] #y_batch.shape = (10,)
                y_predicted = self.forward(x_batch) #y_predicted.shape = (10,)
                err = (y_predicted - y_batch) #err.shape = (10,)
                    
                #배치 손실 구하기
                loss_arr = np.where(y_batch==1, -np.log(y_predicted+epsilon), -np.log(1-y_predicted+epsilon)) #loss_arr.shape = (10,)
                batch_loss = np.mean(loss_arr)
                epoch_loss.append(batch_loss)

                #가중치 업데이트 
                wd = (np.dot(x_batch.T, err) / batch_size).reshape(-1, 1) 
                self.W = optim.update(self.W, wd, lr)

            #나머지 데이터로 학습하기
            if int(x.shape[0]%batch_size) != 0:
                idx = int(x.shape[0]//batch_size)*batch_size
                x_batch = x[idx:]
                y_batch = y[idx:]
                y_predicted = self.forward(x_batch)
                err = (y_predicted-y_batch)

                #배치 손실 구하기
                loss_arr = np.where(y_batch==1, -np.log(y_predicted+epsilon), -np.log(1-y_predicted+epsilon))
                batch_loss = np.mean(loss_arr)
                epoch_loss.append(batch_loss)

                #가중치 업데이트
                wd = (np.dot(x_batch.T, err) / batch_size).reshape(-1, 1)  
                self.W = optim.update(self.W, wd, lr)

            #에폭당 손실 평균 구하기
            loss = np.mean(np.array(epoch_loss))

            if epoch%100==0:
                print("cost {}, batch_size {}, epoch {}".format(loss,batch_size,epoch))
        # ============================================================
        return loss

    def forward(self, x):
        threshold = 0.5
        y_predicted = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # The model predicts the label as 1 if the probability is greater or equal to 'threshold'
        # Otherwise, it predicts as 0

        # ========================= EDIT HERE ========================
        dot_ = np.dot(x,self.W) 
        sigmoid = self._sigmoid(dot_)
        y_predicted = np.where(sigmoid>=threshold, 1, 0).squeeze() #y_predicted.shape = (10,)
        # ============================================================

        return y_predicted

    def _sigmoid(self, x):
        sigmoid = None

        # Sigmoid Function
        # The function returns the sigmoid of 'x'

        # ========================= EDIT HERE ========================
        sigmoid = 1/(1+np.exp(-x))
        # ============================================================
        return sigmoid
