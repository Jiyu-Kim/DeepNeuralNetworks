import numpy as np

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch_size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses
        # Weights are updated through the optimizer, not directly within 'train' function.
        # ========================= EDIT HERE ========================
        y = y.reshape((x.shape[0], 1))
        print("x.shape {}, y.shape{}".format(x.shape,y.shape))
        print(int(x.shape[0]/batch_size)) #iteration
        
        for epoch in range(epochs):

            epoch_loss = []
            wd = np.zeros_like(self.W)

            for i in range(x.shape[0]//batch_size):
                start = i*batch_size
                end = start + batch_size
                x_batch = x[start:end]
                y_batch = y[start:end]
                y_predicted = self.forward(x_batch)
                err = (y_predicted-y_batch)
                
                #배치 손실 구하기
                batch_loss = np.mean(np.square(err))
                epoch_loss.append(batch_loss)

                #가중치 업데이트
                wd = (np.dot(x_batch.T, err)/batch_size)
                self.W = optim.update(self.W, wd, lr)

            #나머지 데이터로 학습하기    
            if int(x.shape[0]%batch_size) != 0:
                start = int(x.shape[0]//batch_size)*batch_size
                x_batch = x[start:]
                y_batch = y[start:]
                y_predicted = self.forward(x_batch)
                err = (y_predicted-y_batch)

                #배치 손실 구하기
                batch_loss = np.mean(np.square(err))
                epoch_loss.append(batch_loss) 

                #가중치 업데이트
                last_batch_size = x_batch.shape[0]
                wd = (np.dot(x_batch.T, err)/last_batch_size)
                self.W = optim.update(self.W, wd, lr)

            #에폭 손실 구하기        
            final_loss = np.mean(np.array(epoch_loss))

            if epoch%1000 == 0:
                print("cost {}, batch_size {}, epoch {}".format(final_loss, batch_size, epoch))
        # ============================================================
        return final_loss

    def forward(self, x):
        y_predicted = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'

        # ========================= EDIT HERE ========================
        y_predicted = np.dot(x, self.W)
        # ============================================================
        return y_predicted

