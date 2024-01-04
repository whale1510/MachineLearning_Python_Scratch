#project : Multi Layer Perceptron 구현 / skku 기계학습 수업
#name : 조병웅

#임포트할 라이브러리
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


#멀티레이어를 구현하는 클래스
class my_MLP():
    def __init__(self):
        self.author = __author__
        self.id = __id__

        self.lr = 0.01
        self.neurons = 32
        self.N_hidden = 1
        self.batch_size = 8
        self.epochs = 20
        self.hidden_layers = []
        self.weights = []
        self.N_class = None
        
    #각 층마다 활성화 함수로는 sigmoid로 구현.
    def sigmoid(self, x):
        result = 1/(1+np.exp(-x)) #시그모이드 함수를 코드로 구현.
        return result 

    #마지막 은닉층에서 출력층은 softmax로 구현
    def softmax(self, x):
        m = np.max(x)
        x = np.exp(x-m)
        sum_x = np.sum(x)
        result = x / sum_x #softmax 수식을 코드로 구현.
        return result 


    def prime_sigmoid(self, x):
        """

        해당 함수는 주어진 x에 대하여 sigmoid 함수를 미분한 식을 연산하는 함수입니다.
        """
        
        return x * (1 - x)


    def feed_forward(self, X_train):
        """
        주어진 X_train에 대하여 순전파 작업을 진행.
        현재 hidden layer 정보를 weight와 곱한 값에 대해서 sigmoid 연산을 취한 값이 다음 hidden layer 정보가 됩니다.

        """
        hidden_layer = X_train # 첫번째 hidden layer 정보는 X_train 값
        self.hidden_layers[0] = hidden_layer # X_train 값을 첫번째 hidden layer로 저장

        for i, weights in enumerate(self.weights): # weights의 shape
            temp = np.dot(self.hidden_layers[i], weights) # 내적하여 temp값에 임시저장합니다. 
            self.hidden_layers[i + 1] = self.sigmoid(temp) # 값을 시그모이드 함수를 거쳐 다음 히든 레이어로 전달합니다.

        output = self.softmax(self.hidden_layers[-1]) # 마지막 layer에 대해 softmax를 취함으로써 최종 값 도출
        return output 

        
    def back_propagation(self, output, y_onehot):
        """
        주어진 순전파 결과인 output과 정답 정보인 y_onehot에 대해 역전파 작업을 진행.

        """
        delta_t = (output - y_onehot) * self.prime_sigmoid(self.hidden_layers[-1]) #
        for i in range(1, len(self.weights)+1): # 1,2를 반복
            self.weights[-i] -= self.lr*(np.dot(self.hidden_layers[-i-1].T,delta_t)) # 이전 히든레이어와 델타T를 내적을 통해 연산한다. 그리고 lr을 곱한 후, 새로운 weights로 저장한다.
            delta_t = (np.dot(delta_t, self.weights[-i].T)) * self.prime_sigmoid(self.hidden_layers[-i-1])  #새로운 weights를 가지고 delta와 내적한후, 이전 레이어에 대한 연산을 구해 delta_T를 업그레이드 한다.

            #여기서 수식을 잘못썼는지 정확도가 향상되지 않음. 여러번 시도했지만 고쳐야할 부분을 파악하지 못함. 양해부탁드립니다.           


                # self.weights
                    # 1) 이전 hidden_layer와 delta_t를 연산함으로써 Cost function의 gradient를 계산
                    # 2) 연산된 gradient에 learning rate를 곱한 최종 값을 기존 weight에서 빼줌으로써 weight 업데이트

                # delta
                    # 3) 업데이트 된 weight와 delta_t를 연산
                    # 4) 이전 hidden_layer 값에 대한 prime_sigmoid 값을 3)의 최종 값과 연산함으로써 delta_t 업데이트
                
            

    def fit(self, X_train, y_train):
        """

        해당 함수는 주어진 훈련 데이터 X_train와 y_train을 통해 MLP 모델을 훈련시키는 함수입니다.
        """

        self.N_class = len(np.unique(y_train)) # 분류할 클래스의 개수 설정   / 10개(예시)
        y_onehot = np.eye(self.N_class)[y_train] # y_train 정보를 원핫인코딩 변환  / 0000100000
        # 훈련 레이어의 크기 정보를 담은 리스트 : [input layer size, hidden layer size(neuron size * layer 수), output layer size]
        total_layer_size = np.array([X_train.shape[1]] + [self.neurons]*self.N_hidden + [y_onehot.shape[1]]) #3개륽 가진 array / 784, 32, 10
        self.hidden_layers = [np.empty((self.batch_size, layer)) for layer in total_layer_size] # 훈련 레이어 정보를 담을 리스트 / [(8,784),(8,32),(8,10)]

        # 초기 랜덤 가중치 할당
        self.weights = list()
        for i in range(total_layer_size.shape[0]-1): # 전체 레이어 사이사이 가중치 생성 / 783 / 0,1
            self.weights.append(np.random.uniform(-1,1,size=[total_layer_size[i], total_layer_size[i+1]]))

        self.weights = np.array(self.weights)

        # epoch 만큼 훈련 반복
        for epoch in range(self.epochs):
            shuffle = np.random.permutation(X_train.shape[0]) # 랜덤한 훈련 데이터 index 생성
            X_batches = np.array_split(X_train[shuffle], X_train.shape[0]/self.batch_size) # X batch 생성
            y_batches = np.array_split(y_onehot[shuffle], X_train.shape[0]/self.batch_size) # y batch 생성
            
            for x_batch, y_batch in zip(X_batches, y_batches):
                output = self.feed_forward(x_batch) # 순전파 단계

                self.back_propagation(output, y_batch) # 역전파 단계 (교안의 cost function 계산과 함께 진행)


    def predict(self, X_test):
        """
        해당 함수는 훈련된 가중치를 기반으로 평가 데이터 X_test의 예측값 y_pred를 연산하는 함수입니다.
        """

        # 평가 레이어의 크기 정보를 담은 리스트
        test_layer_size = np.array([X_test.shape[1]] + [self.neurons]*self.N_hidden + [self.N_class])
        self.hidden_layers = [np.empty((X_test.shape[0], layer)) for layer in test_layer_size] # 평가 레이어 정보를 담을 리스트

        output = self.feed_forward(X_test) # 순전파
        y_pred = output.argmax(axis=1) # 최종 예측값

        return y_pred

    


















