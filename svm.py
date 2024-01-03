#project : 선형 SVM 스크래치 (이진분류 경우) / skku 기계학습 수업
#name : 조병웅

#임포트할 라이브러리
import numpy as np
import matplotlib.pyplot as plt

#  fit 함수 
#  predict 함수 
#  get_accuracy 함수 
#  visualization 함수

#svm 분류기 함수
class SVMClassifier:
    def __init__(self,n_iters=100, lr = 0.0001, random_seed=3, lambda_param=0.01):
        self.author = __author__
        self.id = __id__
        self.n_iters = n_iters # 몇 회 반복하여 적절한 값을 찾을지 정하는 파라미터
        self.lr = lr  # 학습률과 관련된 파라미터 
        self.lambda_param = lambda_param
        self.random_seed = random_seed
        np.random.seed(self.random_seed)


    def fit(self, x, y):
        """
        본 함수는 x, y를 활용하여 훈련하는 과정을 코딩하는 부분입니다.
        아래 reference 사이트의 gradient 계산 부분을 참고하세요.
        reference: https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
        """
        n_samples, n_features = x.shape

        #y값을 SVM 계산에 활용해주기 위하여 0에 해당하는 y값들을 -1로 변환
        for i in range(len(y)): #for문과 if문을 사용하여, 0인 y값들을 -1로 갖는 y_어레이를 만든다.
            if y[i] == 0:
                y[i] = -1
            elif y[i] == 1:
                y[i] = 1
        y_ = y #변환
        
        # hint: w값 초기화, (n_features, )의 크기를 가지는 0과 1사이의 랜덤한 변수 어레이 (필수: 넘파이로 정의해야 함)
        init_w = np.random.rand(n_features, ) #rand 메서드를 사용해 랜덤한 변수 어레이를 생성한다.
        self.w = init_w
        self.b = 0 # b값 초기화

        for _ in range(self.n_iters):
            for i in range(n_samples):
                x_i = x[i]
                y_i = y_[i]

                # hint: y(i) * (w · x(i) + b) >= 1 를 만족하는 경우의 의미가 담기도록 if문을 채우세요.
                if y_i*(np.dot(self.w, x_i) + self.b) >= 1: #조건을 만족하면 True 값을 설정한는 if문 작성
                    condition = True  #
                else :
                    condition = False # 조건 만족 못할시
                if condition:
                    # hint: w에 대하여 Gradient Loss Function 수식을 이용하여 W를 업데이트 하세요.
                    self.w -= self.lr * (2*self.w*self.lambda_param) #loss함수를 이용한 수식 작성
                else:
                    # hint: w에 대하여 Gradient Loss Function 수식을 이용하여 W를 업데이트 하세요.
                    self.w = self.w - self.lr*(2*self.w*self.lambda_param) + self.lr*(x_i*y_i)
                    self.b -= self.lr * y_i

        return self.w, self.b # 결괏값으로 가중치와 편향을 제공한다


    def predict(self, x): 
        """
            [n_samples x features]로 구성된 x가 주어졌을 때, fit을 통해 계산된 
            self.w와 self.b를 활용하여 예측값을 계산합니다.

            @args:
                [n_samples x features]의 shape으로 구성된 x
            @returns:
                [n_samples, ]의 shape으로 구성된 예측값 array

            아래의 수식과 수도코드를 참고하여 함수를 완성.
                approximation = W·X - b
                if approximation >= 0 {
                    output = 1
                }
                else{
                    output = 0
                }
        """
        result = np.zeros(len(x)) #해당 어레이 변수는 out값을 담는 변수다.
        for i in range(len(x)):
            output = 0
            approximation = np.dot(x[i],self.w) - self.b #근사치
            if approximation >= 0: #근사치에 따라 Output 분류
                output = 1
            else:
                output = 0
            result[i] = output
        return result


    def get_accuracy(self, y_true, y_pred):
        """
            y_true, y_pred가 들어왔을 때, 정확도를 계산하는 함수.
            sklearn의 accuracy_score 사용 불가능 / sklearn의 accuracy_score 함수를 구현한다고 생각하면 됩니다.
            넘파이만을 활용하여 정확도 계산 함수를 작성하세요.
        """
        temp = 0 # 실제값과 예측값이 동일한 횟수를 나타내는 임시값을 저장할 변수 생성
        result = 0 # 결과값을 저장할 변수 생성
        for i in range(len(y_true)): # 값들의 크기 만큼 순환한다.
            if y_true[i] == y_pred[i]: # 만약, 예측과 실제값이 일치한다면, temp를 1 증가시킨다.
                temp += 1
        
        result = temp / len(y_true) #전체의 경우와 비교한다.
        return result


    def visualization(self, X_test, y_test, coef, interrupt):
        """
            Test set에 대한 SVM Classification의 예측 결과를 시각화하는 함수.
            함께 제공된 ipynb 파일의 예시처럼 시각화를 수행하면 됩니다.
            Test set의 class 별 데이터를 색 구분을 통해 나타내며, 빨간색 선을 통해 학습시킨 모델이 예측한 hyperplane을 보여줍니다.

        """
        # X_test = [[실수, 실수],[실수, 실수]...] coef = [실수, 실수] 
        for i in range(len(y_test)): # y_test 값이 1이면 노란색, -1이면 보라색으로 색 구분한다, 
            if y_test[i] == 1:
                plt.scatter(X_test[i][0], X_test[i][1], c = "yellow")
            else :
                plt.scatter(X_test[i][0], X_test[i][1], c = "purple")

        # x_test의 x값의 최대 최소를 구간으로 갖는 x 변수 어레이를 구한다.
        x = np.linspace(np.min(np.split(X_test, 2, axis = 1)[0]), np.max(np.split(X_test, 2, axis = 1)[0]))
        y = (-coef[0] / coef[1]) * x + (interrupt / coef[1]) # 수식을 통해 y값을 구한다
        plt.plot(x,y, color = "red")
        plt.show()
        
        plt.savefig(self.author + '_' + self.id + '_' + 'visualization.png')
