#project : 나이브 베이지 파이썬 스크래치 (0,1 이진 레이블 경우) / skku 기계학습 수업
#name : 조병웅

#라이브러리 임포트
import numpy as np


#나이브 베이즈 분류기 함수
class NaiveBayesClassifier:
    def __init__(self, smoothing=1):
        #self.author = __author__
        #self.id = __id__
        self.smoothing=smoothing #우도가 0이 될 일이 없게 하는 라플라스 스무딩
        self.epsilon = 1e-10

    def fit(self, x, y):

        """        
        실제적인 훈련(확률 계산)이 이뤄지는 부분입니다.
        self.data에 [도큐먼트수 x 단어피쳐수(500)]인 넘파이 행렬이
        self.labels에 각 도큐먼트에 대한 라벨인 [도큐먼트수, ]인 넘파이 행렬이 저장됩니다.

        본 함수가 호출된 이후, self.label_index 변수에 아래와 같은 딕셔너리형태가 저장됩니다.
        self.label_index = {0: array([   1,    4,    5, ..., 3462, 3463, 3464]), 1: array([   0,    2,    3, ..., 3447, 3449, 3453])}
        0번 라벨의 도큐먼트 id가 넘파이 어레이 형태로, 1번 라벨의 도큐먼트 id가 넘파이 어레이 형태로 위와 같이 정리됩니다.

        이후, label_index, prior, likelihood를 순차적으로 계산하여 줍니다.
        아래에서 호출되는 self.get_... 함수들은 계산 함수입니다.
        """

        self.data = x # 카운트기반 벡터화된 단어 데이터
        self.labels = y # 0과 1로 이루어진 정답 데이터

        self.label_index = dict() # 라벨 0, 1에 속한 도큐먼트 ID를 각각의 밸류로 구분하기 위한 딕셔너리.
        self.label_name = set(self.labels) # 라벨이 총 몇개로 이루어졌는지 구하기 위해 중복을 허용하지 않는 set사용
        for lab in self.label_name: 
            self.label_index[lab] = [] # 0,1 라벨을 키로 생성

        for index, label in enumerate(self.labels): 
            self.label_index[label].append(index) # 0을 가지는 라벨 데이터들은 0을 키로 가지는 딕셔너리로 넣고, 인덱스를 밸류로 추가
            
        for lab in self.label_name:
            self.label_index[lab] = np.array(self.label_index[lab]) # 어레이로 치환.

        self.get_prior()
        self.get_likelihood()


    def get_prior(self):
        """
        prior(사전확률)를 계산하는 함수입니다.
        본 함수가 처리된 이후, self.prior 변수에 라벨이 key, 라벨에 대한 prior가 value로 들어가도록 하세요.
        self.prior = {0: 0번 라벨 prior[실수값], 1: 1번 라벨 prior[실수값]}
        """
        self.prior = dict()

        for key in self.label_name: # 각 라벨을 키로 생성하기 위해 for문을 사용.
            self.prior[key] = len(self.label_index[key])/len(self.labels) # 각 라벨의 밸류로 prior를 저장하는데,
            # 이때, len함수를 사용해 해당 '라벨 데이터의 개수 / 총 데이터 개수'를 구한다.  
        return self.prior


    def get_likelihood(self):

        """
        likelihood(우도)를 계산하는 함수입니다. 
        구하는 방법은 특정 레이블에 등장한 모든 단어의 빈도 수의 총합을 분모로하고, 
        그 레이블서 특정 단어가 총 등장한 빈도의 수를 분자로 하는 것입니다.
        본 함수가 처리된 이후, self.likelihood에 라벨이 key, 라벨에 대한 단어별 likelihood를 계산하여 value로 넣어주세요.

        """
        # likelihood는 단어에 대하여, 클래스 내 모든 단어의 총 출현 횟수와 해당 단어의 출현 횟수, 그리고 스무딩을 고려해야한다.

        label_data_sum = dict() # 라벨을 키로 갖고, 단어별 likelihood를 self.data의 인덱스에 맞는 위치에 갖는 value를 구한다. 이때, 크기를 맞추기 위하여 어레이는 500의 크기를 갖는다.
        data_sum = dict() # 클래스내 모든 단어의 총 출현 횟수를 구하기 위한 변수

        for i in self.label_name: 
            label_data_sum[i] = np.zeros(500) # 0과 1을 갖는 키에 500크기의 어레이를 만든다. 다른 값이 들어가지 않도록 제로 벡터로 만든다.

            for j in self.label_index[i]: 
                label_data_sum[i] += self.data[j] # 각 라벨의 label_index를 통해 각 라벨에만 속한 데이터 어레이를 불러오고, 이 데이터 어레이들을 각 라벨 키에 맞게 합한다.
                # 이때 label_data_sum은 각 키에 맞는 데이터 어레이들의 합만을 갖는다 -> 이것의 인덱스에 속한 값을 해당 값에 인코딩된 인덱스 단어의 출현 횟수로 구한다.
            
            data_sum[i] = label_data_sum[i].sum()

        self.likelihood = {} 

        for i in self.label_name: 
            self.likelihood[i] = np.zeros(500) # 인덱스 값을 self.data의 인코딩된 인덱스 값과 일치시키기 위해, 500 크기의 np영벡터를 만든다. 
            for j in range(500): # 각 인코딩된 값과 일치되는 인덱스에 밸류 값을 넣기 위해 500크기를 사용하는 for문을 사용
                self.likelihood[i][j] = label_data_sum[i][j] + self.smoothing / data_sum[i] # likelihood 계산식을 사용한다.
                
        return self.likelihood


                



    def get_posterior(self, x):

        """
        self.likelihood와 self.prior를 활용하여 posterior(사후확률)를 계산하는 함수입니다.
        0, 1 라벨에 대한 posterior를 계산하세요.

        Overflow를 막기위해 log와 exp를 활용합니다. 아래의 식을 고려해서 posterior를 계산.
        posterior 
        = prior * likelihood 
        = exp(log(prior * likelihood))  refer. log(ab) = log(a) + log(b)
        = exp(log(prior) + log(likelihood))

        nan을 막기 위해 possibility 계산시에 분모에 self.epsilon을 더해주세요.

        """
        self.posterior = [] 
        likelihood = self.likelihood
        prior = self.prior
        for i in range(len(x)): # x의 어레이 데이터 각각의 경우에 posterior를 구하고자 for문을 사용
            temp_0 = 0 # 라벨이 0일 경우 값을 저장하기 위한 변수
            temp_1 = 0 # 1일때

            for j in range(500): # 크기가 500인 어레이 데이터를 순회하며 단어를 가지고 있는지 여부 확인하기 위한 for문
                if x[i][j] != 0: # 단어가 하나 이상 존재할때
                    temp_0 = temp_0 + np.log(likelihood[0][j]) #log를 통해 연산
                    temp_1 = temp_1 + np.log(likelihood[1][j])

            temp_0 = np.exp(temp_0 + np.log(prior[0])) # prior 연산
            temp_1 = np.exp(temp_1 + np.log(prior[1]))

            temp_0 = temp_0 / (temp_0 + temp_1 + self.epsilon) # 확률 구하는 수식.         
            temp_1 = 1 - temp_0

            self.posterior.append([temp_0, temp_1]) # 각 확률을 리스트에 저장한다.            


        return self.posterior


    def predict(self, x):
        """
        이 함수는 수정하지 않아도 됩니다.
        likelihood, prior를 활용하여 실제 데이터에 대해 posterior를 구하고 확률로 변환하는 함수.
        """
        posterior = self.get_posterior(x)
        return np.argmax(posterior, axis=1)



