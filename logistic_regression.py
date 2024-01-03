#project : 로지스틱 회귀 구현 / skku 기계학습 수업
#name : 조병웅

#임포트할 라이브러리
import numpy as np

#  activation 함수 
#  fwpass 함수 
#  bwpass 함수 
#  initialize_w 함수 
#  fit 함수 
#  predict 함수 
#  feature_importance 함수 


class LogisticRegression:
    def __init__(self, max_iter=500, penalty="l2", initialize = "one", random_seed = 1213):
        self.author = __author__
        self.id = __id__
        
        self.max_iter = max_iter
        self.penalty = penalty
        self.initialize = initialize
        self.random_seed = random_seed
        self.lr = 0.1
        self.lamb = 0.01
        np.random.seed(self.random_seed)

        if self.penalty not in ["l1", "l2"]:
            raise ValueError("Penalty must be l1 or l2")

        if self.initialize not in ["one", "LeCun", "random"]:
            raise ValueError("Only [LeCun, One, random] Initialization supported")

            
    def activation(self, z): #로지스틱의 경우, 시그모이드
        a = 1/(1+np.exp(-z)) #sigmoid함수 수식을 파이썬 코드로 바꾼 수식
        return a


    def fwpass(self, x): #내적연산 과정
        """
        x가 주어졌을 때, 가중치 w와 bias인 b를 적절히 x와 내적하여
        아래의 식을 계산하세요.

        z = w1*x1 + w2*x2 + ... wk*xk + b
        
        z = 0
        x1 = [i[0] for i in x] # x[i]의 첫번째 파라미터만 모은 리스트를 만든다.
        x2 = [i[1] for i in x]
        x3 = [i[2] for i in x]
        x4 = [i[3] for i in x]
        x5 = [i[4] for i in x]
        x6 = [i[5] for i in x]
        x7 = [i[6] for i in x]
        x1_data = np.array(x1) # 해당 리스트를 계산을 위하여 어레이로 변환한다.
        x2_data = np.array(x2)
        x3_data = np.array(x3)
        x4_data = np.array(x4)
        x5_data = np.array(x5)
        x6_data = np.array(x6)
        x7_data = np.array(x7)
        """
        # 내적하는 수식을 작성한다. 
        #z = self.w[0]*x1_data + self.w[1]*x2_data + self.w[2]*x3_data + self.w[3]*x4_data + self.w[4]*x5_data + \
        #    self.w[5]*x6_data + self.w[6]*x7_data + self.b
        z = np.dot(x, self.w) + self.b
        # Code Here! 이 부분에 w1*x1 + w2*x2 + ... wk*xk + b 의 값을 가지도록 계산하세요. (넘파이 행렬 활용 추천) 
        z = self.activation(z) #시그모이드 함수로 0~1사이의 값으로 바꾼다.                
        return z


    def bwpass(self, x, err): #그레디언트 구하기 함수
        #x는 712크기의 이중리스트, err는 712 크기의 리스트(712,1) (예시)
        """
        x와 오차값인 err가 들어왔을 때, w와 b에 대한 기울기인 w_grad와 b_grad를 구해서 반환.
        l1, l2을 기반으로한 미분은 https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261
        이 문서를 확인하세요.

        w_grad는 (num_data, num_features) 712,7
        b_grad는 (num_data, )

        의 데이터 shape을 가지는 numpy array 라고 예시로 규정.
        w_grad 계산 부분에서 속도의 차이가 발생

        self.lamb을 통해 lambda를 활용하세요.
        """
        #x = (712,7) err = (712,)
        if self.penalty == "l1":
            sum = 0 #시그마 수식처럼 특정 값들을 순회하며 저장하는 변수 sum을 정의한다. 
            for i in range(len(self.w)):
                sum += abs(self.w[i]) #L1 정규화를 만들기 위하여 abs()를 사용해 절대값으로 바꾼뒤 sum을 구한다.
            w_grad = (2/len(x))*(np.dot(err,x)) + self.lamb*sum #내적 dot()을 사용하여 gradient 수식을 코드화한다.
        elif self.penalty == "l2":
            sum = 0 # 위와 동일한 과정
            for i in range(len(self.w)):
                sum += self.w[i]**2 #L2정규화를 만들기 위해 ** 연산자로 제곱한다. 
            w_grad = (2/len(x))*(np.dot(err,x)) + self.lamb*sum #위와 동일환 과정
        
        b_grad = (2/len(x))*np.sum(err) #편향값을 업데이트하는 공식.

        return w_grad, b_grad

    
    def initialize_w(self, x): #초기화 함수
        """
        L8(이번주차 강의)-NN-GD2와 https://reniew.github.io/13/ 의 LeCun 초기화 수식을 참고하여
        LeCun 가중치 초기화 기법으로 초기 w를 설정할 수 있도록 코딩. (힌트: np.random.uniform 활용)
        동일하게 랜덤한 값으로 w가중치를 초기화.

        w_library에서 one과 같은 shape이 되도록 다른 값을 설정.
        """
        w_library = {
            "one":np.ones(x.shape[1]),
            "LeCun":np.random.uniform(low = -np.sqrt(1.0 / x.shape[1]), high = np.sqrt(1.0 / x.shape[1]), size = x.shape[1]), 
            #random 라이브러리를 사용하여 가중치 초기화 수식을 작성한다. 제곱근은 sqrt메서드로 구현한다.  
            "random":np.random.randint(0,1, size = x.shape[1]) 
            # randint메서드를 사용해 구현. 
        }

        return w_library[self.initialize]


    def fit(self, x, y):
        """
        실제로 가중치를 초기화 하고, 반복을 진행하며 w, b를 미분하여 계산하는 함수입니다.
        다른 함수를 통하여 계산이 수행되니 self.w, self.b 의 업데이트 과정만 코딩하세요.
        """
        self.w = self.initialize_w(x)
        self.b = 0
        for _ in range(self.max_iter):
            z = self.fwpass(x)
            err = -(y - z) # 712크기의 리스트
            w_grad, b_grad = self.bwpass(x, err)



            # (각 gradient에 learning_rate을 곱한 후 평균을 활용하여 값을 업데이트)
            self.w = self.w - self.lr*w_grad #앞에서 1/n으로 나누었기 때문에, 평균을 따로 활용하지 않았다.

            # Code Here! b를 b_grad를 활용하여 업데이트. 
            self.b = self.b - self.lr*self.b #위와 동일하다.

        return self.w, self.b


    def predict(self, x): #0,1로 변환
        """
        test용 x가 주어졌을 때, fwpass를 통과한 값을 기반으로
        0.5초과인 경우 1, 0.5이하인 경우 0을 반환하세요.
        """
        z = self.fwpass(x)
        for i in range(len(z)):
            if z[i] > 0.5: #for문을 순회하면서 값을 비교하며, 초과하면 해당 값에 1을 저장한다.
                z[i] = 1
            else :
                z[i] = 0
        return z

    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
    
    
    def feature_importance(self, coef, column_to_use):
        """
        예시 과제에서 사용한 feature들의 중요도를 '순서대로' 보여줌.
        함께 제공된 ipynb 파일의 예시처럼 "순위, feature, 해당 feature의 가중치"가 모두 나타남.
        가중치의 순위를 활용하여 코딩. (힌트: np.argsort)
        """
        w = self.w.copy() #완전히 분리된 변수를 위해 copy()를 통해 복제한다.
        for i in range(len(self.w)): 
            w[i] = abs(self.w[i]) #w변수는 self.w의 값에 절댓값을 씌운 값이다.
        dic = {} # 해당하는 위치를 가진 컬럼 이름들을 찾아주기 위해 dict를 사용한다.
        for i in range(len(w)):
            dic[w[i]] = column_to_use[i]
        s = w.argsort() #argsort를 이용해 정렬하고, 정렬 순서를 다른 어레이에도 전달한다.
        w = w[s]
        self.w = self.w[s]  #절대값크기에 따른 정렬 순서에 따라 정렬된다.   
        rank_li = [] # 결과를 보관할 변수
        for i in range(len(column_to_use)):
            rank_li.append(((len(w)-(i+1)), [dic[w[i]], self.w[i]])) #결과를 저장한다.
        rank_li.reverse()
        return rank_li
