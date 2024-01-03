#project : evaluation 스크래치 / skku 기계학습 수업
#name : 조병웅

#임포트할 라이브러리
import numpy as np

#  accuracy 함수    
#  my_confusion_matrix 함수 
#  recall 함수 
#  precision 함수 
#  f1 함수 
#  TF-IDF 함수 

class my_evaluation_metrics:
    
    def __init__(self):
        self.author = __author__
        self.id = __id__

    def my_accuracy(self, y_true, y_pred):
        """
        정확도를 계산하는 함수입니다.
        Binary classification은 물론 Multi-label classification에도 적용 가능하도록 구현.

        y_true : 실제 값입니다.
        y_pred : 예측 값입니다.

        output type은 float 입니다.
        """
        temp = 0 # 실제값과 예측값이 동일한 횟수를 나타내는 임시값을 저장할 변수 생성
        result = 0 # 결과값을 저장할 변수 생성
        for i in range(len(y_true)): # 값들의 크기 만큼 순환한다.
            if y_true[i] == y_pred[i]: # 만약, 예측과 실제값이 일치한다면, temp를 1 증가시킨다. (tn, tp일 경우) 
                temp += 1
        
        result = temp / len(y_true) #전체의 경우와 tn, tp 경우 비율을 구한다.
        ## fill the blank
        return result


    def my_confusion_matrix(self, y_true, y_pred, pos_label=1):
        """
        Confusion Matrix를 출력하는 함수입니다.
        True Positive, True Negative, False Positive, False Negative를 위한 조건을 None 부분에 입력
        반드시 pos_label 변수를 활용하셔야 합니다.

        pos_label : Positive로 설정할 label (Binary classification에서 일반적으로 1을 뜻함)

        output type은 numpy array 입니다.
        """

        cm_result = [[0, 0], [0, 0]]
        for i, value in enumerate(y_pred):
            if (value == pos_label) & (value == y_true[i]): # 양성이고, 정답일 경우
                cm_result[1][1] += 1 #tp
            elif (value == pos_label) & (value != y_true[i]): # 양성이고, 오답일 경우
                cm_result[0][1] += 1 #fp
            elif (value != pos_label) & (value == y_true[i]): # 음성이고, 정답일 경우
                cm_result[0][0] += 1 #tn
            elif (value != pos_label) & (value != y_true[i]): # 음성이고, 오답일 경우
                cm_result[1][0] += 1 #fn

        return np.array(cm_result)


    def my_recall(self, y_true, y_pred):
        """
        Recall을 출력하는 함수입니다.
        Binary classification에서만 작동하도록 구현.
        
        output type은 float 입니다.
        """
        fn = 0 # fn 값을 저장할 변수를 생성
        tp = 0 # tp 값을 저장할 변수를 생성
        result = 0 # 결과를 저장할 변수를 생성
        for i in range(len(y_true)): # 케이스의 수만큼 순회한다,
            if y_true[i] == 1: # 만약 진짜 스펨메일일 경우
                if y_pred[i] == 1: # 그리고 예측도 맞알을 경우
                    tp += 1 # tp의 경우이기 때문에 1을 증가시킨다.
                else :
                    fn += 1 # 그렇지 않다면 tn의 경우이다. 
        result = tp / (fn + tp) # 구한 변수들의 최종적인 값을 통해 계산
        ## fill the blank
        return result


    def my_precision(self, y_true, y_pred):
        """
        Precision을 출력하는 함수입니다.
        Binary classification에서만 작동하도록 구현.
        
        output type은 float 입니다.
        """
        result = 0 # 결과를 저장할 변수를 생성
        tp = 0 
        fp = 0
        for i in range(len(y_true)): #크기를 맞춰 순회
            if y_pred[i] == 1: # 예측이 스펨일 모든 경우
                if y_true[i] == 1: # 그중에서 진짜 스펨메일이었을 경우
                    tp += 1 # tp의 경우이기에 1을 증가시킨다
                else :
                    fp += 1 # fp의 경우이기에 1을 증가시킨다
        result = tp / (fp + tp) # 구한 변수들을 통해 계산

        ## fill the blank
        return result
    

    def my_f1(self, y_true, y_pred):
        """
        F1 score를 출력하는 함수입니다.
        Binary classification에서만 작동하도록 구현.
        
        output type은 float 입니다.
        """
        fn = 0
        tp = 0
        my_recall = 0
        for i in range(len(y_true)):
            if y_true[i] == 1:
                if y_pred[i] == 1:
                    tp += 1
                else :
                    fn += 1
        my_recall = tp / (fn + tp)
        # 위에서 쓰인 recall 함수를 그대로 가져와 변수에 올바른 값을 저장한다.
        my_precision = 0
        tp = 0
        fp = 0
        for i in range(len(y_true)):
            if y_pred[i] == 1:
                if y_true[i] == 1:
                    tp += 1
                else :
                    fp += 1
        my_precision = tp / (fp + tp)
        # 위에서 쓰인 precision 함수를 가져와 변수에 올바른 값을 대입한다.
        result = 2*((my_precision*my_recall) / (my_recall + my_precision))
        ## 식을 구현해 결과 변수에 저장한다.
        return result


    def my_tf_idf(self, documents):
        """
        TF-IDF를 출력하는 함수입니다.
        교안을 참고하여 tf_idf 변수에 적합한 값을 입력.
        tf_idf의 shape은 (len(documents), len(word_list))임을 유의하세요.
        """

        # 전체 documents에서 등장하는 word의 리스트입니다.
        word_list = [] 
        for doc in documents:
            splited = doc.split(' ')
            for word in splited:
                if word not in word_list:
                    word_list.append(word)

        # TF-IDF를 연산하기 위해 numpy array를 초기화합니다.
        tf_idf = np.zeros((len(documents), len(word_list)))

        ## tf_idf와 똑같은 크기와 차원을 가진 tf 어레이를 만든다. tf는 출현 빈도를 요소로 가진다. 
        tf = np.zeros((len(documents), len(word_list)))
        for i in range(len(word_list)):
            for j in range(len(documents)):
                if word_list[i] in documents[j]:
                    tf[j][i] += documents[j].count(word_list[i]) 
                    # 해당문서에 해당 단어가 있다면, 해당하는 tp 순서에 값을 추가한다.
        
        df = [] # df는 word_list의 index와 동일한 index 순서를 가지나, 요소로 해당 단어가 나온 문서의 횟수를 저장한다.
        for i in range(len(word_list)):
            temp = 0
            for j in range(len(documents)):
                if tf[j][i] > 0:
                    temp += 1
            df.append(temp)

        idf = np.zeros(len(word_list))
        for i in range(len(word_list)):
            idf[i] += (np.log((len(documents)) / (1+ df[i])))  
        # df를 활용해 idf를 구한다. 그리고 아래 식을 구현한다.
        tf_idf = idf * tf
        return tf_idf







