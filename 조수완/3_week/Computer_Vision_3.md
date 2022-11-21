# Computer Vision3
## Data

- 모든 머신 러닝 workflow의 중요한 단계
- 머신 러닝 엔지니어가 대부분의 시간을 투자하는 부분 → 데이터 파이프 라인을 구축하고, 데이터 시각화(Data visualization)를 만들고 dataset 에 대해 가능한 많이 이해하려고 노력

## Exploratory Data Analysis (EDA)

탐색적 데이터 분석(EDA)는 모든 ML 프로젝트에서 중요함 → 이 단계에서 ML 엔지니어는 dataset에 익숙해지고 데이터의 잠재적 문제를 발견

- weather / light conditions
    - 맑은 이미지로만 훈련된 알고리즘은 비가 오거나 야간 데이터가 있을 때는 잘 작동되지 않음
- sensor
    - 센서 변경 또는 다른 처리 방법은 domain shift를 발생시킴
- environment
    - 저강도 트래픽 데이터에 대해 훈련된 알고리즘은 고강도 트래픽 데이터에서 잘 작동되지 않음

## Cross Validation (교차 검증)

모델의 일반화(generalization) 능력을 평가

- overfitting : 모델이 일반화가 잘 되지 않은 경우
- bias-variance tradeoff : 균형있는 모델을 만드는 것이 왜 어려운가?
- cross validation : 모델이 일반화를 얼마나 잘하는지 평가하는 기술

### 1. Overfitting

모델이 overfitting 되면 일반화할 수 있는 능력을 잃는다.

선택한 모델이 너무 복잡하고 의미 있는 기능 대신 노이즈 추출을 시작할 때 종종 발생

ex) 자동차 감지 모델 → 더 넓은 특징(바퀴, 모양 등) 대신에 데이터 세트에서 자동차의 브랜드별 특징(자동차 로고 등)을 추출하기 시작할 때를 overfitting 이라 할 수 있다.

![image](https://user-images.githubusercontent.com/79623246/203062689-2c31053a-c399-453c-9b5d-dd1e85ab5123.png)

  ㅤㅤㅤㅤㅤㅤ^ 선형 회귀 모델ㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤㅤ^ 다항 회귀 모델

다항 회귀 모델이 더 정확함. 하지만, 새로운 데이터가 생겼을땐?

![image](https://user-images.githubusercontent.com/79623246/203063074-3f0ca6cd-591b-4403-8728-e8e8b395dd4e.png)

선형 회귀 모델이 더 정확함 → 이를 새로운 데이터에 잘 일반화된다고 함

⇒ ML 프로젝트의 목표는 overfitting 없이 사용 가능한 데이터에서 우수한 성능을 발휘하는 모델을 만드는 것

데이터 용어

- Training dataset : ML 모델 훈련엔 사용되는 이용가능한 데이터
- Test dataset : 훈련에 사용하지 않은 데이터 → overfitting 해결을 위해 사용됨

### 2. Bias & Variance Trade-off

$TestError = Var + Bias + ε (epsilon)$

- Variance : training data 에 대한 모델의 민감성를 정량화
    - 분산이 낮다 → training data에 민감하지 않고 잘 일반화 됨
- Bias : training data 에 대한 모델의 적합도를 정량화
    - 편향이 낮다 → training dataset에서 오류율이 매우 낮다
- Test error : test dataset 에 대한 error 비율
- Variance와 Bias 모두 낮아야 함 → training data와 test data 모두에서 우수한 성과를 발휘해야 함
- Bias 최소화로 training dataset 의 overfitting 으로 Variance가 증가되는 경향이 있다.

![image](https://user-images.githubusercontent.com/79623246/203063118-b6310638-4cb5-4664-ba99-2eebab90f379.png)

### 3) Validation Sets & Cross Validation

Cross Validation 은 overfitting 을 완화하기 위해 모델의 능력을 평가하는 일련의 기술

이 과정에서 사용 가능한 데이터를 두개로 분할하는 vaildation set approach 방식을 사용

- A training set : 알고리즘을 생성하는데 사용됨(일반적으로 사용 가능한 데이터의 80~90%)
- A validation set : 평가하는데 사용(사용 가능한 데이터의 10~20%)

LOO(Leave One Out) 또는 k-fold cross validation 과 같은 cross validation 방법이 존재하지만 DL 알고리즘에는 적합하지 않음
