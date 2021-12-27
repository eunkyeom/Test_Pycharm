"""
# 신경망 구현 순서
1. Sequential 모형 클래스 객체 생성
2. add 메서드로 레이어 추가.
  - 입력단부터 순차적으로 추가한다.
  - 레이어는 출력 뉴런 갯수를 첫번째 인수로 받는다.
  - 최초의 레이어는 input_dim 인수로 입력 크기를 설정해야 한다.
  - activation 인수로 활성화 함수 설정
3. compile 메서드로 모형 완성.
  - loss 인수로 비용함수 설정
  - optimizer 인수로 최적화 알고리즘 설정
4. fit 메서드로 트레이닝
  - epochs로 에포크(epoch) 횟수 설정
"""

# tensorflow lib. 설치(prompt 창에서), 텐서플로우를 이용해서 모델 생성, 학습된 모델로 예측치까지 도출
# - conda install tensorflow
import tensorflow as tf
import numpy as np

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

tf.model = tf.keras.Sequential() # 시퀀셜, 반환되어지는 객체를 모델, 모델 생성을 위한 대상을 먼저 생성

# input_dim: input shape, units: output shape, 모델을 생성하는 과정
tf.model.add(tf.keras.layers.Dense(input_dim=1, units=1))

# SGD(Stochastic Gradient Descent, Gradient Descendent), lr(learning rate)
sgd = tf.keras.optimizers.SGD(lr=0.1)

# mse(mean_squared_error), 컴파일 메서드는 매개변수를 이용해서, 잔차의 값을 제곱해서 손실함수(비용함수), 알고리즘 세팅
tf.model.compile(loss='mse', optimizer=sgd)

tf.model.summary()

tf.model.fit(x_train, y_train, epochs=200) # 200번 반복하도록 세팅팅
# 예측값
y_predict = tf.model.predict(np.array([5, 4]))
print(y_predict)










