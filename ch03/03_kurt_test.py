##3.6 손글씨 숫자 인식 직접 구현해보기
import sys, os
#sys.path.append(os.pardir)  #부모 디렉토리의 파일을 가져올 수 있도록 설정
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # 부모 디렉터리의 파일을 가져올 수 있도록 설정

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle


#해당 코드를 실행할려고 dataset 폴더를 ch03 밑으로 옮김
from dataset.mnist import load_mnist

'''
load_mnist는 ("훈련 이미지, 훈련 레이블"), ("시험 이미지", "시험 레이블") 형식으로 반환
인수로는 normalize, flatten, one_hot_label 3가지 설정이 가능
normalize = 픽셀의 값을 0~1사이로 정규화 할지  (True = 0~1, False = 0~255)
flatten = 픽셀배열을 1차원 배열로 만들지 결정 (True = 784 1차원 배열로 , False = 1*28*28 3차원 배열로)
one_hot_label = 레이블을 one-hot encoding 형태로 저장할지 정한다

one-hot encoding = [0,0,1,0,0,0,0,0,0] 처럼 정답을 뜻하는 원소만1이고 나머지는 0모두 0인 배열
                 (True =위와 같은 인코딩 형태로 저장, False = '7'이나'2'와 같은 숫자 형태의 레이블 저장)
                 
Python - pickle 기능 이란??
  - 프로그램 실행중에 특정 객체를 파일로 저장하는 기능
  - 저장해둔 pickle 파일을 로드하면 실행 당시의 객체를 즉시 복원할 수 있다
  - MNIST 데이터셋을 읽는 load_mnist() 함수에서도 (2번째 이후의 읽기 시) pickle을 사용
  - pickle 덕분에 MNIST 데이터를 순식간에 준비 할 수 있다
  
이미지 표시에는 PIL 모듈을 사용 (PIL = Python Language Library)
'''

#시그모이드 함수 구현하기
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#소프트맥스(softmax) 함수 개선버전
def softmax(a):
    C = np.max(a)
    exp_a = np.exp(a - C)   #overflow 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def img_show(img):
  pil_img = Image.fromarray(np.uint8(img))
  pil_img.show()
  
def img_show_plt(img):
  plt.imshow(img, cmap='grey')
  plt.axis('off') #no axis ticks
  plt.show()
    
def get_data():
  (x_train, t_train),(x_test, t_test) \
  = load_mnist(normalize=True, flatten=True, one_hot_label=False)
  return x_test, t_test

def init_network():
 with open(os.path.dirname(__file__) + "/sample_weight.pkl", 'rb') as f:
      network = pickle.load(f)
      return network
  

def predict(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']
  
  a1 = np.dot(x,W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = softmax(a3)
  return y
  



#################################################################  
    
#처음 한번은 몇분 정도 걸린다
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

#각 데이터의 형상 출력
print(x_train.shape)    #(60000, 784)
print(t_train.shape)    #(60000, )
print(x_test.shape)     #(10000, 784)
print(t_test.shape)     #(10000)

img = x_train[0]
label = t_train[0]
print(label)            #5

print(img.shape)        #(784, )
img = img.reshape(28, 28) 
print(img.shape)        #(28, 28)

#img_show(img)     #그림파일 실행
#img_show_plt(img) #그림파일 그림


x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
  y = predict(network, x[i])
  p = np.argmax(y)  #확률이 가장 높은 원소의 인덱스를 얻는다
  if p == t[i]:
    accuracy_cnt += 1
  
print("Accuracy : " + str(float(accuracy_cnt) / len(x))) # Accuracy : 0.9352

