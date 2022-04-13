# Attendance-check-model
Image Detection과 CNN을 활용한 비대면 수업 자동 출석체크 시스템

<center>
<img width="757" alt="image" src="https://user-images.githubusercontent.com/77783047/162958154-038be211-0144-4598-9a24-42718683348e.png">
</center>

# install
```
# Image objectation을 위한 yolov5
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install


# Image classification을 위한 CNN 모델
https://drive.google.com/file/d/1-EUAMx7jI1h6dKuh-DaJircK2cCOGfjU/view?usp=sharing
```
# requirement
 Python>=3.7.0 environment, including PyTorch>=1.7.
 
# Overview
최종 출석체크 시스템은 크게 두 파트로 구성됩니다. 
* Part1) Face Detection - 줌 화면에서 얼굴 영역을 인식하고, CNN모델이 인식하기 쉽도록 얼굴 이미지를 정규화합니다
* Part2) Face Classification - 누구의 얼굴인지 분류합니다. 
<img width="756" alt="image" src="https://user-images.githubusercontent.com/77783047/162958205-4456f7e9-2aa9-446d-953a-234a557ff341.png">

# Data Preprocessing
[Kaggle의 Face Mask Detection 데이터셋](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)과 직접 크롤링한 데이터 총 1,000여장에 데이터 증강을 적용하여 최종 6,527장의 데이터셋을 확보했습니다. 이때, 이목구비의 위치를 바꿔 정확도를 떨어트릴 것이라는 가설을 바탕으로 vflip과 rotation 기법은 제외하고, hflip과 grayscale 기법만을 적용하여 학습을 진행했습니다. 

<img width="758" alt="image" src="https://user-images.githubusercontent.com/77783047/162958279-0424f596-d614-4886-bfc9-076adcaaa405.png">
<img width="757" alt="image" src="https://user-images.githubusercontent.com/77783047/162958400-30e8795a-10e7-4387-ae43-f01219b5b21d.png">


# Face Detection by Yolov5
객체의 class와 위치(Bounding Box)를 빠르게 찾는 Yolov5의 pre-trained model을 가져와 줌 화면 속 다양한 얼굴들-마스크를 쓴 얼굴, 측면만 나온 얼굴 등-을 인식할 수 있게 학습시켰습니다. 

<img width="756" alt="image" src="https://user-images.githubusercontent.com/77783047/162958504-6ca12444-10bf-403f-ab82-d9c07391ba40.png">
<img width="759" alt="image" src="https://user-images.githubusercontent.com/77783047/162958530-f8b1cb13-f4b7-4843-9df7-ccad7216a7d9.png">

## Results
<img width="758" alt="image" src="https://user-images.githubusercontent.com/77783047/162958577-4b57fd27-ed97-4ba3-8fac-117712e574a5.png">

# Face Image Normalization by FaceLankmark
Yolov5로 인식한 얼굴을 CNN모델이 보다 더 잘 얼굴로 인식할 수 있게, FaceLandmark를 통해 얼굴의 이목구비 위치를 동일하게 정렬했습니다.

<img width="757" alt="image" src="https://user-images.githubusercontent.com/77783047/162958687-cc1617fe-6860-49f1-a4bd-491602854daf.png">
<img width="757" alt="image" src="https://user-images.githubusercontent.com/77783047/162958730-6f73777a-7325-489e-b22f-c97d179ef266.png">

## Results
<img width="755" alt="image" src="https://user-images.githubusercontent.com/77783047/162958776-e949ddfc-30d8-4396-aa55-0794c0cad8ca.png">

# Face Classification by 4 layer CNN 
출석여부를 판단하기 위해 Yolov5로 인식한 얼굴이 누구의 얼굴인지 분류할 수 있는 4층 CNN 모델을 만들었습니다.

## Train Dataset
CNN모델을 학습하기 위해선 실제 수강생들의 얼굴 이미지 데이터셋이 수집되어야 합니다. 해당 프로젝트에서는 팀원 6명의 다양한 얼굴 이미지를 확보하기 위해 동영상을 촬영한 뒤 프레임 단위로 이미지를 추출했습니다. 

<img width="701" alt="image" src="https://user-images.githubusercontent.com/77783047/162959145-45c9b841-b5c7-4d34-a104-99dd8938dcbb.png">

## Training model
* Convolutional Layer 4층, 활성함수는 ReLU, Optimizer는 Adam, Dropout 적용


<img width="701" alt="image" src="https://user-images.githubusercontent.com/77783047/162959396-5f786ce3-8b57-42e2-bb72-1ef999845c37.png">

## Results
10 epoch 훈련한 뒤, Test set에 대한 예측 정확도입니다. 

<img width="701" alt="image" src="https://user-images.githubusercontent.com/77783047/162959502-c5e7142b-6466-40eb-bb14-4aab78e532c6.png">

<img width="702" alt="image" src="https://user-images.githubusercontent.com/77783047/162959443-5a0d7089-cb18-45ee-abe7-778bb3460d4a.png">

## Final Output
최종 출석체크 시스템의 로직은 다음과 같습니다.


<img width="700" alt="image" src="https://user-images.githubusercontent.com/77783047/162959593-b38cfc7f-f797-45f2-8749-5a1530f686de.png">

# Conclusion 
<img width="701" alt="image" src="https://user-images.githubusercontent.com/77783047/162959662-e72937f7-02c5-43ad-8667-eab558a69fff.png">


