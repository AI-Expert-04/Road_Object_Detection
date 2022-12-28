# RoadObject-Detection

## Window Python 환경

- anaconda 4.6.11
  - python 3.6
  - numpy 1.17.3
  - matplotlib 3.1.1
  - tensorflow 2.0.1
  - python-opencv
  - pyqt5


## Mac Python 환경 
- anaconda 4.6.11
  - python 3.8 
  - matplotlib #전부 최신버전 !
  - conda install tensorflow  #전부 최신버전 !
  - python-opencv  #전부 최신버전 !
  - pyqt5 #전부 최신버전 !

## 환경 설정

1. Window conda 환경 생성 

```
>>> conda create -n [가상환경 이름]-env python=3.6

```
1. Mac conda 환경 생성 

```
>>> conda create —name [가상환경 이름]-env python=3.8
```


2. conda 환경 활성화

```
>>> conda activate [가상환경 이름]-env
```



3. 프로젝트 경로로 이동

```
>>> cd (프로젝트 경로)
```



4. python 모듈 설치

```
>>> pip install -r requirements.txt
```  
# report [Link](https://docs.google.com/document/d/16T0VQJriU-VXSOssZI7Cu45VG0dLgNY1N7YgtebJXVk/edit?usp=sharing)

5. 데이터 수집
DBB100K data

6. 데이터 로드

7. 데이터 분류

8. 데이터 전처리(one hot encoding)

9. 전처리된 데이터 분류 Train_set(70%), Validation_set(28%), Test_set(2%)

10. 학습 
1000개의 사물을 충분히 학습된 가중치인 imagenet을 전이학습 시켜 학습에 도움을 줌

11. 시각화(예측)

12. yolov3 model VS RetinaNet 성능 비교 및 코드 분석






