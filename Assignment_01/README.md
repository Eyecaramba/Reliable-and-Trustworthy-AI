# MNIST와 CIFAR-10에서의 분류 모델 공격 실험

이 프로젝트는 **MNIST**와 **CIFAR-10** 데이터셋에서 분류 모델을 학습한 뒤, **FGSM(Fast Gradient Sign Method)** 및 **PGD(Projected Gradient Descent)** 공격을 수행하여 모델의 취약성을 분석하는 실험을 다룹니다. 

## 프로젝트 개요
- **MNIST**: 간단한 사용자 정의 CNN 모델을 학습합니다.
- **CIFAR-10**: PyTorch에서 제공하는 ResNet18 모델을 학습합니다.
- 학습된 모델에 대해 **FGSM** 및 **PGD** 공격(타겟팅/비타겟팅)을 수행하여 공격 전후의 정확도 및 공격 성공률(ASR, Attack Success Rate)을 측정합니다.

---

## Config 설정
이 프로젝트는 `config.ini` 파일을 사용하여 다양한 실험 설정을 조정할 수 있습니다. 아래는 `config.ini` 파일의 기본 구조와 각 설정값에 대한 설명입니다:

### **config.ini 내용**
```
[DEFAULT]
Dataset=mnist
AttackMethod=untargeted_fgsm
Epsilon=0.03
K=10
EpsStep=0.01
BatchSize=64
LearningRate=0.001
Epochs=5
```




### **설정값 설명**
1. **Dataset**:
   - 사용할 데이터셋을 지정합니다.
   - 가능한 값: `mnist`, `cifar`.

2. **AttackMethod**:
   - 수행할 공격 방법을 지정합니다.
   - 가능한 값:
     - `targeted_fgsm`: 타겟팅된 FGSM 공격.
     - `untargeted_fgsm`: 비타겟팅 FGSM 공격.
     - `targeted_pgd`: 타겟팅된 PGD 공격.
     - `untargeted_pgd`: 비타겟팅 PGD 공격.

3. **Epsilon**:
   - 허용되는 최대 perturbation 크기(공격 강도).

4. **K**:
   - PGD 공격에서 반복(iteration) 횟수.

5. **EpsStep**:
   - PGD 공격에서 각 반복(iteration)마다 적용되는 perturbation 크기.

6. **BatchSize**:
   - 학습 및 평가에 사용할 배치 크기.

7. **LearningRate**:
   - 학습 시 사용할 학습률.

8. **Epochs**:
   - 모델 학습 시 수행할 에포크 수.

---

## 실행 방법

### 1. 라이브러리 설치 
프로젝트 실행 전에 필요한 라이브러리를 설치합니다. 프로젝트 디렉토리에서 다음 명령어를 실행하세요:
`pip install -r requirements.txt`


### 2. 실험 실행
`test.py`를 실행하여 모델 학습, 평가, 공격 수행 및 결과를 확인할 수 있습니다:


