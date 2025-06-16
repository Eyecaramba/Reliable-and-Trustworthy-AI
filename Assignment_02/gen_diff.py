# gen_diff_cifar10_tf2.py (Final TF2 Version)

import argparse
import random
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, UpSampling2D, Flatten, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, UpSampling2D
from tensorflow.keras.datasets import cifar10
from PIL import Image # Pillow 라이브러리에서 Image 모듈을 가져옵니다.

# 원본 DeepXplore 코드의 헬퍼 파일
from configs import bcolors
from utils import init_coverage_tables, update_coverage, neuron_covered, neuron_to_cover, deprocess_image, normalize, constraint_light, constraint_occl, constraint_black, neuron_to_cover_V2

# ==============================================================================
# 0. 설정 및 파라미터
# ==============================================================================

# 커맨드 라인 인자(argument) 분석
parser = argparse.ArgumentParser(
    description='Main function for difference-inducing input generation in CIFAR-10 dataset (TF2)')
# ... (argparse 부분은 이전과 동일하게 유지) ...
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparam to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparam to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(10, 10), type=tuple)

args = parser.parse_args()

# ==============================================================================
# 1. 데이터셋 및 모델 로딩
# ==============================================================================

# CIFAR-10에 맞는 입력 이미지 차원 설정
img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)
num_classes = 10
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
inputs1 = Input(shape=input_shape)
inputs2 = Input(shape=input_shape)
def create_cifar10_resnet50():
    inputs = Input(shape=input_shape)
    x = UpSampling2D(size=(7, 7))(inputs)
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = resnet(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Build model structures
print("Building model structures...")
model1 = create_cifar10_resnet50()
model2 = create_cifar10_resnet50()

# Compile models
print("Compiling models...")
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load trained weights
print("Loading weights...")
model1.load_weights('resnet50_cifar10_seed10.h5')  # Ensure this file exists
model2.load_weights('resnet50_cifar10_seed20.h5')  # Ensure this file exists

# 뉴런 커버리지 테이블 초기화
print("Initializing coverage tables...")
model_layer_dict1, model_layer_dict2 = init_coverage_tables(model1, model2)

# CIFAR-10 테스트 데이터 로드 및 정규화
print("Loading CIFAR-10 data...")
(_, _), (x_test, _) = cifar10.load_data()
x_test = x_test.astype('float32') / 255.0

# ==============================================================================================
# 2. 차이 유발 입력 생성 시작
# ==============================================================================================

# 생성된 입력을 저장할 디렉토리 생성
if not os.path.exists('./generated_inputs'):
    os.makedirs('./generated_inputs')

# 시드 이미지 개수만큼 반복
for i in range(args.seeds):
    # 테스트 이미지 인덱스로 파일명 저장
    save_name_prefix = 'test_img_{0}'.format(i)
    print("--- Running {0}/{1} ---".format(save_name_prefix, args.seeds))

    # 테스트 이미지 선택
    gen_img_numpy = np.expand_dims(x_test[i], axis=0)
    orig_img = gen_img_numpy.copy()

    # 1. 먼저 원본 이미지가 이미 차이를 유발하는지 확인
    pred1 = model1.predict(gen_img_numpy, verbose=0)
    pred2 = model2.predict(gen_img_numpy, verbose=0)
    label1, label2 = np.argmax(pred1[0]), np.argmax(pred2[0])

    if label1 != label2:
        print(bcolors.OKGREEN + 'Input already causes different outputs: {}, {}'.format(
            cifar10_labels[label1], cifar10_labels[label2]) + bcolors.ENDC)
        gen_img_deprocessed = deprocess_image(gen_img_numpy)
        Image.fromarray(gen_img_deprocessed.astype('uint8')).save('./generated_inputs/{0}_already_differ.png'.format(save_name_prefix))
        continue

    # 2. 두 모델의 예측이 동일하면, 차이 유발 시작
    orig_label = label1
    print("All models agree on '{0}'. Starting differential testing...".format(cifar10_labels[orig_label]))

    # 아직 커버되지 않은 뉴런 선택
    layer_name1, index1 = neuron_to_cover_V2(model_layer_dict1)
    layer_name2, index2 = neuron_to_cover_V2(model_layer_dict2)

    print(layer_name1)
    print(layer_name2)
    
    # tf.Variable로 변환하여 그래디언트 추적을 가능하게 함
    gen_img = tf.Variable(gen_img_numpy)

    # 경사 상승법을 통해 이미지 수정
    for iters in range(args.grad_iterations):
        # CHANGED: K.function 대신 tf.GradientTape를 사용하여 그래디언트 계산
        with tf.GradientTape() as tape:
            tape.watch(gen_img)
            
            # 모델 예측 (training=False는 Dropout 등을 비활성화하는 추론 모드를 의미)
            pred1_tensor = model1(gen_img, training=False)
            pred2_tensor = model2(gen_img, training=False)

            # 공동 손실 함수 구성
            if args.target_model == 0:
                loss1 = -args.weight_diff * tf.reduce_mean(pred1_tensor[..., orig_label])
                loss2 = tf.reduce_mean(pred2_tensor[..., orig_label])
            else: # args.target_model == 1
                loss1 = tf.reduce_mean(pred1_tensor[..., orig_label])
                loss2 = -args.weight_diff * tf.reduce_mean(pred2_tensor[..., orig_label])
            
            # 뉴런 커버리지 손실 추가
            loss1_neuron = tf.reduce_mean(Model(model1.input, model1.get_layer(layer_name1).output)(gen_img)[..., index1])
            loss2_neuron = tf.reduce_mean(Model(model2.input, model2.get_layer(layer_name2).output)(gen_img)[..., index2])
            layer_output = (loss1 + loss2) + args.weight_nc * (loss1_neuron + loss2_neuron)

            # 최종 손실 함수
            final_loss = tf.reduce_mean(layer_output)

        # 그래디언트 계산
        grads_value = tape.gradient(final_loss, gen_img)
        
        # 그래디언트에 제약 조건 적용
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point, args.occlusion_size)
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)

        # 이미지 업데이트
        gen_img.assign_add(grads_value * args.step)
        
        # 수정된 이미지로 다시 예측
        pred1_gen = model1.predict(gen_img.numpy(), verbose=0)
        pred2_gen = model2.predict(gen_img.numpy(), verbose=0)
        label1_gen, label2_gen = np.argmax(pred1_gen[0]), np.argmax(pred2_gen[0])

        if label1_gen != label2_gen:
            print(bcolors.OKGREEN + 'Difference found at iteration {0}: {1}, {2}'.format(
                iters + 1, cifar10_labels[label1_gen], cifar10_labels[label2_gen]) + bcolors.ENDC)

            update_coverage(gen_img.numpy(), model1, model_layer_dict1, args.threshold)
            update_coverage(gen_img.numpy(), model2, model_layer_dict2, args.threshold)
            
            # 후처리 및 결과 저장
            gen_img_deprocessed = deprocess_image(gen_img.numpy())
            orig_img_deprocessed = deprocess_image(orig_img)
            
            save_name = '{0}_{1}_{2}_{3}'.format(
                save_name_prefix, args.transformation, cifar10_labels[label1_gen], cifar10_labels[label2_gen])
            
            Image.fromarray(gen_img_deprocessed.astype('uint8')).save('./generated_inputs/{0}.png'.format(save_name))
            Image.fromarray(orig_img_deprocessed.astype('uint8')).save('./generated_inputs/{0}_orig.png'.format(save_name))
            break