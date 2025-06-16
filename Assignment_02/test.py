from tensorflow.keras.models import load_model

# --- 사용자 수정 필요 ---
# 확인하고 싶은 모델 파일의 경로를 정확하게 입력하세요.
MODEL_FILE_PATH = 'resnet50_cifar10_seed10.h5' 
# -------------------------

try:
    # 파일에서 모델 불러오기
    model = load_model(MODEL_FILE_PATH)
    
    # 모델 구조 요약 출력
    print(f"--- Summary for {MODEL_FILE_PATH} ---")
    model.summary()

except FileNotFoundError:
    print(f"오류: '{MODEL_FILE_PATH}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"모델을 불러오는 중 오류가 발생했습니다: {e}")