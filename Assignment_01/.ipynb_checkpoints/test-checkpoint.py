import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import configparser
import os 
import sys
sys.path.append(os.getcwd())  
from attacks import fgsm_targeted, fgsm_untargeted, pgd_targeted, pgd_untargeted  # Import attacks

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Config 파일 로드 함수
def load_config(config_path="config.ini"):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

# 데이터 로드 함수
def load_data(dataset_name, batch_size):
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_name == "cifar":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset: Choose 'mnist' or 'cifar'")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# 학습 함수 정의 
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    for epoch in range(config_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader,
                            desc=f"Epoch {epoch + 1}/{config_epochs}")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass 및 손실 계산
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

# 정확도 평가 함수 
def evaluate_model_accuracy(model, test_loader, device):
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted_labels = torch.max(outputs.data.detach(), dim=1)
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    return accuracy

# adversarial attack 함수
def evaluate_attack_success(model,
                            test_loader,
                            attack_method_name,
                            epsilon=None,
                            k=None,
                            eps_step=None,
                            device="cpu"):
    
    model.eval()
    total_count = 0
    attack_success_count = 0  # 공격 성공한 샘플 수
    correct_after_attack = 0  # 공격 후에도 원래 라벨을 맞춘 경우 (정확도 확인용)

    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)

        # 공격 수행
        if attack_method_name == "targeted_fgsm":
            random_target_class = (labels + 1) % 10 if labels.max() < 10 else labels.clone() - 1
            adv_examples = fgsm_targeted(model=model,
                                         x=images,
                                         target=random_target_class,
                                         eps=epsilon)

        elif attack_method_name == "untargeted_fgsm":
            adv_examples = fgsm_untargeted(model=model,
                                           x=images,
                                           label=labels,
                                           eps=epsilon)

        elif attack_method_name == "targeted_pgd":
            random_target_class = (labels + 1) % 10 if labels.max() < 10 else labels.clone() - 1
            adv_examples = pgd_targeted(model=model,
                                        x=images,
                                        target=random_target_class,
                                        k=k,
                                        eps=epsilon,
                                        eps_step=eps_step)

        elif attack_method_name == "untargeted_pgd":
            adv_examples = pgd_untargeted(model=model,
                                          x=images,
                                          label=labels,
                                          k=k,
                                          eps=epsilon,
                                          eps_step=eps_step)

        else:
            raise ValueError(f"Unsupported attack method: {attack_method_name}")

        # 원본 이미지에 대한 모델 예측
        with torch.no_grad():
            outputs_original = model(images)
            _, predicted_original_labels = torch.max(outputs_original, dim=1)

        # 공격 후 모델 예측
        outputs_adv = model(adv_examples)
        _, predicted_adv_labels = torch.max(outputs_adv, dim=1)

        if "untargeted" not in attack_method_name:
            # Targeted attack의 경우 목표 타겟으로 변경된 비율을 공격 성공률로 설정
            attack_success_count += (predicted_adv_labels == random_target_class).sum().item()
        else:
            # Untargeted attack의 경우 원래 예측과 달라진 비율을 공격 성공률로 설정
            attack_success_count += (predicted_adv_labels != predicted_original_labels).sum().item()

        # 공격 이후에도 정답을 맞춘 경우 확인 (공격 후 정확도)
        correct_after_attack += (predicted_adv_labels == labels).sum().item()
        total_count += labels.size(0)

    attack_success_rate = attack_success_count / total_count * 100
    accuracy_after_attack = correct_after_attack / total_count * 100

    print(f"Attack Success Rate ({attack_method_name}): {attack_success_rate:.2f}%")
    print(f"Accuracy After Attack: {accuracy_after_attack:.2f}%")

    
# 메인 실행 코드
if __name__ == "__main__":
    # Config 로드 및 설정값 읽기
    config_path = "config.ini"
    config = load_config(config_path)

    dataset_name = config["DEFAULT"]["Dataset"]
    attack_method_name = config["DEFAULT"]["AttackMethod"]
    
    epsilon = float(config["DEFAULT"]["Epsilon"])
    k_steps_pgd_attack = int(config["DEFAULT"].get("K", 10))
    eps_step_pgd_attack = float(config["DEFAULT"].get("EpsStep", 0.01))
    
    batch_size = int(config["DEFAULT"]["BatchSize"])
    learning_rate = float(config["DEFAULT"]["LearningRate"])
    config_epochs = int(config["DEFAULT"]["Epochs"])

    # 하드웨어 설정 (GPU 사용 가능 시 GPU로 설정)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    if dataset_name == 'mnist': 
        model = MNISTModel()
    elif dataset_name == 'cifar':
        model = models.resnet18(pretrained=True)  # Pytorch ResNet18 모델 사용 
        model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 클래스 수에 맞게 출력 조정

    model.to(device)
    train_loader, test_loader = load_data(dataset_name, batch_size)

    # 학습 수행 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Training model...")
    train_model(model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device)

    print("Evaluating Model...")
    acc = evaluate_model_accuracy(model, 
                            test_loader, 
                            device)
    print(f"Accuracy Before Attack : {acc}")
    
    evaluate_attack_success(model=model,
                            test_loader=test_loader,
                            attack_method_name=attack_method_name,
                            epsilon=epsilon,
                            k=k_steps_pgd_attack,
                            eps_step=eps_step_pgd_attack,
                            device=device)
    
