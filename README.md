# PyTorchForAudio
## PyTorch Flow 
**1. Download Datasets**
```python
def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor(),
    )
    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data, validation_data
```
- datasets.MNIST(): torchvision.datasets에서 MNIST(손글씨) 데이터셋을 로드
- `root=data` : 지정한 data directory에 데이터셋을 저장 ( 해당 폴더가 없는 경우 자동으로 디렉토리 생성 )
- `download=true` : 지정한 directory에 해당 데이터가 없는 경우에만 다운로드를 수행 

### Train과 Validation DataSet 차이와 필요성 
- **훈련 데이터**: 모델을 학습시키는 데 사용. 모델은 이 데이터를 보고 패턴을 학습
- **검증 데이터**: 학습 중 모델의 성능을 평가하는 데 사용
- **필요성**: 두 데이터셋을 분리함으로써 모델이 새로운,보지 않은 데이터에 대해 얼마나 잘 일반화되는지 평가할 수 있습니다. 이는 모델의 실제 성능을 더 정확하게 추정하는 데 도움

### MNIST(ImageData)를 `ToTensor()`를 이용해 분리하는 이유와 방법

- **필요성**: PyTorch의 신경망은 텐서 형태의 입력을 요구, 텐서는 다차원 배열로, 효율적인 수치 연산이 가능
  - **변환 방법**: ToTensor() 변환은 다음과 같은 작업을 수행
    - a : 이미지 픽셀 값을 0-255에서 0-1 범위로 정규화
    - b : 이미지 차원을 (높이, 너비, 채널)에서 (채널, 높이, 너비)로 변경
    - c : numpy 배열이나 PIL Image를 PyTorch 텐서로 변환
    - 예를 들어, MNIST 이미지(28x28 픽셀, 흑백)는 [1, 28, 28] 형태의 텐서로 변환 
    여기서 1은 채널 수(흑백이므로 1), 28, 28은 이미지의 높이와 너비.

### `__name__ == "__main__"` 구문 뜻 

```python
if __name__ == "__main__":
    # download MNIST dataset
    train_data, _ = download_mnist_datasets()
    print("MNIST Datasets Downloaded")
```
- `__name__`: 현재 모듈의 이름을 나타내는 특별한 변수, 이 값은 모듈이 어떻게 실행되고 있는지에 따라 달라짐

- `__main__`: 이는 Python 인터프리터에 의해 직접 실행되는 스크립트를 나타내는 특별한 이름

`if __name__ == "__main__":` 구문의 의미:
1. 스크립트가 직접 실행될 때:
   - `__name__`의 값이 `"__main__"`으로 설정
   - 이 조건문 아래의 코드 블록이 실행
2. 스크립트가 다른 모듈에 의해 import될 때:
   - `__name__`의 값은 해당 모듈의 이름(파일명)이 됨
   - 이 조건문은 거짓이 되어 아래 코드 블록이 실행되지 않음.

따라서, 이 구문은 "이 스크립트가 메인 프로그램으로 직접 실행되고 있는가?"를 확인하는 것으로. 직접 실행될 때만 특정 코드(여기서는 MNIST 데이터셋 다운로드)를 실행하고, 모듈로 import될 때는 이 코드를 실행하지 않도록 하는 방법이다.
이 구조를 사용하면 같은 Python 파일을 독립적인 스크립트로도, 다른 프로그램에서 import하여 사용할 수 있는 모듈로도 활용할 수 있다. 

**2. Create data loader for the train set** 
```python
    train_data_loader =
        DataLoader(train_data, batch_size=BATCH_SIZE)
```

### DataLoader의 역할과 Batch_Size 의미 및 고려사항
- DataLoader의 역할:
   - 데이터셋을 반복 가능한 객체로 감싸는 래퍼(wrapper).
   - 데이터를 배치 단위로 로드하고, 셔플링, 병렬 로딩, 메모리 핀닝 등의 기능을 제공.
   - 학습 과정에서 데이터를 효율적으로 공급하는 역할.

- batch_size의 의미:
   - 한 번에 모델에 입력되는 데이터 샘플의 수.
   - 예: batch_size가 32면, 32개의 이미지가 한 번에 처리.

- batch_size 조정 방법:
   - `DataLoader` 생성 시 `batch_size` 매개변수로 지정.
   - 일반적으로 2의 거듭제곱 값을 사용 (예: 16, 32, 64, 128 등).

- batch_size 조정 이유:
   - 학습 속도와 메모리 사용량 조절
   - 일반화 성능 영향
   - 하드웨어 자원 활용 최적화

- 조정 시 고려 사항:

   a. 메모리 사용량:
      - 큰 batch_size는 더 많은 메모리를 사용.
      - GPU 메모리 한계를 고려.

   b. 학습 속도:
      - 큰 batch_size는 일반적으로 학습 속도를 높임.
      - 너무 크면 업데이트 횟수가 줄어 수렴이 느려질 수 있음.

   c. 일반화 성능:
      - 작은 batch_size는 때때로 더 나은 일반화 성능을 보임.
      - 큰 batch_size는 sharp minimizers에 빠질 수 있음.

   d. 하드웨어 활용:
      - GPU 사용률을 최대화할 수 있는 크기를 선택.

   e. 데이터셋 크기:
      - 전체 데이터셋 크기에 비해 너무 큰 batch_size는 피함.

   f. 모델 아키텍처:
      - 일부 모델은 특정 batch_size에서 더 잘 작동할 수 있음.

   g. 학습 알고리즘:
      - 일부 최적화 알고리즘은 특정 batch_size 범위에서 더 효과적.

> batch_size 조정은 실험과 경험을 통해 최적값을 찾는 것이 중요.
> 일반적으로 가능한 큰 batch_size를 사용하되, 메모리 한계와 일반화 성능을 고려하여 조정.

** 3. build model**
```python
class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data):
        flatten_data = self.flatten(input_data)
        logits = self.dense(flatten_data)
        predictions = self.softmax(logits)
        return  predictions
```
- `FeedForwardNet(nn.Module)` : nn.Module을 상속받아 PyTorch 신경망 모듈을 정의 
- `self.flateen = nn.Flatten()` 
  - 입력 데이터를 1차원으로 평탄화 
  - MNIST 이미지(28x28)를 784(28*28) 크기의 1차원 벡터로 변환 
- `self.dense` : 신경망 모델에서 fully connected layer의 sequence를 나타냄 
  - `nn.Sequential` : 여러 layer를 순차적으로 쌓을 수 있게 해주는 container , layer의 출력이 다음 layer의 입력이 됨
  - `nn.Linear(28*28, 256)` : 입력 28*28(784), 출력 256 입력 layer
  - `nn.ReLU()`활성화 함수로, 비선형성을 도입 -> f(x) = max(0, x)
  - `nn.Linear(256, 10)` : 입력 256, 출력 10개인 출력 layer 

### ReLU(Rectified Linear Unit) 함수의 역할
1. 기본 정의:
   - f(x) = max(0, x)
   - 입력이 0보다 작으면 0을 출력, 0 이상이면 입력을 그대로 출력
2. 비선형성 도입:
   - 신경망에 비선형성을 추가하여 복잡한 패턴을 학습할 수 있도록 함
   - 선형 변환의 연속만으로는 표현할 수 없는 함수들을 근사를 가능케 함
3. 기울기 소실 문제 완화:
   - 양수 입력에 대해 기울기가 항상 1이므로, 깊은 신경망에서도 기울기가 잘 전파
4. 희소성(Sparsity) 촉진:
   - 음수 입력을 0으로 만들어 네트워크의 희소성(sparsity)을 증가
   - 이는 일부 뉴런만 활성화되어 과적합을 줄이는 데 도움
5. 경사 하강법 최적화:
   - 양의 구간에서 기울기가 상수이므로 최적화가 안정적

-  `self.softmax = nn.LogSoftmax(dim=1` : 
  - 머신러닝 모델의 출력을 해석 가능한 확률로 변환하는 도구
  - 특히 분류 문제에서 각 클래스에 속할 확률을 계산하는 데 사용

- `def forward`
  - forward 구조의 신경망 정의 
  - 입력 → 평탄화 → 완전연결층 → 소프트맥스 의 순서로 데이터가 처리
  - 소프트맥스를 사용하여 최종 출력을 확률로 해석할 수 있게 만듬
