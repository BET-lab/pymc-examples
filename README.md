# PyMC Examples

PyMC를 활용한 Bayesian 추론과 Gaussian Process 모델링 예제 저장소입니다.

## 📋 목차

- [환경 설치](#환경-설치)
- [프로젝트 구조](#프로젝트-구조)
- [예제 소개](#예제-소개)
- [기술적 배경](#기술적-배경)
- [사용법](#사용법)
- [주요 기능](#주요-기능)
- [의존성](#의존성)

## 🚀 환경 설치

### 1. 저장소 클론
```bash
git clone <repository-url>
cd pymc-examples
```

### 2. Python 환경 설정
이 프로젝트는 `uv`를 사용하여 의존성을 관리합니다.

```bash
# uv 설치 (아직 설치되지 않은 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 의존성 설치
uv sync
```

### 3. 가상환경 활성화
```bash
# 가상환경 활성화
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate     # Windows
```

### 4. Jupyter Notebook 실행
```bash
# Jupyter Notebook 서버 시작
uv run jupyter notebook
```

## 📁 프로젝트 구조

```
pymc-examples/
├── README.md                    # 프로젝트 설명서
├── pyproject.toml              # 프로젝트 설정 및 의존성
├── examples/                   # 예제 노트북 폴더
│   └── pymc-gp-toy_model.ipynb # Gaussian Process 예제
└── dataset/                    # 데이터셋 폴더
    ├── datacomp_hourly.csv     # 컴퓨터 시뮬레이션 데이터
    └── datafield_hourly.csv    # 실제 관측 데이터
```

## 📊 예제 소개

### Gaussian Process Toy Model (`pymc-gp-toy_model.ipynb`)

이 예제는 PyMC를 사용한 Gaussian Process 기반 Bayesian 추론의 완전한 구현을 제공합니다.

#### 🎯 주요 목표
- 컴퓨터 시뮬레이션 데이터와 실제 관측 데이터를 결합
- Gaussian Process를 에뮬레이터로 활용
- 물리적 파라미터의 Bayesian 추정 및 불확실성 정량화

#### 📈 데이터 모델
실제 측정 데이터 $z$는 다음과 같이 모델링됩니다:

$$z = \eta(x, t) + e$$

여기서:
- $\eta(x, t)$: 실제 물리 프로세스 또는 컴퓨터 시뮬레이션 함수
- $x$: 입력 변수 (외기온도, 상대습도)
- $t$: 모델 파라미터 (기기밀도, 조명밀도, COP)
- $e$: 관측 오차 (랜덤 노이즈)

#### 🔬 데이터 타입

**1. 컴퓨터 시뮬레이션 데이터**
- 형태: `(yc, xc1, xc2, tc1, tc2, tc3)`
- `yc`: 에너지 사용량 (시뮬레이션 출력값)
- `xc1, xc2`: 외기온도, 상대습도
- `tc1, tc2, tc3`: 기기밀도, 조명밀도, COP (추정 대상)

**2. 실제 관측 데이터**
- 형태: `(yf, xf1, xf2)`
- `yf`: 에너지 사용량 (실제 측정값)
- `xf1, xf2`: 외기온도, 상대습도
- 물리적 파라미터는 미지수로 추정

## 🧮 기술적 배경

### Bayesian Model Calibration

이 예제는 Kennedy and O'Hagan (2001)의 Bayesian Model Calibration 방법론을 기반으로 합니다:

1. **완벽한 모델 가정**: 컴퓨터 모델이 실제 물리 프로세스를 완벽하게 재현한다고 가정
2. **Gaussian Process 에뮬레이션**: 복잡한 시뮬레이션 함수를 GP로 근사
3. **통합된 추론**: 시뮬레이션과 관측 데이터를 하나의 모델로 통합

### Gaussian Process 모델링

**커널 함수**: Exponential Quadratic (RBF) 커널 사용
$$k(x_i, x_j) = \eta^2 \exp\left(-\frac{1}{2}\sum_{d=1}^{5}\frac{(x_{i,d} - x_{j,d})^2}{l_d^2}\right)$$

**하이퍼파라미터**:
- `ls`: Length-scale 파라미터 (각 차원의 변동성)
- `eta`: Amplitude 파라미터 (함수 값의 스케일)
- `sigma`: 관측 노이즈의 표준편차

### MCMC 샘플링

**NUTS (No-U-Turn Sampler)** 사용:
- Hamiltonian Monte Carlo의 개선된 버전
- 자동 튜닝과 효율적인 수렴
- NumPyro 백엔드로 GPU 가속 지원

## 💻 사용법

### 1. 노트북 실행
```bash
# Jupyter Notebook에서 examples/pymc-gp-toy_model.ipynb 열기
uv run jupyter notebook examples/pymc-gp-toy_model.ipynb
```

### 2. 단계별 실행
노트북은 다음과 같은 단계로 구성되어 있습니다:

1. **라이브러리 임포트 및 환경 설정**
2. **데이터 임포트 및 탐색적 분석**
3. **데이터 시각화 및 탐색적 분석**
4. **Bayesian Inference 모델 정의**
5. **MCMC 샘플링 수행**
6. **사후 분포 분석 및 시각화**
7. **사후 예측 분포 (Posterior Predictive Distribution)**

### 3. 결과 해석

#### 파라미터 추정 결과
- `theta[0]`: 기기밀도 (Equipment Density)
- `theta[1]`: 조명밀도 (Lighting Density)  
- `theta[2]`: COP (Coefficient of Performance)

#### Length-scale 해석
- 큰 length-scale: 해당 차원에서 입력 변화에 대한 출력 변화가 작음
- 작은 length-scale: 해당 차원이 출력에 큰 영향을 미침

## ⭐ 주요 기능

### 1. 데이터 정규화
- Min-Max Scaling으로 모든 데이터를 [0, 1] 범위로 정규화
- 수치적 안정성과 수렴성 향상

### 2. Marginal Gaussian Process
- Latent 변수를 적분하여 제거
- 계산 효율성 향상
- 정규분포 노이즈 가정 하에서 해석적 marginal likelihood 계산

### 3. 통합된 데이터 모델링
- 시뮬레이션과 관측 데이터를 하나의 GP로 모델링
- 시뮬레이션 정보가 파라미터 추정에 활용

### 4. 사후 예측 분포
- 새로운 위치에서의 예측 및 불확실성 정량화
- 95% 신뢰구간과 예측 평균 제공

### 5. 시각화
- Trace plot으로 MCMC 수렴성 진단
- Pair plot으로 파라미터 간 상관관계 분석
- GP 예측 결과 시각화

## 📦 의존성

### 핵심 라이브러리
- **`pymc`**: Bayesian 통계 모델링과 MCMC 추론
- **`numpy`**: 수치 연산과 다차원 배열 처리
- **`arviz`**: Bayesian 분석 결과 요약 및 시각화
- **`pandas`**: 표 형식 데이터 처리
- **`matplotlib`**: 데이터 시각화
- **`pytensor`**: 텐서 연산 지원 (PyMC 백엔드)

### 추가 라이브러리
- **`jupyter`**: 노트북 환경
- **`uv`**: 패키지 관리

## 🔧 고급 설정

### GPU 가속 사용
```python
import jax
# GPU 확인
jax.devices()
```

### 다중 체인 샘플링 (권장)
```python
trace = pm.sample(
    draws=1000,
    tune=1000,
    chains=4,        # 4개 체인으로 병렬 샘플링
    cores=4,         # 4개 코어 사용
    target_accept=0.8
)
```

## 📚 참고 자료

- [PyMC 공식 문서](https://www.pymc.io/)
- [ArviZ 문서](https://python.arviz.org/)
- Kennedy, M. C., & O'Hagan, A. (2001). Bayesian calibration of computer models. Journal of the Royal Statistical Society: Series B, 63(3), 425-464.

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.