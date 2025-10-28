# PyMC Examples 초보자 가이드

이 문서는 PyMC를 활용한 Bayesian 추론과 Gaussian Process 모델링에 대한 초보자용 종합 가이드입니다. 이 프로젝트를 처음 접하시는 분들을 위해 핵심 개념을 단계적으로 설명하고, 실습 예제를 통해 학습할 수 있도록 구성했습니다.

## 📖 목차

1. [프로젝트 소개](#프로젝트-소개)
2. [핵심 개념 이해하기](#핵심-개념-이해하기)
3. [프로젝트 구조](#프로젝트-구조)
4. [예제별 상세 설명](#예제별-상세-설명)
5. [학습 경로 제안](#학습-경로-제안)
6. [참고 자료](#참고-자료)

---

## 프로젝트 소개

### 🎯 프로젝트의 목적

이 프로젝트는 **Bayesian Model Calibration**을 통해 컴퓨터 시뮬레이션과 실제 관측 데이터를 결합하여 물리 시스템의 미지 파라미터를 추정하는 방법을 보여줍니다. 특히 다음과 같은 상황에 유용합니다:

- **실험 비용이 높거나 제약이 있는 경우**: 컴퓨터 시뮬레이션으로 얻은 데이터를 활용
- **모델의 불완전성을 고려해야 하는 경우**: 시뮬레이션과 실제 현상 사이의 차이를 명시적으로 모델링
- **불확실성을 정량화해야 하는 경우**: 단순한 점 추정값이 아닌 확률 분포로 결과를 제공

### 🔬 실제 적용 예시

이 프로젝트의 예제는 건축물의 에너지 소비량을 다룹니다:

- **입력 변수**: 외기온도, 상대습도 등 (실험 조건)
- **출력 변수**: 에너지 사용량 (관측 결과)
- **추정 대상 파라미터**: 기기밀도, 조명밀도, COP 등 (미지수)

단, 이러한 방법론은 다른 도메인(기후 모델링, 공학 설계, 약물 발견 등)에도 적용 가능합니다.

---

## 핵심 개념 이해하기

### 1. Bayesian 추론 (Bayesian Inference)이란?

Bayesian 추론은 데이터를 관측한 후 우리가 모르는 파라미터에 대한 **확률 분포**를 업데이트하는 방법론입니다.

#### 전통적 접근법 vs Bayesian 접근법

**전통적 접근법 (빈도주의)**:
- 파라미터는 "고정된 값"으로 간주
- 관측 데이터를 통해 하나의 점 추정값을 구함 (예: 평균값)
- 예: "기기밀도는 15.3 W/m²입니다"

**Bayesian 접근법**:
- 파라미터도 "확률 변수"로 간주
- 관측 데이터를 통해 파라미터의 확률 분포를 구함
- 예: "기기밀도는 평균 15.3, 표준편차 1.2인 정규분포를 따릅니다"

#### Bayesian 업데이트 공식

Bayesian 추론의 핵심은 **Bayes' Theorem**입니다:

$$P(\boldsymbol{\theta} \mid \mathbf{y}) = \frac{P(\mathbf{y} \mid \boldsymbol{\theta}) \cdot P(\boldsymbol{\theta})}{P(\mathbf{y})}$$

여기서:
- $P(\boldsymbol{\theta} \mid \mathbf{y})$: **사후 분포 (Posterior)** - 데이터를 관측한 후의 파라미터 분포
- $P(\mathbf{y} \mid \boldsymbol{\theta})$: **가능도 (Likelihood)** - 주어진 파라미터에서 데이터가 관측될 확률
- $P(\boldsymbol{\theta})$: **사전 분포 (Prior)** - 데이터를 보기 전 우리가 가지고 있던 파라미터에 대한 믿음
- $P(\mathbf{y})$: **정규화 상수 (Normalizing Constant)** - 분포의 합이 1이 되도록 만드는 상수

#### 실제 예시로 이해하기

예를 들어, 기기밀도 파라미터를 추정한다고 가정해봅시다:

1. **사전 분포 설정**: "기기밀도는 대략 10~20 W/m² 사이일 것 같다"
   - 물리적 지식이나 과거 경험을 바탕으로 설정
   
2. **데이터 관측**: 실제 건물에서 에너지 사용량 측정 데이터 수집

3. **사후 분포 계산**: 데이터를 보니 "기기밀도는 14~16 W/m² 사이일 가능성이 높다"
   - 사전 분포와 데이터가 결합되어 더 정확한 추정이 이루어짐

4. **불확실성 정량화**: "95% 신뢰구간은 13.5~16.8 W/m²입니다"
   - 단순히 15.1 W/m²라는 값만 주는 것이 아니라, 얼마나 확실한지도 함께 제공

### 2. Gaussian Process (GP)란?

Gaussian Process는 함수를 확률적으로 모델링하는 방법입니다. 복잡한 수학적 함수를 직접 정의하는 대신, "함수값들이 어떤 확률 분포를 따를 것인가"를 정의합니다.

#### 직관적 이해

함수 $f(x)$를 모델링한다고 생각해봅시다. 전통적인 방법은 명시적인 수식을 쓰는 것입니다:
- 예: $f(x) = ax^2 + bx + c$

GP는 다른 접근을 합니다:
- 예: "x가 가까울수록 함수값도 비슷할 것이다 (부드러운 함수)"
- 어떤 점에서의 함수값은 주변 점들의 함수값과 상관관계를 가집니다

#### 수학적 정의

Gaussian Process는 다음으로 정의됩니다:

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

여기서:
- $m(\mathbf{x})$: **평균 함수 (Mean Function)** - 함수의 평균적인 경향
- $k(\mathbf{x}, \mathbf{x}')$: **공분산 함수 (Covariance Function)** 또는 **커널 (Kernel)** - 두 점 사이의 유사도를 나타내는 함수

#### 커널 함수의 역할

커널 함수는 두 입력값 $\mathbf{x}$와 $\mathbf{x}'$가 얼마나 유사한지 나타냅니다. 가장 많이 사용되는 RBF (Radial Basis Function) 커널은 다음과 같습니다:

$$k(\mathbf{x}_i, \mathbf{x}_j) = \eta^2 \exp\left(-\frac{1}{2}\sum_{d=1}^{D}\frac{(x_{i,d} - x_{j,d})^2}{l_d^2}\right)$$

여기서:
- $\eta^2$: **Amplitude 파라미터** - 함수값의 전체적인 변동 폭을 조절
- $l_d$: **Length-scale 파라미터** - $d$번째 차원에서 입력 변화에 대한 함수 변화의 민감도를 조절
  - $l_d$가 크면: 해당 차원에서 입력이 많이 바뀌어도 함수값이 크게 변하지 않음 (부드러움)
  - $l_d$가 작으면: 해당 차원에서 입력이 조금만 바뀌어도 함수값이 크게 변함 (급변)

#### GP의 장점

1. **유연성**: 복잡한 비선형 함수도 모델링 가능
2. **불확실성 정량화**: 예측 시 불확실성도 함께 제공
3. **데이터 효율성**: 적은 수의 시뮬레이션 실행만으로도 함수를 학습

### 3. Model Calibration이란?

Model Calibration은 시뮬레이션 모델의 파라미터를 실제 관측 데이터에 맞추는 과정입니다.

#### 기본 아이디어

컴퓨터 시뮬레이션은 보통 다음과 같은 형태입니다:

$$y = \eta(\mathbf{x}, \boldsymbol{\theta})$$

여기서:
- $\mathbf{x}$: 입력 변수 (예: 온도, 습도)
- $\boldsymbol{\theta}$: 모델 파라미터 (예: 기기밀도) ← **이 값을 찾고 싶음**
- $y$: 시뮬레이션 출력 (예: 에너지 사용량)

실제 관측 데이터 $z$가 있을 때, $\boldsymbol{\theta}$를 조정하여 시뮬레이션 결과 $y$가 실제 관측값 $z$와 가장 잘 맞도록 합니다.

#### Kennedy and O'Hagan (KOH) 프레임워크

완벽한 모델을 가정하는 것보다, 더 현실적인 접근은 모델의 불완전성을 인정하는 것입니다. KOH 프레임워크는 다음과 같이 모델링합니다:

$$z = \rho \cdot \eta(\mathbf{x}, \boldsymbol{\theta}) + \delta(\mathbf{x}) + e$$

여기서:
- $\eta(\mathbf{x}, \boldsymbol{\theta})$: 컴퓨터 시뮬레이션 함수
- $\rho$: 시뮬레이션 출력의 스케일 조정 파라미터
- $\delta(\mathbf{x})$: **모델 불완전성 (Model Discrepancy)** 또는 **바이어스 텀** - 시뮬레이션이 실제를 완벽히 재현하지 못하는 부분
- $e$: 관측 오차 (랜덤 노이즈)

이 프레임워크의 핵심은:
- 시뮬레이션 자체도 GP로 모델링 (에뮬레이터)
- 모델 불완전성도 GP로 모델링
- 두 가지를 함께 추정하여 더 정확한 파라미터 추정 가능

### 4. MCMC (Markov Chain Monte Carlo)란?

MCMC는 복잡한 확률 분포에서 샘플을 생성하는 알고리즘입니다. Bayesian 추론에서 사후 분포 $P(\boldsymbol{\theta} \mid \mathbf{y})$가 복잡할 때, 이를 직접 계산할 수 없으므로 샘플링을 통해 근사합니다.

#### 왜 필요한가?

사후 분포를 계산하려면 정규화 상수 $P(\mathbf{y})$를 계산해야 하는데, 이것이 고차원 적분이라 매우 어렵습니다:

$$P(\mathbf{y}) = \int P(\mathbf{y} \mid \boldsymbol{\theta}) P(\boldsymbol{\theta}) d\boldsymbol{\theta}$$

MCMC는 이를 우회하여 직접 사후 분포에서 샘플을 생성합니다.

#### NUTS (No-U-Turn Sampler)

이 프로젝트에서 사용하는 NUTS는 Hamiltonian Monte Carlo의 개선 버전입니다:

- **자동 튜닝**: Step size와 integration time을 자동으로 조정
- **효율적인 탐색**: 기울기 정보를 활용하여 빠르게 수렴
- **수렴 안정성**: 다른 MCMC 방법보다 더 안정적인 결과

#### 수렴성 진단

MCMC 샘플링이 제대로 되었는지 확인하는 지표들:

1. **Trace Plot**: 파라미터 값이 시간에 따라 안정적으로 변동하는지 확인
2. **R-hat ($\hat{R}$)**: 여러 체인을 실행했을 때 체인 간 일치도 측정 (1.01 이하면 좋음)
3. **ESS (Effective Sample Size)**: 독립적인 샘플의 개수 (클수록 좋음)

---

## 프로젝트 구조

```
pymc-examples/
├── README.md                    # 프로젝트 설명서
├── BEGINNER_GUIDE.md            # 이 문서 (초보자 가이드)
├── pyproject.toml               # 프로젝트 설정 및 의존성
├── uv.lock                      # 의존성 버전 잠금 파일
├── examples/                    # 예제 노트북 폴더
│   ├── pymc-gp-toy_model.ipynb  # 기본 GP 예제 (단순화된 버전)
│   ├── pymc-koh-example.ipynb   # KOH 프레임워크 예제 (완전한 버전)
│   └── pymc-gp-example.ipynb    # 추가 GP 예제
└── dataset/                     # 데이터셋 폴더
    ├── datacomp_hourly.csv      # 컴퓨터 시뮬레이션 데이터 (시간별)
    ├── datacomp_monthly.csv     # 컴퓨터 시뮬레이션 데이터 (월별)
    ├── datafield_hourly.csv     # 실제 관측 데이터 (시간별)
    └── datafield_monthly.csv    # 실제 관측 데이터 (월별)
```

### 주요 파일 설명

#### 예제 노트북

1. **`pymc-gp-toy_model.ipynb`**: 
   - 가장 기본적인 예제
   - 모델 불완전성을 고려하지 않는 단순화된 버전
   - 완벽한 모델 가정: $z = \eta(\mathbf{x}, \boldsymbol{\theta}) + e$
   - **초보자는 여기서 시작하는 것을 권장합니다**

2. **`pymc-koh-example.ipynb`**: 
   - Kennedy and O'Hagan 프레임워크의 완전한 구현
   - 모델 불완전성을 명시적으로 모델링
   - 전체 공식: $z = \rho \cdot \eta(\mathbf{x}, \boldsymbol{\theta}) + \delta(\mathbf{x}) + e$
   - 기본 예제를 이해한 후 학습

3. **`pymc-gp-example.ipynb`**: 
   - 추가적인 GP 모델링 기법과 예제

#### 데이터셋

- **`datacomp_*.csv`**: 컴퓨터 시뮬레이션으로 생성된 데이터
  - `yc`: 에너지 사용량 (시뮬레이션 출력)
  - `xc1`, `xc2`: 입력 변수 (외기온도, 상대습도)
  - `tc1`, `tc2`, `tc3`: 시뮬레이션에 사용된 파라미터 값들

- **`datafield_*.csv`**: 실제 건물에서 측정된 데이터
  - `yf`: 에너지 사용량 (실제 측정값)
  - `xf1`, `xf2`: 입력 변수 (외기온도, 상대습도)
  - 파라미터 값들은 **알 수 없음** (추정 대상)

---

## 예제별 상세 설명

### 예제 1: 기본 GP 모델 (`pymc-gp-toy_model.ipynb`)

#### 모델 구조

이 예제는 가장 단순한 형태의 Bayesian Model Calibration을 다룹니다:

$$z = \eta(\mathbf{x}, \boldsymbol{\theta}) + e$$

여기서 $\eta$는 GP로 모델링됩니다. 컴퓨터 시뮬레이션 데이터와 실제 관측 데이터를 결합하여 미지 파라미터 $\boldsymbol{\theta}$를 추정합니다.

#### 데이터 모델링

**컴퓨터 시뮬레이션 데이터**:
$$
\begin{align}
f_{c,i} &\sim \mathcal{GP}(\mu_c, k_c([\mathbf{x}_{c,i}, \boldsymbol{\theta}_i])) \\
y_{c,i} &\sim \mathcal{N}(f_{c,i}, \sigma_c^2)
\end{align}
$$

**실제 관측 데이터**:
$$
\begin{align}
f_{p,i} &\sim \mathcal{GP}(\mu_p, k_p([\mathbf{x}_{f,i}, \boldsymbol{\theta}_{true}])) \\
z_i &\sim \mathcal{N}(f_{p,i}, \sigma^2)
\end{align}
$$

두 GP가 동일한 함수 $\eta$를 모델링하므로, 시뮬레이션 데이터에서 학습한 정보가 관측 데이터의 파라미터 추정에 활용됩니다.

#### 학습 포인트

1. **데이터 정규화**: 모든 입력 변수를 [0, 1] 범위로 변환하여 GP 모델링의 수치적 안정성 확보
2. **Marginal GP**: 중간 변수를 적분하여 계산 효율성 향상
3. **Length-scale 해석**: 각 차원의 length-scale이 해당 변수의 중요도를 나타냄

### 예제 2: KOH 프레임워크 (`pymc-koh-example.ipynb`)

#### 모델 구조

이 예제는 Kennedy and O'Hagan (2001)의 완전한 프레임워크를 구현합니다:

$$z = \rho \cdot \eta(\mathbf{x}, \boldsymbol{\theta}) + \delta(\mathbf{x}) + e$$

여기서:
- $\eta(\mathbf{x}, \boldsymbol{\theta})$: 시뮬레이션 함수 (GP로 모델링)
- $\delta(\mathbf{x})$: 모델 불완전성 함수 (별도의 GP로 모델링)
- $\rho$: 시뮬레이션 출력의 스케일 조정 파라미터

#### 데이터 모델링

**컴퓨터 시뮬레이션 데이터**:
$$
\begin{align}
f_{c,i} &\sim \mathcal{GP}_\eta(\mu_\eta, k_\eta([\mathbf{x}_{c,i}, \boldsymbol{\theta}_i])) \\
y_{c,i} &\sim \mathcal{N}(f_{c,i}, \sigma_c^2)
\end{align}
$$

**실제 관측 데이터**:
$$
\begin{align}
f_{\eta,i} &\sim \mathcal{GP}_\eta(\mu_\eta, k_\eta([\mathbf{x}_{f,i}, \boldsymbol{\theta}_{true}])) \\
f_{\delta,i} &\sim \mathcal{GP}_\delta(\mu_\delta, k_\delta(\mathbf{x}_{f,i})) \\
f_{p,i} &= \rho \cdot f_{\eta,i} + f_{\delta,i} \\
z_i &\sim \mathcal{N}(f_{p,i}, \sigma^2)
\end{align}
$$

#### 추가 학습 포인트

1. **모델 불완전성의 중요성**: 시뮬레이션이 완벽하지 않다는 것을 인정하고 명시적으로 모델링
2. **두 GP의 분리**: 시뮬레이션 함수와 바이어스 함수를 별도로 추정
3. **스케일 파라미터**: $\rho$를 통해 시뮬레이션 출력의 전역적 스케일 조정

---

## 학습 경로 제안

### 단계별 학습 계획

#### 1단계: 환경 설정 (약 30분)

1. `uv` 설치
2. 프로젝트 의존성 설치 (`uv sync`)
3. Jupyter Notebook 실행 확인
4. 데이터셋 파일 확인

**체크리스트**:
- [ ] `uv --version` 명령어가 정상 작동
- [ ] `uv sync` 완료 후 `.venv` 디렉토리 생성 확인
- [ ] Jupyter Notebook이 정상적으로 실행됨
- [ ] 데이터 파일들이 `dataset/` 폴더에 있음

#### 2단계: 개념 학습 (약 2-3시간)

이 문서의 [핵심 개념 이해하기](#핵심-개념-이해하기) 섹션을 읽고 다음을 이해합니다:

- Bayesian 추론의 기본 원리
- Gaussian Process의 직관적 이해
- Model Calibration의 목적과 방법

**추가 자료**:
- PyMC 공식 튜토리얼 (기본 개념)
- Gaussian Process에 대한 간단한 논문이나 블로그 포스트

#### 3단계: 기본 예제 실행 (약 2-3시간)

`pymc-gp-toy_model.ipynb` 노트북을 처음부터 끝까지 실행합니다:

1. 각 셀을 실행하고 출력 결과를 확인
2. 코드의 각 부분이 무엇을 하는지 이해
3. 결과 그래프를 해석
4. 파라미터 추정 결과의 의미 파악

**실습 활동**:
- Length-scale 파라미터의 값을 바꿔보고 결과가 어떻게 변하는지 관찰
- 사전 분포를 변경하여 사후 분포에 미치는 영향 확인
- MCMC 샘플 수를 늘려보고 수렴성의 변화 확인

#### 4단계: 고급 예제 학습 (약 3-4시간)

`pymc-koh-example.ipynb` 노트북을 실행하고 기본 예제와 비교합니다:

**비교 포인트**:
- 두 모델의 수식 차이점
- 모델 불완전성 함수 $\delta(\mathbf{x})$의 역할
- 스케일 파라미터 $\rho$의 의미
- 어떤 경우에 KOH 프레임워크가 필요한가?

#### 5단계: 자신만의 분석 (선택사항)

다른 데이터셋이나 파라미터로 실험해봅니다:

1. 다른 시간 간격 데이터 사용 (hourly → monthly)
2. 사전 분포 범위 변경
3. 다른 커널 함수 시도
4. 자신의 연구 문제에 적용

---

## 참고 자료

### 공식 문서

- [PyMC 공식 문서](https://www.pymc.io/)
  - 튜토리얼과 예제가 매우 잘 정리되어 있음
  - Gaussian Process 섹션 특히 추천

- [ArviZ 문서](https://python.arviz.org/)
  - Bayesian 분석 결과 시각화 및 진단 도구
  - Trace plot, posterior plot 등 생성 방법

- [uv 공식 문서](https://docs.astral.sh/uv/)
  - Python 패키지 관리 도구 사용법

### 학술 자료

1. **Kennedy, M. C., & O'Hagan, A. (2001)**. Bayesian calibration of computer models. *Journal of the Royal Statistical Society: Series B*, 63(3), 425-464.
   - 이 프로젝트의 이론적 기반이 되는 논문
   - 수학적 배경을 이해하려면 필독

2. **Rasmussen, C. E., & Williams, C. K. (2006)**. *Gaussian processes for machine learning*. MIT press.
   - Gaussian Process에 대한 가장 포괄적인 교과서
   - 무료 온라인 버전: http://www.gaussianprocess.org/gpml/

3. **McElreath, R. (2020)**. *Statistical rethinking: A Bayesian course with examples in R and Stan*. CRC press.
   - Bayesian 추론에 대한 직관적이고 실용적인 설명
   - 이론보다 실용성에 중점

### 온라인 강의 및 튜토리얼

- PyMC의 공식 예제 갤러리
- Gaussian Process에 대한 YouTube 강의 영상
- Bayesian Statistics 관련 MOOC 강좌

---

## 자주 묻는 질문 (FAQ)

### Q1: Python이나 통계에 대한 선수 지식이 필요한가요?

**A**: 기본적인 Python 프로그래밍 능력과 확률/통계의 기본 개념을 알고 있으면 도움이 됩니다. 그러나 이 가이드를 따라가면서 필요한 개념을 설명하고 있으므로, 꾸준히 학습한다면 초보자도 이해할 수 있습니다.

### Q2: MCMC 샘플링이 너무 오래 걸립니다. 어떻게 해야 하나요?

**A**: 다음을 시도해보세요:
- `draws`와 `tune` 파라미터를 줄여서 테스트 (예: `draws=500, tune=500`)
- 데이터 크기를 줄여서 실험 (예: 일부 데이터만 사용)
- 더 빠른 백엔드 사용 확인 (NumPyro가 기본적으로 사용됨)

### Q3: 수렴이 안 되는 것 같습니다. 어떻게 진단하나요?

**A**: 
- Trace plot을 확인: 파라미터가 안정적으로 변동하는가?
- R-hat 값 확인: 1.01보다 큰 파라미터가 있는가?
- `tune` 값을 늘려보기
- 사전 분포 범위 확인: 너무 넓거나 좁지 않은가?
- 데이터 정규화 확인: 모든 변수가 올바르게 정규화되었는가?

### Q4: 어떤 예제부터 시작해야 하나요?

**A**: 반드시 `pymc-gp-toy_model.ipynb`부터 시작하세요. 이것이 가장 기본적이고 이해하기 쉽습니다. 이 예제를 완전히 이해한 후에 `pymc-koh-example.ipynb`로 진행하는 것을 권장합니다.

### Q5: 결과를 어떻게 해석하나요?

**A**: 
- **사후 분포**: 파라미터의 가능한 값과 그 확률
- **신뢰구간**: 파라미터가 특정 범위에 있을 가능성 (예: 95% 신뢰구간)
- **Length-scale**: 각 변수의 중요도와 민감도
- **Trace plot**: 샘플링이 제대로 되었는지 확인

---

## 마무리

이 가이드가 PyMC를 활용한 Bayesian 추론과 Gaussian Process 모델링을 학습하는 데 도움이 되기를 바랍니다. 개념이 어렵게 느껴질 수 있지만, 노트북을 직접 실행하고 결과를 해석하면서 점진적으로 이해해 나가시기 바랍니다.

**학습의 핵심 원칙**:
1. **실습 위주**: 코드를 실행해보고 결과를 직접 확인
2. **단계적 이해**: 한 번에 모든 것을 이해하려 하지 말고, 단계별로 학습
3. **질문하기**: 막히는 부분이 있으면 관련 자료를 찾거나 질문
4. **재현하기**: 예제를 단순히 따라하는 것이 아니라, 각 단계의 의미를 이해

행운을 빕니다!

