# Box Logo Detection

Box 이미지에서 Dolby/HDMI 로고를 검출하는 4가지 알고리즘(SIFT, ORB, Template Matching, Canny Edge TM) 비교 연구 프로젝트입니다.

## 📊 결과 보고서

**[실험 결과 보고서 보기](https://bsu1487-star.github.io/box_detection/report.html)**

## 주요 결과

- **Template Matching과 Canny Edge TM**만 유사 로고를 정확히 구분
- **SIFT/ORB**는 11개 템플릿 전부를 FOUND로 판정 (구분 불가)
- 색상 변화 테스트에서 TM/Canny 모두 검출 성공

## 프로젝트 구조

```
box_detect/
├── detect_logos.py      # 메인 검출 스크립트
├── report.html          # 결과 보고서 (GitHub Pages)
├── box_raw.png          # 검출 대상 이미지
├── *.png                # 템플릿 이미지들
├── result_*.png         # 검출 결과 시각화
└── requirements.txt     # Python 패키지
```

## 실행 방법

```bash
# 가상환경 생성 및 활성화
python -m venv .myenv
.myenv\Scripts\activate  # Windows
source .myenv/bin/activate  # Linux/Mac

# 패키지 설치
pip install -r requirements.txt

# 검출 실행
python detect_logos.py
```

## 기술 스택

- Python 3.13
- OpenCV 4.13.0 (opencv-contrib-python)
- NumPy 2.4.2

## 알고리즘 비교

| 방법 | 유사 로고 구분 | 색상 변화 대응 | 정확도 |
|------|----------------|----------------|--------|
| SIFT | ❌ | 보통 | 11/11 FOUND (과검출) |
| ORB | ❌ | 보통 | 11/11 FOUND (과검출) |
| TM | ✅ | 보통 (0.93→0.86) | 4/11 FOUND (정확) |
| Canny TM | ✅ | 강함 (엣지 기반) | 4/11 FOUND (정확) |

자세한 내용은 [결과 보고서](https://bsu1487-star.github.io/box_detection/report.html)를 참조하세요.
