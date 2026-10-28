# ES2I: Ecg Signals inTo Images
This repository contains the official implementation of the paper "Effectiveness of Transforming ECG Signal into Image with Image-Based Pre-trained Model".

The paper was presented as an oral presentation at the Korea Software Congress 2024 (KSC 2024), a major academic conference in computer science and software engineering organized by the Korean Institute of Information Scientists and Engineers (KIISE). It was honored with the Outstanding Presentation Paper Award (우수발표논문상).

Paper: [Effectiveness of Transforming ECG Signal into Image with Image-Based Pre-trained Model](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12041844)

## Environments
- Python: 3.12.3
- PyTorch: 2.12.1
- CUDA: 12.6
- cuDNN: 9.10.2
- GPU: NVIDIA RTX 3090

#### Requirements
- torch==2.12.1+cu126
- torchvision==0.27.1+cu126
- timm==1.0.27
- numpy==2.3.3
- pandas==2.3.3
- Pillow==11.3.0
- scikit-learn==1.7.2
- joblib==1.5.3
- tqdm==4.67.1
- wfdb==4.3.1
- neurokit2==0.2.13
- tensorboard==2.20.0

## Training
```python
!python main.py --signal-path {path for raw signal data}
```
