# ES2I: Ecg Signal inTo Image
This is an official guide for transforming ECG signals into simple images to image-based pre-trained models.

paper: https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12041844

Paper: [KSC2024] Effectiveness of Transforming ECG Signal into Image with Image-Based Pre-trained Model
---
### Requirements
* python
* numpy
* pandas
* wfdb
* torch
* torchvision
* timm

### Training
```python
!python main.py --signal-path {path for raw signal data}

```
