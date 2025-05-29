# ES2I: Ecg Signal inTo Image
This is an official guide for transforming ECG signals into simple images to image-based pre-trained models.

paper: https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12041844

### Requirements
* python 3.10.8
* numpy 1.26.4
* pandas 2.2.3
* wfdb 4.3.0
* torch 2.6.0+cu124
* torchvision 0.14.1
* timm 1.0.15

### Training
```python
!python main.py --signal-path {path for raw signal data}

```
