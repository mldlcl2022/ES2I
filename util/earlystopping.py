class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        """
        Args:
            patience (int): 개선이 없을 경우 몇 에포크까지 기다릴지 설정
            delta (float): 개선으로 간주되는 최소 변화 값
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, valid_loss):
        if self.best_loss is None:
            self.best_loss = valid_loss  # 첫 번째 손실 초기화
        elif valid_loss > self.best_loss - self.delta:
            self.counter += 1  # 개선되지 않으면 카운터 증가
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True  # 조기 종료 플래그 활성화
        else:
            self.best_loss = valid_loss  # 손실 개선 시 업데이트
            self.counter = 0  # 카운터 초기화
