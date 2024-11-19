import wfdb
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader

# load PTB-XL ECG signal data
def load_X(df, sampling_rate, path) :
    if sampling_rate == 100 :
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    elif sampling_rate == 500 :
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

# image shape : offset X, grid X
def ecg_signal_to_image_offsetX_gridX(data, idx, lead_list) :
    # extract data to transforming signal into image
    tmp = []
    for lead in lead_list:
        lead_idx = lead -1
        tmp.append(data[idx,:,lead_idx])
    tmp = np.array(tmp).transpose(1,0)
    
    # visualization
    color_list = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'pink', 'teal', 'black']
    fig = plt.figure(figsize= (15,5))
    for i in range(len(lead_list)) :
        color = color_list[i]
        plt.plot(tmp[:, i], color= color)
        plt.axis('off')
    plt.xlim(-2, 5002)
    
    # save to temporary memory
    buf = BytesIO()
    plt.savefig(buf, format= 'png',  bbox_inches= 'tight', pad_inches= 0)
    plt.close(fig)
    
    # load image in temporary memory
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img_resized = img.resize((224,224))
    
    # return in numpy format
    return np.array(img_resized)

# image shape : offset X, grid O
def ecg_signal_to_image_offsetX_gridO(data, idx, lead_list) :
    # extract data to transforming signal into image
    tmp = []
    for lead in lead_list:
        lead_idx = lead -1
        tmp.append(data[idx,:,lead_idx])
    tmp = np.array(tmp).transpose(1,0)
    
    # visualization
    color_list = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'pink', 'teal', 'black']
    fig = plt.figure(figsize= (15,5))
    for i in range(len(lead_list)):
        color = color_list[i]
        plt.plot(tmp[:, i], color= color)
    plt.xlim(-2, 5002)
    
    # grid setting
    ymin, ymax = plt.gca().get_ylim()
    ytick_interval = (ymax - ymin) / 40
    plt.gca().set_yticks(np.arange(ymin, ymax, ytick_interval))
    plt.gca().set_xticks(np.arange(0, 5000, 100))
    plt.grid(True)
    plt.tick_params(left= False, right= False, labelleft= False, labelbottom= False, bottom= False)
    
    # save to temporary memory
    buf = BytesIO()
    plt.savefig(buf, format= 'png',  bbox_inches= 'tight', pad_inches= 0)
    plt.close(fig)
    
    # load image in temporary memory
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img_resized = img.resize((224,224))
    
    # return in numpy format
    return np.array(img_resized)

# image shape : offset O, grid X
def ecg_signal_to_image_offsetO_gridX(data, idx, lead_list) :
    # extract data to transforming signal into image
    tmp = []
    for lead in lead_list:
        lead_idx = lead -1
        tmp.append(data[idx,:,lead_idx])
    tmp = np.array(tmp).transpose(1,0)
    
    # visualization
    color_list = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'pink', 'teal', 'black']
    offsets = [0, 2, 4]
    fig = plt.figure(figsize= (15,5))
    for i in range(3):
        color = color_list[i]
        plt.plot(tmp[:, i] + offsets[i], color= color)
        plt.axis('off')
    plt.xlim(-2, 5002)
    plt.subplots_adjust(hspace= 0)
    
    # save to temporary memory
    buf = BytesIO()
    plt.savefig(buf, format= 'png',  bbox_inches= 'tight', pad_inches= 0)
    plt.close(fig)
    
    # load image in temporary memory
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img_resized = img.resize((224,224))
    
    # return in numpy format
    return np.array(img_resized)

# image shape : offset O, grid O
def ecg_signal_to_image_offsetO_gridO(data, idx, lead_list) :
    # extract data to transforming signal into image
    tmp = []
    for lead in lead_list:
        lead_idx = lead -1
        tmp.append(data[idx,:,lead_idx])
    tmp = np.array(tmp).transpose(1,0)

    # visualization
    color_list = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'pink', 'teal', 'black']
    offsets = [0, 2, 4]
    fig = plt.figure(figsize= (15,5))
    for i in range(3):
        color = color_list[i]
        plt.plot(tmp[:, i] + offsets[i], color= color)
    plt.xlim(-2, 5002)
    plt.subplots_adjust(hspace= 0)
    
    # grid setting
    ymin, ymax = plt.gca().get_ylim()
    ytick_interval = (ymax - ymin) / 40
    plt.gca().set_yticks(np.arange(ymin, ymax, ytick_interval))
    plt.gca().set_xticks(np.arange(0, 5000, 100))
    plt.grid(True)
    plt.tick_params(left= False, right= False, labelleft= False, labelbottom= False, bottom= False)

    # save to temporary memory
    buf = BytesIO()
    plt.savefig(buf, format= 'png',  bbox_inches= 'tight', pad_inches= 0)
    plt.close(fig)
    
    # load image in temporary memory
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img_resized = img.resize((224,224))
    
    # return in numpy format
    return np.array(img_resized)

# transforming ECG signal into image
def ecg_signal_to_image(signal_path, csv_path, split, lead_list, image_shape, sampling_rate) :
    # load ECG signal data
    data = pd.read_csv(f'{csv_path}ptbxl_super_class_{split}.csv')
    X_signal = load_X(data, sampling_rate= sampling_rate, path= signal_path)
    y = np.array([data.iloc[i,-5:].values.tolist() for i in range(len(data))])

    # transforming ECG signal into image one by one
    X_image = []
    for idx in range(X_signal.shape[0]) :
        if image_shape == 'offsetX_gridX' :
            img_arr = ecg_signal_to_image_offsetX_gridX(X_signal, idx, lead_list)
        elif image_shape == 'offsetX_gridO' :
            img_arr = ecg_signal_to_image_offsetX_gridO(X_signal, idx, lead_list)
        elif image_shape == 'offsetO_gridX' :
            if len(lead_list) != 3 :
                print('lead_list error: Only three leads are supported in the lead list.')
                break
            img_arr = ecg_signal_to_image_offsetO_gridX(X_signal, idx, lead_list)
        elif image_shape == 'offsetO_gridO' :
            if len(lead_list) != 3 :
                print('lead_list error: Only three leads are supported in the lead list.')
                break
            img_arr = ecg_signal_to_image_offsetO_gridO(X_signal, idx, lead_list)
        else :
            print('image_shape error: Only four specific image shapes are supported.')
        X_image.append(img_arr)
    return np.array(X_image).transpose(0,3,1,2), y

# save in numpy format
def make_numpy(lead_list, image_shape, signal_path= './data/PTB-XL/', csv_path= './data/', sampling_rate= 500) :
    leads = ','.join(map(str, lead_list))
    save_path = f'./data/lead{leads}_{image_shape}/'
    os.makedirs(save_path, exist_ok= True)
    
    # train data
    X_train, y_train = ecg_signal_to_image(signal_path= signal_path, csv_path= csv_path, split= 'train', lead_list= lead_list, image_shape= image_shape, sampling_rate= sampling_rate)
    np.save(f'{save_path}X_train.npy', X_train)
    np.save(f'{save_path}y_train.npy', y_train)
    
    # validation data
    X_valid, y_valid = ecg_signal_to_image(signal_path= signal_path, csv_path= csv_path, split= 'valid', lead_list= lead_list, image_shape= image_shape, sampling_rate= sampling_rate)
    np.save(f'{save_path}X_valid.npy', X_valid)
    np.save(f'{save_path}y_valid.npy', y_valid)
    
    # test data
    X_test, y_test = ecg_signal_to_image(signal_path= signal_path, csv_path= csv_path, split= 'test', lead_list= lead_list, image_shape= image_shape, sampling_rate= sampling_rate)
    np.save(f'{save_path}X_test.npy', X_test)
    np.save(f'{save_path}y_test.npy', y_test)

    print('ECG image successfully saved!')

class Dataset(Dataset) :
    def __init__(self, x, y, transform= None) :
        self.x = x
        self.y = y
        self.transform = transform
    def __len__(self) :
        return len(self.x)
    def __getitem__(self, idx) :
        x = self.x[idx]
        y = self.y[idx]

        if isinstance(x, np.ndarray) :
            x = np.transpose(x, (1, 2, 0))
            x = Image.fromarray(np.uint8(x * 255))

        if self.transform : x = self.transform(x)

        if not isinstance(x, torch.Tensor) : x = torch.tensor(x, dtype= torch.float32)
        else : x = x.clone().detach()
        y = torch.tensor(y, dtype= torch.float32)

        return x, y

def make_dataset(X, y, transform) -> Dataset :
    return Dataset(X, y, transform)

def make_dataloader(dataset, batch_size, shuffle) :
    return DataLoader(dataset, batch_size= batch_size, shuffle= shuffle)