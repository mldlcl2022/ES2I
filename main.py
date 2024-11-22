# library
import os
import argparse
import numpy as np
import time

import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import timm

from util.dataset import make_numpy
from util.dataset import make_dataset, make_dataloader
from util.evaluate import eval
from util.earlystopping import EarlyStopping

# parser
parser = argparse.ArgumentParser(description= '')
parser.add_argument('--seed',
                    default= 7,
                    type= int)
parser.add_argument('--device',
                    default= 'cuda',
                    type= str)
parser.add_argument('--leads',
                    default= [1, 2, 5],
                    type= int,
                    nargs= '+',
                    help= 'List of up to 3 ECG lead numbers to use (space-separated, e.g. 1 2 5).')
parser.add_argument('--image-shape',
                    default= 'offsetO_gridX',
                    type= str,
                    help= 'Shape of the ECG image. Options: "offsetX_gridX", "offsetX_gridO", "offsetO_gridX", "offsetO_gridO"')
parser.add_argument('--signal-path',
                    type= str,
                    help= 'Path to the signal data for the PTB-XL dataset.')
parser.add_argument('--batch-size',
                    default= 8,
                    type= int)
parser.add_argument('--model',
                    default= 'resnet18',
                    type= str,
                    help= 'Model name to use. Options: "resnet18", "resnet34", "resnet50", "vit"')
parser.add_argument('--fine-tuning',
                    default= 'linear_probing',
                    type= str,
                    help= 'Type of task. Options: "linear_probing", "fine_tuning"')
parser.add_argument('--early-stopping',
                    default= True,
                    type= bool,
                    help= 'Whether to use early stopping.')
parser.add_argument('--patience',
                    default= 30,
                    type= int,
                    help= 'patience for early stopping.')
parser.add_argument('--learning-rate',
                    default= 1e-3,
                    type= float)
parser.add_argument('--momentum',
                    default= 0.9,
                    type= float)
parser.add_argument('--epochs',
                    default= 30,
                    type= int)

# main
def main() :
    args = parser.parse_args()
    
    # seed setting
    seed_setting(args.seed)
    print('★ seed setting : Done ★')
    
    # device setting
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('\n★ device setting : Done ★')
    print('device :', device)
    
    # transforming ECG signal into image: make numpy file
    if f"lead{','.join(map(str, args.leads))}_{args.image_shape}" not in os.listdir('./data') :
        make_numpy(args.leads, args.image_shape, signal_path= args.signal_path)
        print('\n★ make numpy : Done ★')
    else : print('\n★ already Done ★')
    
    # load image data
    X_train = np.load(f"./data/lead{','.join(map(str, args.leads))}_{args.image_shape}/X_train.npy")
    y_train = np.load(f"./data/lead{','.join(map(str, args.leads))}_{args.image_shape}/y_train.npy")
    X_valid = np.load(f"./data/lead{','.join(map(str, args.leads))}_{args.image_shape}/X_valid.npy")
    y_valid = np.load(f"./data/lead{','.join(map(str, args.leads))}_{args.image_shape}/y_valid.npy")
    X_test = np.load(f"./data/lead{','.join(map(str, args.leads))}_{args.image_shape}/X_test.npy")
    y_test = np.load(f"./data/lead{','.join(map(str, args.leads))}_{args.image_shape}/y_test.npy")
    print('\n★ load image data(npy format) : Done ★')
    print('Train :', X_train.shape, y_train.shape)
    print('Valid :', X_valid.shape, y_valid.shape)
    print('Test  :', X_test.shape, y_test.shape)
    
    # image preprocessing setting
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    # make dataloader
    trainloader = make_dataloader(make_dataset(X_train, y_train, transform= train_transform), batch_size= args.batch_size, shuffle= True)
    validloader = make_dataloader(make_dataset(X_valid, y_valid, transform= test_transform), batch_size= args.batch_size, shuffle= True)
    testloader = make_dataloader(make_dataset(X_test, y_test, transform= test_transform), batch_size= args.batch_size, shuffle= True)
    print('\n★ make loader : Done ★')

    # model setting
    if args.model == 'resnet18' :
        model = models.resnet18(weights= models.ResNet18_Weights.IMAGENET1K_V1)
    elif args.model == 'resnet34' :
        model = models.resnet34(weights= models.ResNet34_Weights.IMAGENET1K_V1)
    elif args.model == 'resnet50' :
        model = models.resnet50(weights= models.ResNet50_Weights.IMAGENET1K_V1)
    elif args.model == 'vit' :
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
    
    if 'resnet' in args.model :
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, y_train.shape[1]), nn.Sigmoid())
    else :
        model.head = nn.Sequential(nn.Linear(model.head.in_features, y_train.shape[1]), nn.Sigmoid())

    # fine tuning
    if args.fine_tuning == 'linear_probing' :
        if 'resnet' in args.model :
            for l,p in model.named_parameters() :
                if 'fc' not in l : p.requires_grad = False
        elif 'vit' in args.model :
            for l,p in model.named_parameters() :
                if 'head' not in l : p.requires_grad = False
    
    print('\n★ model setting : Done ★')
    print(model.__class__.__name__) 
    cnt = sum(1 for l,p in model.named_parameters() if p.requires_grad)
    if cnt > 10 : print('full fine tuning setting : Done')
    else : print('linear probing setting : Done')

    # train setting
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr= args.learning_rate, momentum= args.momentum)
    
    # train
    print('\n★ start train! ★')
    
    model.to(device)
    epoch_losses = []
    valid_losses = []
    best_f1 = float('-inf')
    best_model_path = 'best_model.pth'

    # early stopping
    if args.early_stopping : 
        early_stopping = EarlyStopping(patience= args.patience, delta= 0.001)
        print('\n★ early stopping : Done ★\n')
    
    for epoch in range(args.epochs) :
        s = time.time()
        model.train()
        epoch_loss = 0.0
        
        for inputs, targets in trainloader :
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        epoch_losses.append(epoch_loss / len(trainloader))
        
        # evaluate
        model.eval()
        valid_loss = 0.0
        with torch.no_grad() :
            for inputs, targets in validloader :
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                valid_loss += loss.item()
        valid_losses.append(valid_loss / len(validloader))

        # early stopping
        if args.early_stopping :
            average_valid_loss = valid_loss / len(validloader)
            early_stopping(average_valid_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        validscore = eval(validloader, model, device)
        if best_f1 < validscore['f1'] :
            best_f1 = validscore['f1']
            torch.save(model.state_dict(), best_model_path)
        
        e = time.time()
        t = e-s
        if (epoch+1) % 5 == 0 :
            print('Epoch [{:3.0f}/{}] Time : {:.0f}m {:.0f}s - Loss : {:.4f}'.format(epoch+1, args.epochs, t//60, t%60, loss))
            print(f'Valid AUC : {validscore["auc"]:.4f}, F1 : {validscore["f1"]:.4f}')
        else : 
            print('Epoch [{:3.0f}/{}] Time : {:.0f}m {:.0f}s - Loss : {:.4f}'.format(epoch+1, args.epochs, t//60, t%60, loss))

    # evaluate testset
    model.load_state_dict(torch.load(best_model_path))
    testscore = eval(testloader, model, device)
    print(f"\nTest AUC : {testscore['auc']:.4f}, Test F1 : {testscore['f1']:.4f}\n")
    print('\n★ All process : Done ★')

def seed_setting(seed) :
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only= True)

if __name__ == '__main__' : main()
