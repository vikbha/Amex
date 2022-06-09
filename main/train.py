from matplotlib.pyplot import yscale
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler
from dataloader import SpaceTitanic
from models import Vanilla
from dataloader import get_dataset, create_train_df, create_test_df, clean_df
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from config import get_property
from torch.utils.tensorboard import SummaryWriter
import time

writer = SummaryWriter(get_property('TENSORBOARD', 'log_dir') + '/Vanilla-'+ str(time.time()))

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')



def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # params    
    epochs = 100
    batch_size = 16
    lr = 5e-3
    shuffle = True
    
    # data
    train_dataset, test_dataset = get_dataset(device=device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    train_steps = train_dataset.__len__()
    
    # model parameters
    num_feature = train_dataset[0][0].size(dim=-1)
    num_output = 2
    hidden_dim = 128
    dropout = 0.25
    
    model = Vanilla(num_feature, num_output, hidden_dim, dropout).to(device)
    
    # optim
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    sche_start = 160
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=600)
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = 320
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.05)
    
    
    for e in tqdm(range(epochs)):
        print(f'Epoch : {e}')
        
        model.train()
        total_train_loss = 0.0
        train_running_acc = 0.0
        train_running_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            
            optimizer.zero_grad()
            y = y.type(torch.LongTensor).to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item()
            total_train_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            train_running_acc += (predicted == y).sum().item()
            if (i+1) % 50==0:
                writer.add_scalar('trainig_loss', train_running_loss / 50, e * train_steps + i)
                writer.add_scalar('accuracy', train_running_acc / 50, e * train_steps + i)
                train_running_acc = 0.0
                train_running_loss = 0.0
                
        model.eval()
        running_acc = 0.0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                y_pred = model(x)
                y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
                y = y.cpu().numpy()
                acc = accuracy_score(y_pred, y)
                running_acc += acc
            print(f'Epoch : {e}/{epochs} accuracy :{running_acc/test_loader.__len__():.3f}')
        
        if e > sche_start and e <= swa_start:
            scheduler.step()
        if e > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

    # submision generation
    df = clean_df(create_test_df())
    df['Transported'] = False
    print(df.head())
    sub_dataset = SpaceTitanic(df, device)
    sub_dataloader = DataLoader(sub_dataset, batch_size=1)
    res = []
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(sub_dataloader):
            y = model(x)
            y = torch.argmax(y, dim=1).squeeze().cpu().numpy()
            res.append(y)
    df = df.drop('Transported', axis=1)
    df['Transported'] = np.array([True if i==1 else False for i in res])
    df = df['Transported']
    df.to_csv(get_property('DATA','submission'))
            
    
                
    
    







if __name__=='__main__':
    train()
    
    # df = clean_df(create_df())
    # train_df = df.sample(frac=0.8, random_state=400)
    # test_df = df.drop(train_df.index)
    # X_train, y_train = train_df.iloc[:,:-1], train_df.iloc[:,-1]
    # X_test, y_test = train_df.iloc[:,:-1], train_df.iloc[:,-1]
    # clasifier = GradientBoostingClassifier(n_estimators=2, max_depth=4)
    # clasifier.fit(X_train, y_train)
    # print(f'Cleassifier accuracy training: {accuracy_score(clasifier.predict(X_train), y_train)}')
    # print(f'Cleassifier accuracy test: {accuracy_score(clasifier.predict(X_test), y_test)}')
    

