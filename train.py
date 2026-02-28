import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from models import stackhourglass, basic
from dataset import get_dataloaders

def train(model, optimizer, imgL, imgR, disp_L, model_name, cuda_available):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if cuda_available:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
    else:
        disp_true = disp_L

    mask = (disp_true > 0)
    mask.detach_()

    optimizer.zero_grad()

    if model_name == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        # size_average is deprecated, using reduction='mean'
        loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], reduction='mean') + \
               0.7 * F.smooth_l1_loss(output2[mask], disp_true[mask], reduction='mean') + \
               F.smooth_l1_loss(output3[mask], disp_true[mask], reduction='mean')
    elif model_name == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask], reduction='mean')

    loss.backward()
    optimizer.step()

    return loss.item()

def test(model, imgL, imgR, disp_true, cuda_available):
    model.eval()

    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    
    if cuda_available:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL, imgR)

    pred_disp = output3.data.cpu()
    pred_disp = torch.squeeze(pred_disp, 1)

    disp_true = disp_true.cpu()
    mask = (disp_true > 0)
    loss = torch.mean(torch.abs(pred_disp[mask] - disp_true[mask]))

    return loss.item()

def main():
    parser = argparse.ArgumentParser(description='PSMNet Training')
    parser.add_argument('--datapath', default='/mnt/c/Users/prath/OneDrive/Desktop/Assignments/Thesis/Dataset/DrivingStereo', help='dataset path')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--model_name', default='stackhourglass', choices=['stackhourglass', 'basic'], help='model chosen')
    parser.add_argument('--loadmodel', default=None, help='load model path')
    parser.add_argument('--savemodel', default='./', help='save model path')
    args = parser.parse_args()

    maxdisp = 192
    cuda_available = torch.cuda.is_available()

    if args.model_name == 'stackhourglass':
        model = stackhourglass(maxdisp)
    elif args.model_name == 'basic':
        model = basic(maxdisp)
    else:
        raise ValueError('Model not recognized')

    if cuda_available:
        model = nn.DataParallel(model)
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    start_epoch = 1
    best_val_loss = float('inf')

    if args.loadmodel is not None and os.path.exists(args.loadmodel):
        print(f"Loading checkpoint from {args.loadmodel}")
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
        best_val_loss = state_dict.get('test_loss', float('inf'))
        print(f"Resuming training from epoch {start_epoch}")

    TrainImgLoader, TestImgLoader = get_dataloaders(args.datapath)

    start_full_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        total_train_loss = 0
        total_test_loss = 0

        # Training
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss = train(model, optimizer, imgL_crop, imgR_crop, disp_crop_L, args.model_name, cuda_available)
            print(f'Epoch {epoch}, Iter {batch_idx} training loss = {loss:.3f}, time = {(time.time() - start_time):.2f}')
            total_train_loss += loss
        print(f'epoch {epoch} total training loss = {total_train_loss / len(TrainImgLoader):.3f}')

        # Validation
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            test_loss = test(model, imgL, imgR, disp_L, cuda_available)
            print(f'Iter {batch_idx} validation loss = {test_loss:.3f}')
            total_test_loss += test_loss
        
        avg_test_loss = total_test_loss / len(TestImgLoader)
        print(f'epoch {epoch} total validation loss = {avg_test_loss:.3f}')

        # Saving best checkpoint
        if avg_test_loss < best_val_loss:
            best_val_loss = avg_test_loss
            savefilename = os.path.join(args.savemodel, 'best_checkpoint.tar')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': total_train_loss / len(TrainImgLoader),
                'test_loss': avg_test_loss,
            }, savefilename)
            print(f"Saved best model at epoch {epoch} with validation loss {avg_test_loss:.3f}")

    print('Full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))

if __name__ == '__main__':
    main()
