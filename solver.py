import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid, save_image

import os
import time
import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter

from dataloader import data_loader
from models import ColorTrans
from utils import init_torch_seeds

# for reproductionary
init_torch_seeds(seed=1234)

class Solver():
    def __init__(self, root='dataset/anime_faces', result_dir='result', img_size=224, weight_dir='weight', load_weight=False,
                 batch_size=8, test_batch_size=16, epochs=50, save_every=100, lr=0.0001, beta_1=0.5, beta_2=0.999, \
                     num_epochs=200, logdir=None):
        
        # cpu or gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load model
        self.colornet = ColorTrans()
        self.colornet.to(self.device)

        # loss
        self.smoothl1_loss = nn.SmoothL1Loss().to(self.device)

        # load training dataset
        self.train_loader, _ = data_loader(root=root, batch_size=batch_size, shuffle=True, 
                                                img_size=img_size, mode='train')

        self.test_loader, _ = data_loader(root=root, batch_size=test_batch_size, shuffle=False, 
                                                img_size=img_size, mode='train')

        # optimizer
        self.optimizer = optim.Adam(self.colornet.parameters(), lr=lr, betas=(beta_1, beta_2), eps=1e-8)

        self.writer = SummaryWriter(logdir)

        # Some hyperparameters
        self.result_dir = result_dir
        self.weight_dir = weight_dir
        self.load_weight = load_weight
        self.num_epochs = num_epochs
        self.epochs = epochs
        self.lr = lr
        self.img_size = img_size
        self.start_epoch = 0
        self.save_every = save_every

        # create dir
        if os.path.exists(self.result_dir) is False:
            os.makedirs(self.result_dir)

        if os.path.exists(self.weight_dir) is False:
            os.makedirs(self.weight_dir) 
        
    '''
    <show_model >
    Print model architectures
    '''
    def show_model(self):
        print('================================ Color Unet =====================================')
        print(self.colornet)
        print('==========================================================================================\n\n')
    
    '''
        < load_checkpoint >
        If you want to continue to train, load pretrained weight from checkpoint
    '''
    def load_checkpoint(self, checkpoint):
        print('Load model')
        self.colornet.load_state_dict(checkpoint['colornet_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        
    '''
        < save_checkpoint >
        Save checkpoint
    '''
    def save_checkpoint(self, state, file_name):
        print('saving check_point')
        torch.save(state, file_name)

    '''
        < train >
    '''
    def train(self):
        if self.load_weight is True:
            weight_name = 'checkpoint_{epoch}_epoch.pkl'.format(epoch=self.epochs)
            checkpoint = torch.load(os.path.join(self.weight_dir, weight_name))
            self.load_checkpoint(checkpoint)
        
        # self.show_model()

        print('====================     Training    Start... =====================')
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            start_time = time.time()
            # train color model
            self.colornet.train()

            for iters, (img_gray, img_rgb) in tqdm(enumerate(self.train_loader)):
                # load gray images and ab images
                img_gray = img_gray.type(torch.cuda.FloatTensor)
                img_rgb = img_rgb.type(torch.cuda.FloatTensor)

                # generate fake ab images
                gen_img_rgb = self.colornet(img_gray)

                # calculate loss
                loss = self.smoothl1_loss(gen_img_rgb, img_rgb)
                # save loss to tensorboardX
                self.writer.add_scalar("Loss/train", loss, epoch)

                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # modify learning rate 
                lr_ = self.lr * (1.0 - epoch / self.num_epochs) ** 0.9
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_

                log_file = open('log.txt', 'w')
                log_file.write(str(epoch))

                # Print error, save intermediate result image and weight
                if epoch and (iters % self.save_every) == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    print('[Elapsed : %s /Epoch : %d / Iters : %d] => loss : %f ' %(et, epoch, iters, loss.item()))

                    # Generate fake image
                    self.colornet.eval()
                    with torch.no_grad():
                        for iters, (img_gray, _) in enumerate(self.test_loader):
                            img_gray = img_gray.type(torch.cuda.FloatTensor)
                            gen_img_rgb = self.colornet(img_gray)
                            
                    sample_imgs = gen_img_rgb[:16]

                    img_name = 'generated_colorimg_{epoch}_{iters}.jpg'.format(epoch=epoch, iters=(iters % len(self.test_loader)))
                    img_path = os.path.join(self.result_dir, img_name)

                    img_grid = make_grid(sample_imgs, nrow=4, normalize=True, scale_each=True)
                    save_image(img_grid, img_path, nrow=4, normalize=True, scale_each=True)  

            # Save weight at the end of every epoch
            if (epoch % 10) == 0:
                # self.save_weight(epoch=epoch)
                checkpoint = {
                    "colornet_state_dict": self.colornet.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch
                    }
                path_checkpoint = os.path.join(self.weight_dir, "checkpoint_{}_epoch.pkl".format(epoch))
                self.save_checkpoint(checkpoint, path_checkpoint)  

        # write log to disk
        self.writer.close()    
        print("Training finished!")                