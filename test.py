import os
import argparse
import torch
from torchvision.utils import make_grid, save_image

from models import ColorTrans
from dataloader import data_loader

def main(args):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load testing dataset
    test_loader, _ = data_loader(root=args.root, batch_size=args.batch_size, shuffle=True, 
                                            img_size=args.img_size, mode='test')

    if args.epochs is not None:
        weight_name = 'checkpoint_{epoch}_epoch.pkl'.format(epoch=args.epochs)
    else:
        weight_name = 'checkpoint_1_epoch.pkl'
        
    checkpoint = torch.load(os.path.join(args.weight_dir, weight_name))
    colornet = ColorTrans().to(device)
    colornet.load_state_dict(checkpoint['colornet_state_dict'])
    colornet.eval()

    for _, (black_imgs, _) in enumerate(test_loader):
        black_imgs = black_imgs.type(torch.cuda.FloatTensor)
        # generate color image 
        generated_imgs= colornet(black_imgs)
    
    if os.path.exists(args.result_dir) is False:
        os.makedirs(args.result_dir)
        
    # For example, img_name = random_55.png
    if args.epochs is None:
        args.epochs = 'latest'
    img_name = 'generated_colorimg_{epoch}.png'.format(epoch=args.epochs)
    img_path = os.path.join(args.result_dir, img_name)

    img_grid = make_grid(generated_imgs, nrow=5, normalize=True, scale_each=True)
    save_image(img_grid, img_path, nrow=5, normalize=True, scale_each=True)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../input/ncdataset', 
                        help='Data location')
    parser.add_argument('--result_dir', type=str, default='./',
                        help='Ouput images location')
    parser.add_argument('--weight_dir', type=str, default='../input/colortrans-weight',
                        help='Trained weight location of generator. pkl file location')
    parser.add_argument('--epochs', type=int, default=349,
                        help='Epoch that you want to see the result. If it is None, the most recent epoch')
    parser.add_argument('--batch_size', type=int, default=25, 
                        help='Batch size for generator.')
    parser.add_argument('--img_size', type=int, default=224, 
                        help='Specific image size')

    args = parser.parse_args([])
    main(args)