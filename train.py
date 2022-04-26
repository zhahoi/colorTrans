import argparse
from solver import Solver

def main(args):
    solver = Solver(root = args.root,
                    result_dir = args.result_dir,
                    img_size = args.img_size,
                    weight_dir = args.weight_dir,
                    batch_size = args.batch_size,
                    test_batch_size = args.test_batch_size,
                    lr = args.lr,
                    beta_1 = args.beta_1,
                    beta_2 = args.beta_2,
                    epochs = args.epochs,
                    num_epochs = args.num_epochs,
                    save_every = args.save_every,
                    load_weight = args.load_weight,
                    )  
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../input/sketch', help='Data location')
    parser.add_argument('--result_dir', type=str, default='test', help='Result images location')
    parser.add_argument('--img_size', type=int, default=224, help='Size of image for discriminator input.')
    parser.add_argument('--weight_dir', type=str, default='weight', help='Weight location')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=16, help='Batch size for generator.')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta_1', type=float, default=0.9, help='Beta1 for Adam')
    parser.add_argument('--beta_2', type=float, default=0.999, help='Beta2 for Adam')
    parser.add_argument('--save_every', type=int, default=100, help='How often do you want to see the result?')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epoch.')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epoch.')
    parser.add_argument('--load_weight', type=bool, default=False, help='Load weight or not')
                        
    args = parser.parse_args([])
    main(args=args)


