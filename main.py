'''
Main file for Reconstruction
Attack using Mobilenetv3 as 
the Network for the Autoencoder
'''

from typing import Any
from utils import *
from models import *
from tqdm import tqdm
from torch.utils.data import DataLoader


def train_one_epoch(model, train_loader, device, optimizer, epoch):
    model.train()
  
    loss_fn = nn.MSELoss()
    epoch_losses = []

    for batch_idx, (img, target) in enumerate(tqdm(train_loader)):
        img, target = img.to(device), target.to(device)

    optimizer.zero_grad()
    out = model(img)

    loss = loss_fn(out, target)
    loss.backward()

    epoch_losses.append(loss.item())

    optimizer.step()

    avg_loss = sum(epoch_losses)/len(epoch_losses)
    print("\n\nAverage Training Loss for Epoch {}: {:.6f}".format(epoch, avg_loss)) 
    return avg_loss


def evaluate(model, device, test_loader):
    model.eval()

    loss_fn = nn.MSELoss()
    losses = []

    with torch.no_grad():
        for img, target in test_loader:
            img, target = img.to(device), target.to(device)
            out = model(img)
            loss = loss_fn(out, target)
            losses.append(loss.item())

    avg_loss = sum(losses)/len(losses)
    print("Average Evaluation Loss: {:.6f}".format( avg_loss ))
    return avg_loss


def train(args: Any, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader):
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    epochs = args.num_epochs
    params = [p for p in model.parameters() if p.requires_grad]
    model.to(device)
    
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('Invalid optimizer arg')
    
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        raise ValueError('Invalid scheduler arg')

    train_losses = []
    val_losses = []

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    for epoch in range(1, epochs+1):
        l1 = train_one_epoch(model, train_loader, device, optimizer, epoch)
        l2 = evaluate(model, device, test_loader)
        scheduler.step()

        train_losses.append(l1)
        val_losses.append(l2)
  
        if epoch % 5 == 0:
            torch.save(
                {
                    "epoch":epoch,
                    "model_state_dict":model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "training_losses":train_losses,
                    "val_losses":val_losses,
                    "scheduler_state_dict":scheduler.state_dict()
                },
                os.path.join(args.save_dir, 'ckpt_{}.ckpt'.format(epoch))
            )
  
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pth'))
    return train_losses, val_losses



if __name__ == '__main__':
    args = parse()
    if args.train:
        train_loader, val_loader = get_loaders(args)
        model = Autoencoder()
        train_losses, val_losses = train(args, model, train_loader, val_loader)
        plot_losses(args, train_losses, val_losses)
        visualize_samples(args, model)
    
    elif args.eval_pth or args.eval_ckpt:
        if not os.path.exists(args.model_path):
            raise ValueError('Model path is invalid!')

        model = Autoencoder()
        
        if args.eval_pth:
            state_dict = torch.load(args.eval_pth)
        else:
            ckpt = torch.load(args.model_path)
            state_dict = ckpt['model_state_dict']
        
        model.load_state_dict(state_dict)
        _, val_loader = get_loaders(args)

        if args.device == 'cuda':
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        else:
            device = torch.device('cpu')
        
        avg_loss = evaluate(model, device, val_loader)
    
    elif args.visualize:
        model = Autoencoder()
        if args.model_path:
            model.load_state_dict( torch.load(args.model_path) )
        visualize_samples(args, model)