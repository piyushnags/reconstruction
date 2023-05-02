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


def train_one_epoch(
        model, train_loader, device, 
        optimizer, epoch
    ):
    '''
    Description:
        Helper function to train one epoch of the model
    
    Args:
        model: nn.Module object of the model
        train_loader: DataLoader object for the training data
        device: torch.device object for training on GPU/CPU
        optimizer: torch optimizer module used to update model weights
        epoch: Current epoch number for logging
    
    Return:
        None
    '''
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
    '''
    Description: 
        Helper function to evaluate model performance
        on validation/testing data
    
    Args:
        model: nn.Module object for the model
        device: torch.device object for running inference i.e., GPU/CPU
        test_loader: DataLoader object for loading validation/testing data
    
    Returns:
        None
    '''
    model.eval()

    loss_fn = nn.MSELoss()
    losses = []
    running_psnr = []
    running_ssim = []
    running_msssim = []

    with torch.no_grad():
        for img, target in test_loader:
            img, target = img.to(device), target.to(device)
            out = model(img)
            loss = loss_fn(out, target)
            losses.append(loss.item())

            # Compute PSNR for batch
            psnr = compute_psnr(target, out)
            running_psnr.append(psnr.item())

            # Compute SSIM for batch
            ssim_ = ssim(target, out, data_range=1, size_average=True)
            running_ssim.append(ssim_.item())

            # Compute MS SSIM for batch
            ms_ssim_ = ms_ssim(target, out, data_range=1, size_average=True)
            running_msssim.append(ms_ssim_.item())

    avg_loss = sum(losses)/len(losses)
    print("Average Evaluation Loss: {:.6f}".format( avg_loss ))
    print(f"Average PSNR: { torch.mean( torch.as_tensor(running_psnr) ) } dB")
    print(f"Std. Deviation of PSNR: { torch.std( torch.as_tensor(running_psnr) ) }")

    print(f"Average SSIM: { torch.mean( torch.as_tensor(running_ssim) ) }")
    print(f"Std. Deviation of SSIM: { torch.std( torch.as_tensor(running_ssim) ) }")

    print(f"Average MS SSIM: { torch.mean( torch.as_tensor(running_msssim) ) }")
    print(f"Std. Deviation of MS SSIM: { torch.std( torch.as_tensor(running_msssim) ) }")
    return avg_loss


def train(args: Any, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader):
    '''
    Description:
        Function containing training loop and to parse args to start training
    
    Args:
        args: User Input options from command line (argparser)
        model: nn.Module for model object
        train_loader: DataLoader object for training data
        test_loader: DataLoader object for testing data
    
    Returns:
        train_losses: list containing training loss from each epoch
        val_losses: list containing testing/validation loss from each epoch
    '''

    # Some basic logging
    print('Number of Training samples: {}'.format(len(train_loader)*args.batch_size))
    print('Number of Validation samples: {}'.format(len(test_loader)*args.batch_size))

    # Detect and choose device
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    epochs = args.num_epochs
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])
    
    # More logging
    print("No. of trainable parameters: {}".format(trainable))
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

    # Create save dir if it doesn't exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Training Loop
    for epoch in range(1, epochs+1):
        l1 = train_one_epoch(model, train_loader, device, optimizer, epoch)
        l2 = evaluate(model, device, test_loader)
        scheduler.step()

        train_losses.append(l1)
        val_losses.append(l2)
  
        # Checkpoint the model periodically
        if epoch % args.log_interval == 0:
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
  
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pth'))
    return train_losses, val_losses



if __name__ == '__main__':
    # Initialize argparser object
    args = parse()

    # Training
    if args.train:
        train_loader, val_loader = get_loaders(args)
        model = Autoencoder( args.use_pretrained, depth=args.decoder_depth, interpolation=args.interpolation )
        train_losses, val_losses = train(args, model, train_loader, val_loader)
        plot_losses(args, train_losses, val_losses)
        visualize_samples(args, model)
    
    # Model evaluation
    elif args.eval_pth or args.eval_ckpt:
        if not os.path.exists(args.model_path):
            raise ValueError('Model path is invalid!')

        if args.device == 'cuda':
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        else:
            device = torch.device('cpu')

        model = Autoencoder(depth=args.decoder_depth, interpolation=args.interpolation).to(device)
        
        if args.eval_pth:
            state_dict = torch.load(args.model_path, map_location=device)
        else:
            ckpt = torch.load(args.model_path, map_location=device)
            state_dict = ckpt['model_state_dict']
        
        model.load_state_dict(state_dict)
        _, val_loader = get_loaders(args)
        
        avg_loss = evaluate(model, device, val_loader)
    
    # Generating plots, visualizing layers of the model, etc.
    elif args.visualize:
        if args.device == 'cuda':
            device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        else:
            device = torch.device('cpu')
        model = Autoencoder(depth=args.decoder_depth, interpolation=args.interpolation).to(device)

        if args.model_path[-4:] == '.pth':
            model.load_state_dict( torch.load(args.model_path, map_location=device) )
        elif args.model_path[-5:] == '.ckpt':
            ckpt = torch.load(args.model_path, map_location=device)
            state_dict = ckpt['model_state_dict']
            model.load_state_dict(state_dict)
        visualize_samples(args, model)