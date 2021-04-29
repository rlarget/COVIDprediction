import torch
from helper_evaluation import compute_accuracy

def multi_train(models,
                num_epochs,
                train_loader,
                valid_loader,
                test_loader,
                device,
                logging_interval=50,
                scheduler=None,
                print_epoch_logging=True,
                scheduler_on='valid_acc'):
    
    if not isinstance(models,list):
        models = [models]
    
    saved_losses = [[]*len(models)]
    
    for epoch in range(num_epochs):
        
        for model in models:
            model.train()
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            
            features.to(device)
            targets.to(device)
            
            logits = [model(features) for model in models]
            losses = [torch.nn.functional.cross_entropy(logit, targets) for logit in logits]
            
            for loss, model, saved in zip(losses, models,saved_losses):
                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()
                saved.append(loss.item())
                
            if logging_interval and not batch_idx % logging_interval:
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} ')
                
    return saved_losses
                
            
            