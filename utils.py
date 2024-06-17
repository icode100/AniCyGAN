import torch
from config import Config
config = Config()
class Utils:
    def save(self,model,optimizer,file_name)->None:
        checkpoint = {
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        torch.save(model.state_dict(),file_name)  
        # torch.save(checkpoint,file_name)     --> for training puposes
    def load(self,model,optimizer,file_name)->None:  # --> for training purposes
        checkpoint = torch.load(file_name, map_location = config.DEVICE)
        print('__loading checkpoint for model__')
        model.load_state_dict(checkpoint['model'])
        print('__loading checkpoint for optimizer__')
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group  in optimizer.param_groups:
            param_group['lr'] = config.LEARNING_RATE
        print('__finished loading checkpoint__')

'''
# for current usage of the finalized pretrained GANs, use 

state_dict = torch.load(config.CHECKPOINT_GEN_ANIMATION(creator)) 
model = Generator(in_channels = 3)
model.load_state_dict(state_dict)
'''
