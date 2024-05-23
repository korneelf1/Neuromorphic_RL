import threading 
import torch.multiprocessing as mp
from worker import MasterModel
# from worker_continuous import MasterModel_continuous
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
import wandb

# arguments
SPIKING = True
SAVE_DIR = '\past_trainings'
LR = 1e-4
BETAS = [0.9,0.99]
GAIN = 0.
DT = 0.02
NR_EPISODES = 5e3
MAX_EPISODE_LENGTH = 300
NORMALIZE_STEPS = True

wandb.init(project='drone_snn', config={'lr': LR, 'spiking':SPIKING, 'betas': BETAS, 'gain': GAIN, 'dt': DT, 'nr_episodes': NR_EPISODES, 'max_episode_length': MAX_EPISODE_LENGTH, 'normalize_steps': NORMALIZE_STEPS})
def main():
    Continuous = False
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = 'cpu'
    print('Device in use: ', str(device))
    print('Number of CPUs: ', str(mp.cpu_count()))
    args = {
        'spiking' : SPIKING,
        'device' : device,
        'save_dir': SAVE_DIR,
        'lr': LR,
        'betas': BETAS,
        'gain': GAIN, # add noise to inputs
        'dt': DT,
        'nr_episodes': NR_EPISODES,
        'max_episode_length':MAX_EPISODE_LENGTH,
        'normalize_steps': NORMALIZE_STEPS, # normalizing based on t1*grad1 + t2*grad2 +f ... + tn*gradn/t1+t2+...+tn
        # 'model': 'small', #small smallest currently not implemented yet
    }
    
    
    if Continuous:
        global_model = MasterModel_continuous(**args)
    else:
        global_model = MasterModel(**args)
        global_model.global_model.load_state_dict(torch.load('drone_snn_vel_syn_lif_1e4_3232.pt',map_location=torch.device('cpu')))
          

    global_model.start()

    print('global model started')
    global_model.run()
    print('global model finished')
    global_model.join()
    global_model.save_model(path=f'drone_snn_pos_sparse_reward.pt')
    # plt.figure()
    

    # plt.plot(list(range(len(global_model.episode_times))), global_model.episode_times)
    # plt.savefig('drone_snn_pos.png') # save the figure with name SNN_246_200Hz_100e3
    print(args)
    # global_model.env.render('post')
    # Save the data of the x and y data of the plot in a txt file with the same naming of the figure
    with open('drone_snn_pos.txt', 'w') as f:
        for i in range(len(global_model.episode_times)):
            f.write(f'{i}\t{global_model.episode_times[i]}\n')
    
    


if __name__ == "__main__":        main()

