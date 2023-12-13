import threading 
import torch.multiprocessing as mp
from worker import MasterModel
from worker_continuous import MasterModel_continuous
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm

def main():
    Continuous = False
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = 'cpu'
    print('Device in use: ', str(device))
    print('Number of CPUs: ', str(mp.cpu_count()))
    args = {
        'spiking' : True,
        'device' : device,
        'save_dir': '\past_trainings',
        'lr': 1e-4,
        'betas': [0.9,0.99],
        'gain': 0., # add noise to inputs
        'dt': 0.02,
        'nr_episodes': 5e3,
        'max_episode_length':500,
        'normalize_steps': True, # normalizing based on t1*grad1 + t2*grad2 +f ... + tn*gradn/t1+t2+...+tn
        # 'model': 'small', #small smallest currently not implemented yet
    }
    
    
    if Continuous:
        global_model = MasterModel_continuous(**args)
    else:
        global_model = MasterModel(**args)
        # global_model.global_model.load_state_dict(torch.load('drone_snn_pos.pt'))
          

    global_model.start()

    print('global model started')
    global_model.run()
    print('global model finished')
    global_model.join()
    global_model.save_model(path='drone_snn_vel.pt')

    # plt.plot(list(range(len(global_model.episode_times))), global_model.episode_times)
    # plt.savefig('drone_snn_pos.png') # save the figure with name SNN_246_200Hz_100e3
    print(args)
    # global_model.env.render('post')
    # Save the data of the x and y data of the plot in a txt file with the same naming of the figure
    with open('drone_snn_pos.txt', 'w') as f:
        for i in range(len(global_model.episode_times)):
            f.write(f'{i}\t{global_model.episode_times[i]}\n')
    
    


if __name__ == "__main__":        main()

