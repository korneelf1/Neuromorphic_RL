import threading 
import torch.multiprocessing as mp
from worker import Worker, MasterModel, MasterModel_continuous
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
        'dt': 0.05,
        'nr_episodes': 25e3,
        'max_episode_length':200,
        'normalize_steps': True, # normalizing based on t1*grad1 + t2*grad2 + ... + tn*gradn/t1+t2+...+tn
        # 'model': 'small', #small smallest currently not implemented yet
    }
    if Continuous:
        global_model = MasterModel_continuous(**args)
    else:
        global_model = MasterModel(**args)
    print(global_model.env)
    global_model.start()
    # I want to create workers that perform the run function and update the global model
    # I want to create a global model that is updated by the workers
    # cpu_count = mp.cpu_count()
    # workers = [Worker(global_model, **args) for i in range(cpu_count)]
    # for worker in workers:
    #     worker.start()
    # for worker in workers:
    print('global model started')
    global_model.run()
    print('global model finished')
    plt.plot(list(range(len(global_model.episode_times))), global_model.episode_times)
    plt.savefig('SNN_in248out_25e3_0gain.png') # save the figure with name SNN_128128_200Hz_100e3
    
    # Save the data of the x and y data of the plot in a txt file with the same naming of the figure
    with open('SNN_in248out_25e3_0gain.txt', 'w') as f:
        for i in range(len(global_model.episode_times)):
            f.write(f'{i}\t{global_model.episode_times[i]}\n')
    
    
    global_model.save_model(path='A3C/past_trainings/Figures/SNN_in248out_25e3_0gain.pt')
    global_model.join()

    
    # # global_model.join()
    # import pickle
    # with open("times_SNN_10000", "wb") as fp:   #Pickling
    #     pickle.dump(global_model.episode_times, fp)

if __name__ == "__main__":        main()

