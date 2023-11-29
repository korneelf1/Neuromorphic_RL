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
        'lr': 1e-3,
        'betas': [0.9,0.99],
        'gain': 0., # add noise to inputs
        'dt': 0.02,
        'nr_episodes': 25e3,
        'max_episode_length':500,
        'normalize_steps': True, # normalizing based on t1*grad1 + t2*grad2 +f ... + tn*gradn/t1+t2+...+tn
        # 'model': 'small', #small smallest currently not implemented yet
    }
    if Continuous:
        global_model = MasterModel_continuous(**args)
    else:
        global_model = MasterModel(**args)
        global_model.global_model.load_state_dict(torch.load('drone_snn_small_test_FURTHER.pt'))
    print(global_model.env)
    global_model.start()
    # I want to create workers that perform the run function and update the global model
    # I want to create a global mosnndel that is updated by the workers
    # cpu_count = mp.cpu_count()
    # workers = [Worker(global_model, **args) for i in range(cpu_count)]
    # for worker in workers:
    #     worker.start()
    # for worker in workers:
    print('global model started')
    global_model.run()
    print('global model finished')
    global_model.join()
    global_model.save_model(path='drone_snn_small_test_FURTHER_1.pt')
    print('model saved')
    plt.plot(list(range(len(global_model.episode_times))), global_model.episode_times)
    plt.savefig('drone_snn_small_test_FURTHER_1.png') # save the figure with name SNN_246_200Hz_100e3
    print(args)
    global_model.env.render('post')
    # Save the data of the x and y data of the plot in a txt file with the same naming of the figure
    with open('drone_snn_small_test_FURTHER_1.txt', 'w') as f:
        for i in range(len(global_model.episode_times)):
            f.write(f'{i}\t{global_model.episode_times[i]}\n')
    
    
    

    
    # # global_model.join()
    # import pickle
    # with open("times_snn_small_10000", "wb") as fp:   #Pickling
    #     pickle.dump(global_model.episode_times, fp)

if __name__ == "__main__":        main()

