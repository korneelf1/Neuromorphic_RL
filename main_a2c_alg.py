import threading 
import torch.multiprocessing as mp
from worker import Worker, MasterModel
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm

def main():
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
    }
    global_model = MasterModel(**args)
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
    plt.show()
    global_model.save_model()
    #     worker.join()
    global_model.join()

    
    # global_model.join()
    import pickle
    with open("times_SNN_10000", "wb") as fp:   #Pickling
        pickle.dump(global_model.episode_times, fp)

if __name__ == "__main__":
    main()