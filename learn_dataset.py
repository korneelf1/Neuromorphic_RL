import torch
import actor_critics as ac
import numpy as np

data_path = 'dataset.npy'
print(np.load(data_path, allow_pickle=True).shape)
# data = torch.load(data_path)

# Create a DataLoader using the loaded data
dataloader = torch.utils.data.DataLoader(data_path, batch_size=1, shuffle=True)

# Iterate over the data
for batch in dataloader:
    # Process the batch
    # Your code here
    print(batch)