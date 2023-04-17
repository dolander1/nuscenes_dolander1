# GPT with covernet input

import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np

from nuscenes.prediction.models.backbone import ResNetBackbone

from utilsHannes import CoverNetNoRelu
from utilsHannes import ConstantLatticeLoss
from utilsHannes import mean_pointwise_l2_distance
import utilsHannes as utilsH
importlib.reload(utilsH)

#################################################################################################################################
# Define your custom dataset class that inherits from torch.utils.data.Dataset
class NuscenesDataset(Dataset):
    def __init__(self, image_data, agent_state_data, ground_truth_data):
        self.image_data = image_data
        self.agent_state_data = agent_state_data
        self.ground_truth_data = ground_truth_data
        
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, index):
        image_data_item = self.image_data[index]
        agent_state_data_item = self.agent_state_data[index]
        ground_truth_data_item = self.ground_truth_data[index]
        
        return image_data_item, agent_state_data_item, ground_truth_data_item

################################################################################################################################################
# Load data
version = "v1.0-mini" # v1.0-mini, v1.0-trainval
seconds_of_history_used = 2.0 # 2.0
sequences_per_instance = "one_sequences_per_instance" # one_sequences_per_instance, all_sequences_per_instance

train_img_tensor_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/train_img_tensor_list.pt")
train_agent_state_vector_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/train_agent_state_vector_list.pt")
train_future_xy_local_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/train_future_xy_local_list.pt")

val_img_tensor_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/val_img_tensor_list.pt")
val_agent_state_vector_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/val_agent_state_vector_list.pt")
val_future_xy_local_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/val_future_xy_local_list.pt")

# Squeeze for correct dimensions
for i, train_img_tensor in enumerate(train_img_tensor_list):
    train_img_tensor_list[i] = torch.squeeze(train_img_tensor, dim=0)
    train_agent_state_vector_list[i] = torch.squeeze(train_agent_state_vector_list[i], dim=0)
    
for j, val_img_tensor in enumerate(val_img_tensor_list):
    val_img_tensor_list[j] = torch.squeeze(val_img_tensor, dim=0)
    val_agent_state_vector_list[j] = torch.squeeze(val_agent_state_vector_list[j], dim=0)

    
################################################################################################################################################

# For testing
train_short_size = 12800
short_train_img_tensor_list = train_img_tensor_list[:train_short_size]
short_train_agent_state_vector_list = train_agent_state_vector_list[:train_short_size]
short_train_future_xy_local_list = train_future_xy_local_list[:train_short_size]
val_short_size = 6400
short_val_img_tensor_list = val_img_tensor_list[:val_short_size]
short_val_agent_state_vector_list = val_agent_state_vector_list[:val_short_size]
short_val_future_xy_local_list = val_future_xy_local_list[:val_short_size]


# Prints
train_num_datapoints = len(train_img_tensor_list)
# print(f"train_num_datapoints whole dataset = {train_num_datapoints}")
short_train_num_datapoints = len(short_train_img_tensor_list)
# print(f"train_num_datapoints short = {short_train_num_datapoints}")
# print(f"train_img_tensor_list[0] = {train_img_tensor_list[0].size()}")
# print(f"train_agent_state_vector_list[0] = {train_agent_state_vector_list[0].size()}")
# print(f"train_future_xy_local_list[0] = {train_future_xy_local_list[0].size()}\n")
val_num_datapoints = len(val_img_tensor_list)
# print(f"val_num_datapoints whole dataset = {val_num_datapoints}")
short_val_num_datapoints = len(short_val_img_tensor_list)
# print(f"val_num_datapoints short = {short_val_num_datapoints}")
# print(f"val_img_tensor_list[0] = {val_img_tensor_list[0].size()}")
# print(f"val_agent_state_vector_list[0] = {val_agent_state_vector_list[0].size()}")
# print(f"val_future_xy_local_list[0] = {val_future_xy_local_list[0].size()}\n")


# Variables
batch_size = 16
shuffle = True # Set to True if you want to shuffle the data in the dataloader
num_modes = 64 # 2206, 415, 64 (match with eps_traj_set)
eps_traj_set = 8 # 2, 4, 8 (match with num_modes)
learning_rate = 1e-4 # From Covernet paper: fixed learning rate of 1e−4
num_epochs = 4998

# Define datasets
train_dataset = NuscenesDataset(train_img_tensor_list, train_agent_state_vector_list, train_future_xy_local_list)
train_shortDataset = NuscenesDataset(short_train_img_tensor_list, short_train_agent_state_vector_list, short_train_future_xy_local_list)
val_dataset = NuscenesDataset(val_img_tensor_list, val_agent_state_vector_list, val_future_xy_local_list)
val_shortDataset = NuscenesDataset(short_val_img_tensor_list, short_val_agent_state_vector_list, short_val_future_xy_local_list)

# Instantiate dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
train_shortDataloader = DataLoader(train_shortDataset, batch_size=batch_size, shuffle=shuffle)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
val_shortDataloader = DataLoader(val_shortDataset, batch_size=batch_size, shuffle=shuffle)

# Initialize the CoverNet model
backbone = ResNetBackbone('resnet50') 
covernet = CoverNetNoRelu(backbone, num_modes)

# Lattice and similarity function
with open(f'data/sets/nuscenes-prediction-challenge-trajectory-sets/epsilon_{eps_traj_set}.pkl', 'rb') as f:
    lattice = np.array(pickle.load(f))
similarity_function = mean_pointwise_l2_distance

# Define your loss function and optimizer
loss_function = ConstantLatticeLoss(lattice, similarity_function)
# optimizer = optim.Adam(covernet.parameters(), lr=learning_rate) 
optimizer = optim.SGD(covernet.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4) # from author https://github.com/nutonomy/nuscenes-devkit/issues/578


# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
covernet.to(device)


# Training starts
print("\nTraining starts:")

# batch accumulation parameter
accum_iter = 4 

# Open a file in append mode (will create a new file or append to an existing one)
file_path = f"results_epochs={num_epochs}_eps={eps_traj_set}_batch_size={batch_size*accum_iter}_lr={learning_rate}_shuffle={shuffle}.txt"
results_string = ""

# Training and validation loop
for epoch in range(num_epochs):
    
    # TRAINING
    covernet.train()
    train_epochLoss = 0
    train_total = 0
    train_correct = 0
    for train_batchCount, train_batch in enumerate(train_shortDataloader):

        # Get train_batch data
        image_tensor, agent_state_vector, ground_truth_trajectory = train_batch

        # Send to device
        image_tensor = image_tensor.to(device)
        agent_state_vector = agent_state_vector.to(device)
        ground_truth_trajectory = ground_truth_trajectory.to(device)
        
        # Forward pass
        logits = covernet(image_tensor, agent_state_vector)

        # Compute loss
        loss = loss_function(logits, ground_truth_trajectory)
        train_epochLoss += loss.item()

#         # Backward pass
#         loss.backward()
#         optimizer.step()
        
#         # Zero the gradients
#         optimizer.zero_grad()
         
        loss = loss / accum_iter
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if ((train_batchCount + 1) % accum_iter == 0) or (train_batchCount + 1 == len(train_shortDataloader)):
            optimizer.step()
            optimizer.zero_grad()
        
        # Compute accuracy
        for logit, ground_truth in zip(logits, ground_truth_trajectory):
            _, predicted = torch.max(logit, 0)
            closest_lattice_trajectory = similarity_function(torch.Tensor(lattice).to(device), ground_truth)
            train_total += 1
            train_correct += (predicted == closest_lattice_trajectory).sum().item()

        # Print loss for this train_batch
        # print(f"train_batch [{train_batchCount+1}/{int(short_train_num_datapoints/batch_size)+1}], Batch Loss: {loss.item():.4f}")
     
    
    # VALIDATION
    covernet.eval()
    val_epochLoss = 0
    val_total = 0
    val_correct = 0
    with torch.no_grad():
        for val_batchCount, val_batch in enumerate(val_shortDataloader):

            # Get val_batch data
            image_tensor, agent_state_vector, ground_truth_trajectory = val_batch

            # Send to device
            image_tensor = image_tensor.to(device)
            agent_state_vector = agent_state_vector.to(device)
            ground_truth_trajectory = ground_truth_trajectory.to(device)

            # Forward pass
            logits = covernet(image_tensor, agent_state_vector)

            # Compute loss
            loss = loss_function(logits, ground_truth_trajectory)
            val_epochLoss += loss.item()

            # Compute accuracy
            for logit, ground_truth in zip(logits, ground_truth_trajectory):
                _, predicted = torch.max(logit, 0)
                closest_lattice_trajectory = similarity_function(torch.Tensor(lattice).to(device), ground_truth)
                val_total += 1
                val_correct += (predicted == closest_lattice_trajectory).sum().item()

            # Print loss for this val_batch
            # print(f"val_batch [{val_batchCount+1}/{int(short_val_num_datapoints/batch_size)+1}], Batch Loss: {loss.item():.4f}")
     
    # Print losses for this epoch
    thisResult = f"Epoch [{epoch+1}/{num_epochs}]: Training loss: {train_epochLoss:.3f} | Validation loss: {val_epochLoss:.3f} || Training accuracy: {train_correct/train_total*100:.1f} % | Validation accuracy: {val_correct/val_total*100:.1f} %\n"
    with open(file_path, "a") as file:
        file.write(thisResult)  # Append the text to the file
    print(thisResult)


# Training complete
print("Training complete!")

