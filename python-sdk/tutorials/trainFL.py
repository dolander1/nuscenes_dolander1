# -*- coding: utf-8 -*-
# GPT with covernet input

import importlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import os

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
version = "v1.0-trainval" # v1.0-mini, v1.0-trainval
seconds_of_history_used = 2.0 # 2.0
sequences_per_instance = "one_sequences_per_instance" # one_sequences_per_instance, non_overlapping_sequences_per_instance, all_sequences_per_instance, 

train_img_tensor_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/train_img_tensor_list.pt")
train_agent_state_vector_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/train_agent_state_vector_list.pt")
train_future_xy_local_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/train_future_xy_local_list.pt")

val_img_tensor_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/val_img_tensor_list.pt")
val_agent_state_vector_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/val_agent_state_vector_list.pt")
val_future_xy_local_list = torch.load(f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/val_future_xy_local_list.pt")

scale_factor = 1/10 # downsample images
# Squeeze for correct dimensions
for i, train_img_tensor in enumerate(train_img_tensor_list):
    dummy = torch.nn.functional.interpolate(train_img_tensor, scale_factor=scale_factor, mode='bilinear')
    train_img_tensor_list[i] = torch.squeeze(dummy, dim=0)
    train_agent_state_vector_list[i] = torch.squeeze(train_agent_state_vector_list[i], dim=0)

for j, val_img_tensor in enumerate(val_img_tensor_list):
    dummy = torch.nn.functional.interpolate(val_img_tensor, scale_factor=scale_factor, mode='bilinear')
    val_img_tensor_list[j] = torch.squeeze(dummy, dim=0)
    val_agent_state_vector_list[j] = torch.squeeze(val_agent_state_vector_list[j], dim=0)


################################################################################################################################################
# For testing
train_short_size = 2048
short_train_img_tensor_list = train_img_tensor_list[:train_short_size]
short_train_agent_state_vector_list = train_agent_state_vector_list[:train_short_size]
short_train_future_xy_local_list = train_future_xy_local_list[:train_short_size]
val_short_size = 512
short_val_img_tensor_list = val_img_tensor_list[:val_short_size]
short_val_agent_state_vector_list = val_agent_state_vector_list[:val_short_size]
short_val_future_xy_local_list = val_future_xy_local_list[:val_short_size]


# Prints
train_num_datapoints = len(train_img_tensor_list)
short_train_num_datapoints = len(short_train_img_tensor_list)
print(f"train_num_datapoints = {train_num_datapoints}")
print(f"train_num_datapoints short = {short_train_num_datapoints}")
val_num_datapoints = len(val_img_tensor_list)
short_val_num_datapoints = len(short_val_img_tensor_list)
print(f"val_num_datapoints = {val_num_datapoints}")
print(f"val_num_datapoints short = {short_val_num_datapoints}")


# Variables
batch_size = 8
shuffle = True # Set to True if you want to shuffle the data in the dataloader
num_modes = 415 # 2206, 415, 64 (match with eps_traj_set)
eps_traj_set = 4 # 2, 4, 8 (match with num_modes)
learning_rate = 1e-4 # From Covernet paper: fixed learning rate of 1e−4
start_epoch = 0
num_epochs = 2
accum_iter = 1 # batch accumulation parameter, multiplies batch_size


# Define datasets
train_shortDataset = NuscenesDataset(short_train_img_tensor_list, short_train_agent_state_vector_list, short_train_future_xy_local_list)
val_shortDataset = NuscenesDataset(short_val_img_tensor_list, short_val_agent_state_vector_list, short_val_future_xy_local_list)

# Instantiate dataloaders
train_shortDataloader = DataLoader(train_shortDataset, batch_size=batch_size, shuffle=shuffle)
val_shortDataloader = DataLoader(val_shortDataset, batch_size=batch_size, shuffle=shuffle)

# File path
file_path = f"tmpResults/results_epochs={num_epochs}" # To flower


# Initialize the CoverNet model
backbone = ResNetBackbone('resnet50') 
covernet = CoverNetNoRelu(backbone, num_modes)
# covernet.load_state_dict(torch.load(f'{file_path}_weights.pth'))

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
lattice = torch.Tensor(lattice).to(device)


# Training starts
print("\nTraining starts:")
results_string = "" # To flower
train_logits_file = f'{file_path}_train_logits.npy' # To flower
train_gt_traj_file = f'{file_path}_train_ground_truth.npy' # To flower
val_logits_file = f'{file_path}_val_logits.npy' # To flower
val_gt_traj_file = f'{file_path}_val_ground_truth.npy' # To flower

# Training and validation loop
for epoch in range(start_epoch,num_epochs):
    
    # TRAINING
    covernet.train()
    train_epochLoss = 0
    train_total = 0
    train_correct = 0
    train_logits_list = [] # To flower
    train_gt_traj_list = [] # To flower
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
            closest_lattice_trajectory = similarity_function(lattice, ground_truth)
            train_total += 1
            train_correct += (predicted == closest_lattice_trajectory)#.sum().item()

            
        # Create lists of saved data
        train_logits_list.append(logits.cpu().detach().numpy()) # To flower
        train_gt_traj_list.append(ground_truth_trajectory.cpu().detach().numpy())  # To flower

    # Save logits and ground_truth_trajectory in separate files
    if epoch == 0:  # To flower
        # Concatenate the lists to create numpy arrays
        train_logits_array = np.concatenate(train_logits_list, axis=0) # To flower
        train_gt_traj_array = np.concatenate(train_gt_traj_list, axis=0) # To flower
    else: # To flower
        train_logits_array = np.load(train_logits_file) # To flower
        train_gt_traj_array = np.load(train_gt_traj_file) # To flower
        # Concatenate the lists to create numpy arrays
        train_logits_array = np.concatenate([train_logits_array] + train_logits_list, axis=0) # To flower
        train_gt_traj_array = np.concatenate([train_gt_traj_array] + train_gt_traj_list, axis=0) # To flower
    # save numpy arrays in files
    np.save(train_logits_file, train_logits_array) # To flower
    np.save(train_gt_traj_file, train_gt_traj_array) # To flower
    
    ######################################################################################################################
    
    
    # VALIDATION
    covernet.eval()
    val_epochLoss = 0
    val_total = 0
    val_correct = 0
    val_logits_list = [] # To flower
    val_gt_traj_list = [] # To flower
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
            
            val_total += ground_truth_trajectory.size(0)
            _, predicted = torch.max(logits, 1) 
#             print(f"ground_truth_trajectory.shape = {ground_truth_trajectory.shape}") # [batch_size, 12, 2]
#             print(f"predicted.shape = {predicted.shape}") # [batch_size]
#             print(f"logits.shape = {logits.shape}") # [batch_size, num_modes]
            
            # Accuracy
            for index, ground_truth in enumerate(ground_truth_trajectory):
                closest_lattice_trajectory = similarity_function(lattice, ground_truth)
#                 print(f"predicted[index] = {predicted[index]}")
#                 print(f"closest_lattice_trajectory = {closest_lattice_trajectory}")
                val_correct += (predicted[index] == closest_lattice_trajectory)#.sum().item()
#                 print(f"closest_lattice_trajectory index = {closest_lattice_trajectory}") # 1
#                 print(f"Predicted lattice trajectory index = {predicted[index].item()}") # 1
#                 print(f"ground_truth = {ground_truth}") # [2, 12]
#                 print(f"predicted = {lattice[predicted[index].item()]}") # [2, 12]
#                 print("----------------------------------------------------------------")
                
            
            # Create lists of saved data
            val_logits_list.append(logits.cpu().numpy()) # To flower
            val_gt_traj_list.append(ground_truth_trajectory.cpu().numpy()) # To flower

    # Save logits and ground_truth_trajectory in separate files
    if epoch == 0: # To flower
        # Concatenate the lists to create numpy arrays
        val_logits_array = np.concatenate(val_logits_list, axis=0) # To flower
        val_gt_traj_array = np.concatenate(val_gt_traj_list, axis=0) # To flower
    else: # To flower
        val_logits_array = np.load(val_logits_file) # To flower
        val_gt_traj_array = np.load(val_gt_traj_file) # To flower
        # Concatenate the lists to create numpy arrays
        val_logits_array = np.concatenate([val_logits_array] + val_logits_list, axis=0) # To flower
        val_gt_traj_array = np.concatenate([val_gt_traj_array] + val_gt_traj_list, axis=0) # To flower
    # save numpy arrays in files
    np.save(val_logits_file, val_logits_array) # To flower
    np.save(val_gt_traj_file, val_gt_traj_array) # To flower
  

    # Print losses for this epoch
    thisResult = f"Epoch [{epoch+1}/{num_epochs}]: Training loss: {train_epochLoss/train_total:.5f} | Validation loss: {val_epochLoss/val_total:.5f} || Training accuracy: {train_correct/train_total*100:.1f} % | Validation accuracy: {val_correct/val_total*100:.1f} %\n"  # To flower
    with open(f'{file_path}_loss_and_acc.txt', "a") as file: # To flower
        file.write(thisResult) # To flower
    print(thisResult)  # Not To flower
    
    # Save weights every epoch
    torch.save(covernet.state_dict(), f'{file_path}_weights.pth') # To flower


# Training complete
print("Training complete!")


