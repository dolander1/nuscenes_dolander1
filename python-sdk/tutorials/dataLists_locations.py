import torch
import numpy as np
import utilsHannes as utilsH
import importlib
importlib.reload(utilsH)



version = "v1.0-mini" # v1.0-mini, v1.0-trainval
if version == "v1.0-mini":
    train_subset = "mini_train"
    val_subset = "mini_val"
elif version == "v1.0-trainval":
    train_subset = "train"
    val_subset = "val"

sequences_per_instance = "one_sequences_per_instance" # one_sequences_per_instance, non_overlapping_sequences_per_instance, all_sequences_per_instance

DATAROOT = "data/sets/nuscenes"
seconds_of_history_used = 2.0 # Half second steps

########################################################################################################################################

# Get training data
train_img_list_Boston, train_img_tensor_list_Boston, train_agent_state_vector_list_Boston, train_future_xy_local_list_Boston, train_img_list_Singapore, train_img_tensor_list_Singapore, train_agent_state_vector_list_Singapore, train_future_xy_local_list_Singapore = utilsH.get_and_format_data_with_location(version, DATAROOT, train_subset, seconds_of_history_used, sequences_per_instance)

# Concatenate Boston and Singapore train
train_img_tensor_list = train_img_tensor_list_Boston + train_img_tensor_list_Singapore
train_agent_state_vector_list = train_agent_state_vector_list_Boston + train_agent_state_vector_list_Singapore
train_future_xy_local_list = train_future_xy_local_list_Boston + train_future_xy_local_list_Singapore

# Save training data
torch.save(train_img_tensor_list, f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/location_B={len(train_img_tensor_list_Boston)}_S={len(train_img_tensor_list_Singapore)}_train_img_tensor_list.pt")

torch.save(train_agent_state_vector_list, f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/location_B={len(train_agent_state_vector_list_Boston)}_S={len(train_agent_state_vector_list_Singapore)}_train_agent_state_vector_list.pt")

torch_train_future_xy_local_list = [torch.Tensor(train_future_xy_local) for train_future_xy_local in train_future_xy_local_list]

torch.save(torch_train_future_xy_local_list, f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/location_B={len(train_future_xy_local_list_Boston)}_S={len(train_future_xy_local_list_Singapore)}_train_future_xy_local_list.pt")


########################################################################################################################################

# Get validation data
val_img_list_Boston, val_img_tensor_list_Boston, val_agent_state_vector_list_Boston, val_future_xy_local_list_Boston, val_img_list_Singapore, val_img_tensor_list_Singapore, val_agent_state_vector_list_Singapore, val_future_xy_local_list_Singapore = utilsH.get_and_format_data_with_location(version, DATAROOT, val_subset, seconds_of_history_used, sequences_per_instance)

# Concatenate Boston and Singapore val
val_img_tensor_list = val_img_tensor_list_Boston + val_img_tensor_list_Singapore
val_agent_state_vector_list = val_agent_state_vector_list_Boston + val_agent_state_vector_list_Singapore
val_future_xy_local_list = val_future_xy_local_list_Boston + val_future_xy_local_list_Singapore

# Save validation data
torch.save(val_img_tensor_list, f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/location_B={len(val_img_tensor_list_Boston)}_S={len(val_img_tensor_list_Singapore)}_val_img_tensor_list.pt")

torch.save(val_agent_state_vector_list, f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/location_B={len(val_agent_state_vector_list_Boston)}_S={len(val_agent_state_vector_list_Singapore)}_val_agent_state_vector_list.pt")

torch_val_future_xy_local_list = [torch.Tensor(val_future_xy_local) for val_future_xy_local in val_future_xy_local_list]

torch.save(torch_val_future_xy_local_list, f"dataLists/{version}/{sequences_per_instance}/{seconds_of_history_used}/location_B={len(val_future_xy_local_list_Boston)}_S={len(val_future_xy_local_list_Singapore)}_val_future_xy_local_list.pt")



