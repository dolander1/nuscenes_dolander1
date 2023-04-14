import torch
import numpy as np
import utilsHannes as utilsH
import importlib
importlib.reload(utilsH)



version = "v1.0-trainval" # v1.0-mini, v1.0-trainval
if version == "v1.0-mini":
    train_subset = "mini_train"
    val_subset = "mini_val"
elif version == "v1.0-trainval":
    train_subset = "train"
    val_subset = "val"
    
DATAROOT = "data/sets/nuscenes"
seconds_of_history_used = 2.0 # Half second steps


# Get training data
train_img_list, train_img_tensor_list, train_agent_state_vector_list, train_future_xy_local_list = utilsH.get_and_format_data(version, DATAROOT, train_subset, seconds_of_history_used)

# Save training data
torch.save(train_img_tensor_list, f"dataLists/{version}/{seconds_of_history_used}/train_img_tensor_list.pt")
torch.save(train_agent_state_vector_list, f"dataLists/{version}/{seconds_of_history_used}/train_agent_state_vector_list.pt")
torch_train_future_xy_local_list = [torch.Tensor(train_future_xy_local) for train_future_xy_local in train_future_xy_local_list]
torch.save(torch_train_future_xy_local_list, f"dataLists/{version}/{seconds_of_history_used}/train_future_xy_local_list.pt")

# Load training data
loaded_train_img_tensor_list = torch.load(f"dataLists/{version}/{seconds_of_history_used}/train_img_tensor_list.pt")
loaded_train_agent_state_vector_list = torch.load(f"dataLists/{version}/{seconds_of_history_used}/train_agent_state_vector_list.pt")
loaded_train_future_xy_local_list = torch.load(f"dataLists/{version}/{seconds_of_history_used}/train_future_xy_local_list.pt")


# Get validation data
val_img_list, val_img_tensor_list, val_agent_state_vector_list, val_future_xy_local_list = utilsH.get_and_format_data(version, DATAROOT, val_subset, seconds_of_history_used)

# Save validation data
torch.save(val_img_tensor_list, f"dataLists/{version}/{seconds_of_history_used}/val_img_tensor_list.pt")
torch.save(val_agent_state_vector_list, f"dataLists/{version}/{seconds_of_history_used}/val_agent_state_vector_list.pt")
torch_val_future_xy_local_list = [torch.Tensor(val_future_xy_local) for val_future_xy_local in val_future_xy_local_list]
torch.save(torch_val_future_xy_local_list, f"dataLists/{version}/{seconds_of_history_used}/val_future_xy_local_list.pt")

# Load validation data
loaded_val_img_tensor_list = torch.load(f"dataLists/{version}/{seconds_of_history_used}/val_img_tensor_list.pt")
loaded_val_agent_state_vector_list = torch.load(f"dataLists/{version}/{seconds_of_history_used}/val_agent_state_vector_list.pt")
loaded_val_future_xy_local_list = torch.load(f"dataLists/{version}/{seconds_of_history_used}/val_future_xy_local_list.pt")

    
# Function to check if two lists of tensors are identical
def are_tensor_lists_identical(list1, list2, type):
    if (len(list1) != len(list2)):
        return False
    
    if type == "tensor":
        for tensor1, tensor2 in zip(list1, list2):
            if not torch.allclose(tensor1, tensor2):
                return False
        
    if type == "array":
        for array1, array2 in zip(list1, list2):
            if not np.array_equal(array1, array2):
                return False
    return True

# Check if the two lists of tensors are identical
identical1 = are_tensor_lists_identical(train_img_tensor_list, loaded_train_img_tensor_list, "tensor")
identical2 = are_tensor_lists_identical(train_agent_state_vector_list, loaded_train_agent_state_vector_list, "tensor")
identical3 = are_tensor_lists_identical(torch_train_future_xy_local_list, loaded_train_future_xy_local_list, "tensor")
identical4 = are_tensor_lists_identical(val_img_tensor_list, loaded_val_img_tensor_list, "tensor")
identical5 = are_tensor_lists_identical(val_agent_state_vector_list, loaded_val_agent_state_vector_list, "tensor")
identical6 = are_tensor_lists_identical(torch_val_future_xy_local_list, loaded_val_future_xy_local_list, "tensor")

print("Are the two lists of tensors identical?", identical1)
print("Are the two lists of tensors identical?", identical2)
print("Are the two lists of tensors identical?", identical3)
print("Are the two lists of tensors identical?", identical4)
print("Are the two lists of tensors identical?", identical5)
print("Are the two lists of tensors identical?", identical6)


