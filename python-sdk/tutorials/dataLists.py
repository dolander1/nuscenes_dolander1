import torch
import numpy as np
import utilsHannes as utilsH
import importlib
importlib.reload(utilsH)


version = "v1.0-mini" # v1.0-mini, v1.0-trainval
DATAROOT = "data/sets/nuscenes"
seconds_of_history_used = 2.0

# Get training data
train_subset = "mini_train" # 'mini_train', 'mini_val', 'train', 'val'
train_img_list, train_img_tensor_list, train_agent_state_vector_list, train_future_xy_local_list = utilsH.get_and_format_data(version, DATAROOT, train_subset, seconds_of_history_used)

# Save training data
torch.save(train_img_tensor_list, f"dataLists/{version}/train_img_tensor_list.pt")
torch.save(train_agent_state_vector_list, f"dataLists/{version}/train_agent_state_vector_list.pt")
with open(f"dataLists/{version}/train_future_xy_local_list.npz", "wb") as f:
    np.savez(f, *train_future_xy_local_list)
    
# Load training data
loaded_train_img_tensor_list = torch.load(f"dataLists/{version}/train_img_tensor_list.pt")
loaded_train_agent_state_vector_list = torch.load(f"dataLists/{version}/train_agent_state_vector_list.pt")
with open(f"dataLists/{version}/train_future_xy_local_list.npz", "rb") as f:
    loaded_train_future_xy_local = np.load(f)
    loaded_train_future_xy_local_list = [loaded_train_future_xy_local[key] for key in loaded_train_future_xy_local]


# Get validation data
val_subset = "mini_val" # 'mini_train', 'mini_val', 'train', 'val'.
val_img_list, val_img_tensor_list, val_agent_state_vector_list, val_future_xy_local_list = utilsH.get_and_format_data(version, DATAROOT, val_subset, seconds_of_history_used)

# Save validation data
torch.save(val_img_tensor_list, f"dataLists/{version}/val_img_tensor_list.pt")
torch.save(val_agent_state_vector_list, f"dataLists/{version}/val_agent_state_vector_list.pt")
with open(f"dataLists/{version}/val_future_xy_local_list.npz", "wb") as f:
    np.savez(f, *val_future_xy_local_list)
    
# Load validation data
loaded_val_img_tensor_list = torch.load(f"dataLists/{version}/val_img_tensor_list.pt")
loaded_val_agent_state_vector_list = torch.load(f"dataLists/{version}/val_agent_state_vector_list.pt")
with open(f"dataLists/{version}/val_future_xy_local_list.npz", "rb") as f:
    loaded_val_future_xy_local = np.load(f)
    loaded_val_future_xy_local_list = [loaded_val_future_xy_local[key] for key in loaded_val_future_xy_local]

    
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
identical3 = are_tensor_lists_identical(train_future_xy_local_list, loaded_train_future_xy_local_list, "array")
identical4 = are_tensor_lists_identical(val_img_tensor_list, loaded_val_img_tensor_list, "tensor")
identical5 = are_tensor_lists_identical(val_agent_state_vector_list, loaded_val_agent_state_vector_list, "tensor")
identical6 = are_tensor_lists_identical(val_future_xy_local_list, loaded_val_future_xy_local_list, "array")
print("Are the two lists of tensors identical?", identical1)
print("Are the two lists of tensors identical?", identical2)
print("Are the two lists of tensors identical?", identical3)
print("Are the two lists of tensors identical?", identical4)
print("Are the two lists of tensors identical?", identical5)
print("Are the two lists of tensors identical?", identical6)

