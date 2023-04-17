from nuscenes import NuScenes

import numpy as np

from nuscenes.prediction.models.backbone import ResNetBackbone

import torch

from nuscenes.eval.prediction.splits import get_prediction_challenge_split

#Maybe
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
# %matplotlib inline

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

from nuscenes.prediction import PredictHelper

import cv2

from collections import Counter
import math

def get_and_format_data(
        version: str = 'v1.0-mini',
        DATAROOT: str = 'data/sets/nuscenes',
        subset: str = 'mini_train',
        seconds_of_history_used: float = 2.0,
        sequences_per_instance: str = 'one_sequences_per_instance'
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
    
    nuscenes = NuScenes(version, dataroot=DATAROOT)
    helper = PredictHelper(nuscenes)

    data_set = get_prediction_challenge_split(subset, DATAROOT)

    instance_token_list, sample_token_list = get_instance_tokens_and_sample_tokens(data_set)

    instance_token_list, sample_token_list = remove_short_sequences(seconds_of_history_used, instance_token_list, sample_token_list)

    # Choose all or one sequence per instance
    if sequences_per_instance == "one_sequences_per_instance":
        instance_token_list, sample_token_list = extract_one_instance_per_sequence(seconds_of_history_used, instance_token_list, sample_token_list)
    elif sequences_per_instance == "all_sequences_per_instance":
        instance_token_list, sample_token_list = extract_all_instances_per_sequence(seconds_of_history_used, instance_token_list, sample_token_list)
    
    img_list, agent_state_vector_list, future_xy_local_list = get_data_and_ground_truth(nuscenes, helper, seconds_of_history_used, instance_token_list, sample_token_list)

    img_tensor = create_img_tensor(img_list)

    return img_list, img_tensor, agent_state_vector_list, future_xy_local_list

def get_instance_tokens_and_sample_tokens(
        data_set: List[str]
) -> Tuple[List[str], List[str]]:
    """Function for creating instance token list and sample token list
    from prediction challenge list.

    Parameters
    ----------
    data_set : List[str]
        The data set list.

    Returns
    -------
    token_lists
        token_lists for instances and samples.
    """
   

    instance_token_list = [None] * len(data_set)
    sample_token_list = [None] * len(data_set)

    for index, value in enumerate(data_set):
        instance_token_list[index], sample_token_list[index] = value.split("_")

    # print(data_set)
    # print(instance_token_list)
    # print(sample_token_list)
    # unique_instance_tokens = len(set(instance_token_list)) #Daniel
    # unique_sample_tokens = len(set(sample_token_list)) #Daniel
    # print(f"""After get_instance_tokens_and_sample_tokens: 
    #     unique_instance_tokens = {unique_instance_tokens}
    #     unique_sample_tokens = {unique_sample_tokens}""") #Daniel
    

    return instance_token_list, sample_token_list


def get_data_and_ground_truth(
        nuscenes: NuScenes,
        helper: PredictHelper,
        seconds_of_history_used: float,
        instance_token_list: List[str],
        sample_token_list: List[str]
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
    """Function for creating data lists and ground truth list.

    Parameters
    ----------
    nuscenes: NuScenes,
        NuScenes instance.
    helper: PredictHelper,
        PredictHelper instance.
    seconds_of_history_used: float,
        Seconds of privious agent positions (2 per second) in images 
    instance_token_list: List[str],
        The list of instance tokens.
    sample_token_list: List[str]
        The list of sample tokens.

    Returns
    -------
    img_list
        The imput data, part 1: list containing the images for all instances and samples.
    agent_state_vector_list
        The imput data, part 2: list containing the agent state vector for all instances and samples.
    future_for_agent_list
        The ground truths: list of the (12) future position values for all instances and samples.
    """

    static_layer_rasterizer = StaticLayerRasterizer(helper)
    agent_rasterizer = AgentBoxesWithFadedHistory(helper, seconds_of_history = seconds_of_history_used) # Hannes (seconds_of_history)
    mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

    length_of_lists = len(instance_token_list)

    img_list = [None] * length_of_lists # model input, part 1
    agent_state_vector_list = [None] * length_of_lists # model input, part 2
    future_xy_local_list = [None] * length_of_lists # ground truths
    
    for index in range(length_of_lists):
        # instance_token_img, sample_token_img = 'bc38961ca0ac4b14ab90e547ba79fbb6', '7626dde27d604ac28a0240bdd54eba7a'
        instance_token = instance_token_list[index]
        sample_token = sample_token_list[index]

        anns = [ann for ann in nuscenes.sample_annotation if ann['instance_token'] == instance_token]
        img_list[index] = mtp_input_representation.make_input_representation(instance_token, sample_token)

        agent_state_vector_list[index] = torch.Tensor([[helper.get_velocity_for_agent(instance_token, sample_token),
                                    helper.get_acceleration_for_agent(instance_token, sample_token),
                                    helper.get_heading_change_rate_for_agent(instance_token, sample_token)]])
        
        future_xy_local_list[index] = helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True) # TODO: true/false


    return img_list, agent_state_vector_list, future_xy_local_list

def create_video(image_list, output_filename, fps=2):

    # # Example input to this function:
    # create_video(img_list, 'output_video_2fps_dt2_50.avi')

    height, width, _ = image_list[0].shape
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    out = cv2.VideoWriter(output_filename, fourcc, fps, size)

    for image in image_list:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        out.write(img)

    out.release()


def remove_short_sequences(
        seconds_of_history_used: float,
        instance_token_list: List[str],
        sample_token_list: List[str]
) -> Tuple[List[str], List[str]]:
    """Function for removing sequences shorter than the choosen seconds of history.
    This is to ensure that all images used later contains that right amout of agent history.

    Parameters
    ----------
    seconds_of_history_used: float,
        Seconds of privious agent positions (2 per second) in images 
    instance_token_list: List[str],
        The list of instance tokens.
    sample_token_list: List[str]
        The list of sample tokens.

    Returns
    -------
    token_lists
        updated (shortened) token_lists for instances and samples.
    """
    unique_instance_tokens = len(set(instance_token_list)) #Daniel
    unique_sample_tokens = len(set(sample_token_list)) #Daniel
    print(f"""Before remove_short_sequences:
        unique_instance_tokens = {unique_instance_tokens}
        unique_sample_tokens = {unique_sample_tokens}""") #Daniel
    
     # Initialize an empty list to store the indices of items with a count greater than 2*seconds_of_history_used - 1
    selected_indices = []

    # Count the occurrences of each string in the list
    string_counts = Counter(instance_token_list)

    # Iterate over the list and save the index if the count of the item is greater than 2*seconds_of_history_used - 1
    for index, item in enumerate(instance_token_list):
        if string_counts[item] > (int(2*seconds_of_history_used) - 1): # One history is always avalible 
            selected_indices.append(index)

    filtered_instance_tokens = []
    filtered_sample_tokens = []
    [filtered_instance_tokens.append(instance_token_list[index]) for index in selected_indices]
    [filtered_sample_tokens.append(sample_token_list[index]) for index in selected_indices]
    
    unique_instance_tokens = len(set(filtered_instance_tokens)) #Daniel
    unique_sample_tokens = len(set(filtered_sample_tokens)) #Daniel
    print(f"""After remove_short_sequences:
        unique_instance_tokens = {unique_instance_tokens}
        unique_sample_tokens = {unique_sample_tokens}""") #Daniel
    

    return filtered_instance_tokens, filtered_sample_tokens


def extract_all_instances_per_sequence(
        seconds_of_history_used: float,
        instance_token_list: List[str],
        sample_token_list: List[str]
) -> Tuple[List[str], List[str]]:
    """Function for extracting one instance per sequence.
    It should select the frame with the 

    Parameters
    ----------
    seconds_of_history_used: float,
        Seconds of privious agent positions (2 per second) in images 
    instance_token_list: List[str],
        The list of instance tokens.
    sample_token_list: List[str]
        The list of sample tokens.

    Returns
    -------
    token_lists
        token_lists for instances and samples containing one sample per sequence.
    """
    
    # Initialize an empty list to store the indices of selected items
    selected_indices = []

    # Initialize an empty list to store the indices of encoutered items
    encouteredItems = []
    
    # Instance token list to manipulate
    tmp_instance_token_list = instance_token_list.copy()
        
    # Iterate over the list and save the fouth index of each item
    for index, item in enumerate(instance_token_list):

        # Count the occurrences of each instance in the list
        instance_counts = Counter(tmp_instance_token_list)
        
        if not encouteredItems.__contains__(item):
            if instance_counts[item] > (int(2*seconds_of_history_used) - 1):
                selected_indices.append(index + int(2*seconds_of_history_used)-1)
                tmp_instance_token_list.remove(item)
            else:
                encouteredItems.append(item) 
            

    filtered_instance_tokens = []
    filtered_sample_tokens = []
    [filtered_instance_tokens.append(instance_token_list[index]) for index in selected_indices]
    [filtered_sample_tokens.append(sample_token_list[index]) for index in selected_indices]
    
    unique_instance_tokens = len(set(filtered_instance_tokens)) #Daniel
    unique_sample_tokens = len(set(filtered_sample_tokens)) #Daniel
    print(f"""After extract_all_instances_per_sequence:
        unique_instance_tokens = {unique_instance_tokens}
        unique_sample_tokens = {unique_sample_tokens}""") #Daniel

    return filtered_instance_tokens, filtered_sample_tokens


def extract_one_instance_per_sequence(
        seconds_of_history_used: float,
        instance_token_list: List[str],
        sample_token_list: List[str]
) -> Tuple[List[str], List[str]]:
    """Function for extracting one instance per sequence.
    It should select the frame with the 

    Parameters
    ----------
    seconds_of_history_used: float,
        Seconds of privious agent positions (2 per second) in images 
    instance_token_list: List[str],
        The list of instance tokens.
    sample_token_list: List[str]
        The list of sample tokens.

    Returns
    -------
    token_lists
        token_lists for instances and samples containing one sample per sequence.
    """
    
    # Initialize an empty list to store the indices of selected items
    selected_indices = []

    # Initialize an empty list to store the indices of encoutered items
    encouteredItems = []

    # Iterate over the list and save the fouth index of each item
    for index, item in enumerate(instance_token_list):

        if not encouteredItems.__contains__(item):
            selected_indices.append(index + int(2*seconds_of_history_used)-1)
            encouteredItems.append(item) 

            
    filtered_instance_tokens = []
    filtered_sample_tokens = []
    [filtered_instance_tokens.append(instance_token_list[index]) for index in selected_indices]
    [filtered_sample_tokens.append(sample_token_list[index]) for index in selected_indices]
    
    unique_instance_tokens = len(set(filtered_instance_tokens)) #Daniel
    unique_sample_tokens = len(set(filtered_sample_tokens)) #Daniel
    print(f"""After extract_one_instance_per_sequence:
        unique_instance_tokens = {unique_instance_tokens}
        unique_sample_tokens = {unique_sample_tokens}""") #Daniel

    return filtered_instance_tokens, filtered_sample_tokens


def create_img_tensor(img_list: List[np.ndarray]):
    image_tensor = []
    for index, item in enumerate(img_list):
        image_tensor.append(torch.Tensor(item).permute(2, 0, 1).unsqueeze(0))
    
    return image_tensor


########################################################################################################################
def test_utilsH_functions():
    # Test get_instance_tokens_and_sample_tokens function
    data_set = ["instance1_sample1", "instance2_sample2"]
    instance_tokens, sample_tokens = get_instance_tokens_and_sample_tokens(data_set)
    assert instance_tokens == ["instance1", "instance2"], "get_instance_tokens_and_sample_tokens failed: instance tokens do not match"
    assert sample_tokens == ["sample1", "sample2"], "get_instance_tokens_and_sample_tokens failed: sample tokens do not match"

    # Test remove_short_sequences function
    instance_token_list = ['instance1', 'instance1', 'instance1', 'instance2', 'instance2']
    sample_token_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5']
    seconds_of_history_used = 1.5
    filtered_instance_tokens, filtered_sample_tokens = remove_short_sequences(seconds_of_history_used, instance_token_list, sample_token_list)
    assert filtered_instance_tokens == ['instance1', 'instance1', 'instance1'], "remove_short_sequences failed: instance tokens do not match"
    assert filtered_sample_tokens == ['sample1', 'sample2', 'sample3'], "remove_short_sequences failed: sample tokens do not match"

    # Test extract_one_instance_per_sequence function
    instance_token_list = ['instance1', 'instance1', 'instance2', 'instance2', 'instance3', 'instance3', 'instance3']
    sample_token_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5', 'sample6', 'sample7']
    seconds_of_history_used = 1.0
    extracted_instance_tokens, extracted_sample_tokens = extract_one_instance_per_sequence(seconds_of_history_used, instance_token_list, sample_token_list)
    assert extracted_instance_tokens == ['instance1', 'instance2', 'instance3'], "extract_one_instance_per_sequence failed: instance tokens do not match"
    assert extracted_sample_tokens == ['sample2', 'sample4', 'sample6'], "extract_one_instance_per_sequence failed: sample tokens do not match"

    # # Test get_data_and_ground_truth function # TODO: funkar ej 
    # # nuscenes = MockNuScenes() # Assuming you have a MockNuScenes class to simulate the behavior of the NuScenes class
    # nuscenes = NuScenes('v1.0-mini', dataroot='data/sets/nuscenes')
    # # helper = MockPredictHelper()  # Assuming you have a MockPredictHelper class to simulate the behavior of the PredictHelper class
    # helper = PredictHelper(nuscenes)
    # seconds_of_history_used = 1.5
    # instance_token_list = ['instance1']
    # sample_token_list = ['sample1']
    # img_list, agent_state_vector_list, future_xy_local_list = get_data_and_ground_truth(nuscenes, helper, seconds_of_history_used, instance_token_list, sample_token_list)
    # assert img_list == ['image1'], "get_data_and_ground_truth failed: img_list does not match"
    # assert agent_state_vector_list == ['agent_state_vector1'], "get_data_and_ground_truth failed: agent_state_vector_list does not match"
    # assert future_xy_local_list == ['future_xy_local1'], "get_data_and_ground_truth failed: future_xy_local_list does not match"

    # # Test create_img_tensor function
    # img_list = ['image1']
    # img_tensor_list = create_img_tensor(img_list)
    # assert img_tensor_list == ['img_tensor1'], "create_img_tensor failed: img_tensor_list does not match"
    # Test create_img_tensor function # TODO: fixa denna?
    mock_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)  # Create a random 64x64 RGB image
    img_list = [mock_image]
    img_tensor_list = create_img_tensor(img_list)
    assert len(img_tensor_list) == 1, "create_img_tensor failed: img_tensor_list length does not match"

    # # Test get_and_format_data function # TODO: funkar ej 
    # version = "v1.0-mini"
    # DATAROOT = "data/sets/nuscenes"
    # subset = "mini_train"
    # seconds_of_history_used = 1.5
    # img_list, img_tensor_list, agent_state_vector_list, future_xy_local_list = get_and_format_data(version, DATAROOT, subset, seconds_of_history_used)
    # assert img_list == ['image1'], "get_and_format_data failed: img_list does not match"
    # assert img_tensor_list == ['img_tensor1'], "get_and_format_data failed: img_tensor_list does not match"
    # assert agent_state_vector_list == ['agent_state_vector1'], "get_and_format_data failed: agent_state_vector_list does not match"
    # assert future_xy_local_list == ['future_xy_local1'], "get_and_format_data failed: future_xy_local_list does not match"

    
    
#######################################################################################################
 
# Code written by Freddy Boulton, Tung Phan 2020.
from typing import List, Tuple, Callable, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as f

from nuscenes.prediction.models.backbone import calculate_backbone_feature_dim

# Number of entries in Agent State Vector
ASV_DIM = 3


class CoverNetNoRelu(nn.Module):
    """ Implementation of CoverNet https://arxiv.org/pdf/1911.10298.pdf """

    def __init__(self, backbone: nn.Module, num_modes: int,
                 n_hidden_layers: List[int] = None,
                 input_shape: Tuple[int, int, int] = (3, 500, 500)):
        """
        Inits Covernet.
        :param backbone: Backbone model. Typically ResNetBackBone or MobileNetBackbone
        :param num_modes: Number of modes in the lattice
        :param n_hidden_layers: List of dimensions in the fully connected layers after the backbones.
            If None, set to [4096]
        :param input_shape: Shape of image input. Used to determine the dimensionality of the feature
            vector after the CNN backbone.
        """

        if n_hidden_layers and not isinstance(n_hidden_layers, list):
            raise ValueError(f"Param n_hidden_layers must be a list. Received {type(n_hidden_layers)}")

        super().__init__()

        if not n_hidden_layers:
            n_hidden_layers = [4096]

        self.backbone = backbone

        backbone_feature_dim = calculate_backbone_feature_dim(backbone, input_shape)
        n_hidden_layers = [backbone_feature_dim + ASV_DIM] + n_hidden_layers + [num_modes]

        linear_layers = [nn.Linear(in_dim, out_dim)
                         for in_dim, out_dim in zip(n_hidden_layers[:-1], n_hidden_layers[1:])]

        self.head = nn.ModuleList(linear_layers)
#         self.relu = nn.ReLU()

    def forward(self, image_tensor: torch.Tensor,
                agent_state_vector: torch.Tensor) -> torch.Tensor:
        """
        :param image_tensor: Tensor of images in the batch.
        :param agent_state_vector: Tensor of agent state vectors in the batch
        :return: Logits for the batch.
        """

        backbone_features = self.backbone(image_tensor)

        logits = torch.cat([backbone_features, agent_state_vector], dim=1)

        for linear in self.head:
#             logits = self.relu(linear(logits))
            logits = linear(logits)

        return logits


def mean_pointwise_l2_distance(lattice: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Computes the index of the closest trajectory in the lattice as measured by l1 distance.
    :param lattice: Lattice of pre-generated trajectories. Shape [num_modes, n_timesteps, state_dim]
    :param ground_truth: Ground truth trajectory of agent. Shape [1, n_timesteps, state_dim].
    :return: Index of closest mode in the lattice.
    """
    stacked_ground_truth = ground_truth.repeat(lattice.shape[0], 1, 1)
    return torch.pow(lattice - stacked_ground_truth, 2).sum(dim=2).sqrt().mean(dim=1).argmin()


class ConstantLatticeLoss:
    """
    Computes the loss for a constant lattice CoverNet model.
    """

    def __init__(self, lattice: Union[np.ndarray, torch.Tensor],
                 similarity_function: Callable[[torch.Tensor, torch.Tensor], int] = mean_pointwise_l2_distance):
        """
        Inits the loss.
        :param lattice: numpy array of shape [n_modes, n_timesteps, state_dim]
        :param similarity_function: Function that computes the index of the closest trajectory in the lattice
            to the actual ground truth trajectory of the agent.
        """

        self.lattice = torch.Tensor(lattice)
        self.similarity_func = similarity_function

    def __call__(self, batch_logits: torch.Tensor, batch_ground_truth_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss on a batch.
        :param batch_logits: Tensor of shape [batch_size, n_modes]. Output of a linear layer since this class
            uses nn.functional.cross_entropy.
        :param batch_ground_truth_trajectory: Tensor of shape [batch_size, 1, n_timesteps, state_dim]
        :return: Average element-wise loss on the batch.
        """

        # If using GPU, need to copy the lattice to the GPU if haven't done so already
        # This ensures we only copy it once
        if self.lattice.device != batch_logits.device:
            self.lattice = self.lattice.to(batch_logits.device)

        batch_losses = torch.Tensor().requires_grad_(True).to(batch_logits.device)

        for logit, ground_truth in zip(batch_logits, batch_ground_truth_trajectory):

            closest_lattice_trajectory = self.similarity_func(self.lattice, ground_truth)
            label = torch.LongTensor([closest_lattice_trajectory]).to(batch_logits.device)
            classification_loss = f.cross_entropy(logit.unsqueeze(0), label)

            batch_losses = torch.cat((batch_losses, classification_loss.unsqueeze(0)), 0)

        return batch_losses.mean()
