import math
import time

import numpy as np 
from matplotlib import pyplot as plt

output_file = r"C:\Users\nmiller\OneDrive - EOS DS USA\Documents\output.txt"

#------------------------------------------------------------------------------#
#                           Character Initializations                          #
#------------------------------------------------------------------------------#
character_1 = {
    "id": 2601,
	"steering_behavior": "Continue",
    "steering_behavior_code": "1",
    "inital_position": np.array([0, 0]),
	"inital_velocity": np.array([0, 0]),
	"inital_orientation": 0,
	"max_velocity": 0,
	"max_acceleration": 0,
	"target": 0,				 
    "arrival_radius": 0,
    "slowing_radius": 0,
    "time_to_target": 0,
    "timestep": 0,
    "position_x": 0,
    "position_z": 0,
    "velocity_x": 0,
    "velocity_z": 0,
    "linear_acceleration_x": 0,
    "linear_acceleration_z": 0,
    "orientation": 0,
    "collision_status": False,
    "position": np.array([0, 0])
}


character_2 = {
	"id": 2602,
	"steering_behavior": "Flee",
  "steering_behavior_code": "7",
	"inital_position": np.array([-30, -50]),
	"inital_velocity": np.array([2.0, 7.0]),
	"inital_orientation": math.radians(45),
	"max_velocity": 8,
	"max_acceleration": 1.5,
    "target": 1,				 
    "arrival_radius": 0,
    "slowing_radius": 0,
    "time_to_target": 0,		
    "timestep": 0,
    "position_x":-30,
    "position_z": -50,
    "velocity_x": 2,
    "velocity_z": 7,
    "linear_acceleration_x": 0,
    "linear_acceleration_z": 0,
    "orientation": math.radians(45),
    "collision_status": False,
    "position": (-30, -50),
    "velocity": (2, 7)
}


character_3 = {
	"id": 2603,
	"steering_behavior": "Seek",
  "steering_behavior_code": "6",
	"inital_position": (-50, 40),
	"inital_velocity": (0, 8),
	"inital_orientation": math.radians(270),
	"max_velocity": 8,
	"max_acceleration": 2,
	"target": 1,				 
  "arrival_radius": 0,
  "slowing_radius": 0,
  "time_to_target": 0,
  "timestep": 0,
  "position_x": -50,
  "position_z": 40,
  "velocity_x": 0,
  "velocity_z": 8,
  "linear_acceleration_x": 0,
  "linear_acceleration_z": 0,
  "orientation": math.radians(270),
  "collision_status": False
}


character_4 = {
	"id": 2604,
	"steering_behavior": "Arrive",
  "steering_behavior_code": "8",
	"inital_position": (50, 75),
	"inital_velocity": (-9, 4),
	"inital_orientation": math.radians(180),
	"max_velocity": 10,
	"max_acceleration": 2,
	"target": 1,				 
  "arrival_radius": 4,
  "slowing_radius": 32,
  "time_to_target": 1,
  "timestep": 0,
  "position_x": 50,
  "position_z": 75,
  "velocity_x": -9,
  "velocity_z": 4,
  "linear_acceleration_x": 0,
  "linear_acceleration_z": 0,
  "orientation": math.radians(180),
  "collision_status": False			 
}


#------------------------------------------------------------------------------#
#                               Vector Functions                               #
#------------------------------------------------------------------------------#
def normalize(vector):
  '''
   Normalizes a vector using the NumPy module
  '''
  normalized_vector = vector / np.linalg.norm(vector)
  return normalized_vector


#------------------------------------------------------------------------------#
#                              Steering Behaviors                              #
#------------------------------------------------------------------------------#
def steering_continue(character):
  # Continue moving without changing velocity or orientation
  result = [character["velocity_x"], character["velocity_z"], character["orientation"]]
  return result


def get_steering_seek(character, target):
    character_position = character["inital_position"]
    # kinematic_orientation = character["inital_orientation"]
    target_postion = target["inital_position"]
    max_acceleration = character["max_acceleration"]
    linear_result = character["position"]
    angular_result = character["velocity"]

    # Get the direction to the target
    linear_result = np.subtract(target_postion, character_position)

    # Accelerate at maximum rate
    linear_result = normalize(linear_result)
    linear_result = linear_result * max_acceleration

    # Output steering
    angular_result = [0, 0]    
    return linear_result, angular_result


def output_steering(character):
    with open(output_file, "a") as f:
        print('{}, '.format(character["timestep"]) +
            '{}, '.format(character["id"]) +
            '{}, '.format(character["position_x"]) +
            '{}, '.format(character["position_z"]) +
            '{}, '.format(character["velocity_x"]) +
            '{}, '.format(character["velocity_z"]) +
            '{}, '.format(character["linear_acceleration_x"]) +
            '{},'.format(character["linear_acceleration_z"]) +
            '{}, '.format(character["orientation"]) +
            '{}, '.format(character["steering_behavior_code"]) +
            '{}, '.format(character["collision_status"]), file = f)


#------------------------------------------------------------------------------#
#                                 Main Method                                  #
#------------------------------------------------------------------------------#

# Print initial values
output_steering(character_1)
output_steering(character_2)

for x in range(99):
    
    character_1["timestep"] += 0.5
    character_2["timestep"] += 0.5

    # Run first methods
    character_1["postion"] = steering_continue(character_1)
    character_2["postion"], character_2["velocity"] = get_steering_seek(character_2, character_1)

    # Print updated values
    output_steering(character_1)
    output_steering(character_2)
