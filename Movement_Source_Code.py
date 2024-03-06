# Natalia Miller

'''
The following code has been adapted from psuedocode:
    I. Millington, Artificial Intelligence for Games, Third Edition,
    CRC Press, Boca Raton FL, 2019
''' 

import math
import numpy as np

output_file = r"output.txt"
timestep = 0.5


#------------------------------------------------------------------------------#
#                              Geometry Functions                              #
#------------------------------------------------------------------------------#
def length(vector):
    # Calculates the length/magnitude of a vector
    length = math.sqrt((vector[0]*vector[0]) + (vector[1]*vector[1]))
    return length


def normalize(vector):
    # Normalizes a vector
    magnitude = length(vector)

    if (magnitude != 0):
        normalized_x = vector[0] / magnitude
        normalized_z = vector[1] / magnitude
        return [normalized_x, normalized_z]
    else:
        return [0,0]


def subtract(vector_1, vector_2):
    # Subtracts two vectors
    result = [0, 0]
    result[0] = vector_1[0] - vector_2[0]
    result[1] = vector_1[1] - vector_2[1]
    return result


def scalar_multiply(vector_1, scalar):
    # Multiplies a vector by a scalar
    result = [0, 0]
    result[0] = vector_1[0] * scalar
    result[1] = vector_1[1] * scalar
    return result


def dot(vector_1, vector_2):
    # Calculates the dot product of two vectors
    result = (vector_1[0] * vector_2[0]) + (vector_1[1] * vector_2[1])
    return result


def closest_point(query_point, point_a, point_b):
    # Find point on segement closest to query point
    q_a = subtract(query_point, point_a)
    b_a = subtract(point_b, point_a)
    vector_dot_1 = dot(q_a, b_a)
    vector_dot_2 = dot(b_a, b_a)
    t = vector_dot_1 / vector_dot_2
    var_1 = t * b_a[0]
    var_2 = t * b_a[1]
    return [point_a[0] + var_1, point_a[1] + var_2] 


def closest_point_segment(query_point, point_a, point_b):
    # Find point on segment closest to query point in 2D
    q_a = subtract(query_point, point_a)
    b_a = subtract(point_b, point_a)
    vector_dot_1 = dot(q_a, b_a)
    vector_dot_2 = dot(b_a, b_a)
    t = vector_dot_1 / vector_dot_2

    if t <= 0:
        return point_a
    elif t >= 1:
        return point_b
    else:
        var_1 = t * b_a[0]
        var_2 = t * b_a[1]
        return [point_a[0] + var_1, point_a[1] + var_2] 


def distance(point_x, point_z):
    # Find the distance between two points
    x_start = point_x[0]
    x_end = point_x[1]
    z_start = point_z[0]
    z_end = point_z[1]

	# dist = numpy.linalg.norm(a-b)
    return math.sqrt((pow((x_end - x_start), 2)) + (pow((z_end - z_start), 2)))


#------------------------------------------------------------------------------#
#                                Path Operations                               #
#------------------------------------------------------------------------------#
def get_position(path_x, path_y, param):   
    val_x = np.max(np.where(param[0] > path_x)) 
    val_y = np.max(np.where(param[1] > path_y)) 
    a_path_param = np.array([np.where(path_x == val_x), np.where(path_y == val_y)])  
    b_path_param = np.array([np.where(path_x == val_x) + 1, np.where(path_y == val_y) + 1]) 

    difference_1 = subtract(param, a_path_param)
    difference_2 = subtract(b_path_param, a_path_param)

    t_var = np.divide(difference_1, difference_2)
          
    position = a_path_param + np.multiply(t_var, difference_2)

    return(position)  


def get_param(path_x, path_y, position):
    # Find point on path closest to given position
    closest_distance = float('inf')
    closest_point = 0
    closest_segment = 0

    for index, x in enumerate(path_x):
        print(index)
        a_path = np.array([path_x[index], path_y[index]])

        if index != 8:
            b_path = np.array([path_x[index], path_y[index]])

        check_point = closest_point_segment(position, a_path, b_path) 
        check_distance = distance(position, check_point)  
    
        if check_distance < closest_distance:
            closest_distance = check_point
            closest_distance = check_distance
            closest_segment = index
	
    # Calculate path parameter of closest point.  
    a_path_param = np.array([np.where(path_x == closest_segment[0]), np.where(path_y == closest_segment[1])])
    b_path_param = np.array([np.where(path_x == (closest_segment[0] + 1)), np.where(path_y == (closest_segment[1] + 1))])
    
    c_path_param = closest_point  

    difference_1 = subtract(c_path_param, a_path_param)
    difference_2 = subtract(b_path_param, a_path_param)

    t_var = np.divide(length(difference_1), length(difference_2))  
    c_param = a_path_param + np.multiply(t_var, difference_2)  
     
    return(c_param)  


#------------------------------------------------------------------------------#
#                              Movement Functions                              #
#------------------------------------------------------------------------------#
def steering_update(character):
    ''' Updates character’s movement variables
        - Outputs: New values for position, orientation, velocity, rotation
        - Inputs: linear and angular accelerations
        - Inputs generated by movement behaviors
    '''
    position = character["position"]
    if not position:
        position = [0,0]
    velocity = character["velocity"]
    if not velocity:
        velocity = [0,0]
    orientation = character["orientation"]
    rotation = 0
    angular = [0,0]
    max_velocity = character["max_velocity"]
    time = timestep
    character_id = character["id"]
    linear_acceleration = character["linear_acceleration"]
    if not linear_acceleration:
        linear_acceleration = [0,0]

    # Update the position and orientation  
    position[0] += velocity[0] * time  
    position[1] += velocity[1] * time  
    orientation += rotation * time  

    # Update the velocity and rotation  
    velocity[0] += linear_acceleration[0] * time  
    velocity[1] += linear_acceleration[1] * time  
    rotation += angular[0] * time  

    # Check for speed above max and clip 
    if length(velocity) > max_velocity:
        normalized_velocity = normalize(velocity)
        normalized_velocity = scalar_multiply(normalized_velocity, max_velocity)
        velocity = normalized_velocity
    
    # Update character position, orientation, and velocity
    if (character_id == 2601):
        character_1["position"] = position
        character_1["orientation"] = orientation
        character_1["velocity"] = velocity
    if (character_id == 2602):
        character_2["position"] = position
        character_2["orientation"] = orientation
        character_2["velocity"] = velocity
    if (character_id == 2603):
        character_3["position"] = position
        character_3["orientation"] = orientation
        character_3["velocity"] = velocity
    if (character_id == 2604):
        character_4["position"] = position
        character_4["orientation"] = orientation
        character_4["velocity"] = velocity


def get_steering_continue(character):
    # Continue moving without changing velocity or orientation
    result = character["velocity"]
    return result


def get_steering_seek(character, target):
    character_position = character["position"]
    target_position = target["position"]
    max_acceleration = character["max_acceleration"]
    linear_result = [0, 0]

    # Get the direction to the target
    linear_result = subtract(target_position, character_position)

    # Accelerate at maximum rate
    linear_result = normalize(linear_result)
    linear_result = scalar_multiply(linear_result, max_acceleration)

    # Output steering
    return linear_result


def get_steering_flee(character, target):
    character_position = character["position"]
    target_position = target["position"]
    max_acceleration = character["max_acceleration"]
    linear_result = [0, 0]

    # Get the direction to the target
    linear_result = subtract(character_position, target_position)

    # Accelerate at maximum rate
    linear_result = normalize(linear_result)
    linear_result = scalar_multiply(linear_result, max_acceleration)

    # Output steering
    return linear_result


def get_steering_arrive(character, target):
    character_position = character["position"]
    target_position = target["position"]
    max_acceleration = character["max_acceleration"]
    linear_result = [0, 0]
    target_radius = character["arrival_radius"]
    slow_radius = character["slowing_radius"]
    time_to_target = character["time_to_target"]
    character_velocity = character["velocity"]
    target_speed = 0
    max_speed = character["max_velocity"]
    direction = subtract(target_position, character_position)
    distance = length(direction)
    
    # Test for arrival
    if distance < target_radius:
        return None

    # Outside slowing-down (outer) radius, move at max speed
    if distance > slow_radius:
        target_speed = character["max_velocity"]
    # Between radii, scale speed to slow down
    else:
        target_speed = max_speed * distance / slow_radius

    # Target velocity combines speed and direction
    target_velocity = normalize(direction)
    target_velocity = scalar_multiply(target_velocity, target_speed)

    # Accelerate to target velocity
    linear_result = subtract(target_velocity, character_velocity)
    linear_result[0] = linear_result[0] / time_to_target
    linear_result[1] = linear_result[1] / time_to_target

    # Test for too fast acceleration
    if length(linear_result) > max_acceleration:
        linear_result = normalize(linear_result)
        linear_result = scalar_multiply(linear_result, max_acceleration)

    # Output steering
    return linear_result


def follow_path(character, character_path_x, character_path_y):
    # Calculate target to delegate to Seek
    character_position = character["position"] # Current position on the path, as a path parameter
    path_offset = character["path_offset"] # Distance farther along path to place target
	
	# Find current position on path
    currentParam = get_param(character_path_x, character_path_y, character_position)  
		
	# Offset it
    target_param = min(1, currentParam + path_offset)
		
	# Get the target position
    target_position = get_position(character_path_x, character_path_y, target_param)

    target = {"target_position": target_position}
		
	# Delegate to seek
    return get_steering_seek(character, target) # Delegate offset target to seek


def output_steering(character):
    position = character["position"]
    velocity = character["velocity"]
    if not velocity: 
        velocity = [0, 0]
    linear_acceleration = character["linear_acceleration"]
    if not linear_acceleration: 
        linear_acceleration = [0, 0]
    
    # Output formatted information to file
    with open(output_file, "a") as f:
        print('{},'.format(character["timestep"]) +
            '{},'.format(character["id"]) +
            '{},'.format(position[0]) +
            '{},'.format(position[1]) +
            '{},'.format(velocity[0]) +
            '{},'.format(velocity[1]) +
            '{},'.format(linear_acceleration[0]) +
            '{},'.format(linear_acceleration[1]) +
            '{},'.format(character["orientation"]) +
            '{},'.format(character["steering_behavior_code"]) +
            '{}'.format(character["collision_status"]), file = f)


#------------------------------------------------------------------------------#
#                           Character Initializations                          #
#------------------------------------------------------------------------------#
character_1 = {
    "id": 2601,
	"steering_behavior": "Continue",
    "steering_behavior_code": "1",
    "position": [0, 0],
	"velocity": [0, 0],
	"orientation": 0,
	"max_velocity": 0,
	"max_acceleration": 0,
	"target": 0,				 
    "arrival_radius": 0,
    "slowing_radius": 0,
    "time_to_target": 0,
    "timestep": 0,
    "linear_acceleration": [0, 0],
    "collision_status": False,
}


character_2 = {
	"id": 2602,
	"steering_behavior": "Flee",
    "steering_behavior_code": "7",
	"position": [-30, -50],
	"velocity": [2, 7],
	"orientation": math.radians(45),
	"max_velocity": 8,
	"max_acceleration": 1.5,
    "target": 1,				 
    "arrival_radius": 0,
    "slowing_radius": 0,
    "time_to_target": 0,		
    "timestep": 0,
    "linear_acceleration": [0, 0],
    "collision_status": False,
}


character_3 = {
	"id": 2603,
	"steering_behavior": "Seek",
    "steering_behavior_code": "6",
	"position":[-50, 40],
	"velocity": [0, 8],
	"orientation": math.radians(270),
	"max_velocity": 8,
	"max_acceleration": 2,
	"target": 1,				 
    "arrival_radius": 0,
    "slowing_radius": 0,
    "time_to_target": 0,
    "timestep": 0,
    "linear_acceleration": [0, 0],
    "collision_status": False,
}


character_4 = {
	"id": 2604,
	"steering_behavior": "Arrive",
    "steering_behavior_code": "8",
	"position": [50, 75],
	"velocity": [-9, 4],
	"orientation": math.radians(180),
	"max_velocity": 10,
	"max_acceleration": 2,
	"target": 1,				 
    "arrival_radius": 4,
    "slowing_radius": 32,
    "time_to_target": 1.0,
    "timestep": 0,
    "linear_acceleration": [0, 0],
    "collision_status": False			 
}


character_5 = {
    "id": 2701,
	"steering_behavior": "Follow path",
    "steering_behavior_code": "11",
    "position": [20, 95],
	"velocity": [0, 0],
	"orientation": 0,
	"max_velocity": 4,
	"max_acceleration": 2,
	"target": 0,				 
    "arrival_radius": 0,
    "slowing_radius": 0,
    "time_to_target": 0,
    "timestep": 0,
    "linear_acceleration": [0, 0],
    "collision_status": False,
    "path_to_follow": 1,
    "path_offset": 0.04
}


character_path_x = [0, -20, 20, -40, 40, -60, 60, 0]

character_path_y = [90, 65, 40, 15, -10, -35, -60, -85]


#------------------------------------------------------------------------------#
#                                 Main Method                                  #
#------------------------------------------------------------------------------#

# Print initial values
output_steering(character_5)

for x in range(100):

    # Increase timestep by 0.5
    character_5["timestep"] += timestep

    # Call the character's steering movement behavior
    character_5["linear_acceleration"] = follow_path(character_5, character_path_x, character_path_x)

    # Update the character's postion, orientation, and velocity
    steering_update(character_5)

    # Print updated values
    output_steering(character_5)
