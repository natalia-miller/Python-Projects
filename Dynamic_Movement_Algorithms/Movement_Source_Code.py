# Natalia Miller

'''
The following code has been adapted from psuedocode:
    I. Millington, Artificial Intelligence for Games, Third Edition,
    CRC Press, Boca Raton FL, 2019
''' 

import math

output_file = r"output.txt"
timestep = 0.5


#------------------------------------------------------------------------------#
#                              Geometry Functions                              #
#------------------------------------------------------------------------------#
def length(vector):
    # Calculates the length/magnitude of a vector
    length = math.sqrt((pow(vector[0],2) + pow(vector[1],2)))

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


def multiply(vector_1, vector_2):
    # Subtracts two vectors
    result = [0, 0]
    result[0] = vector_1[0] * vector_2[0]
    result[1] = vector_1[1] * vector_2[1]

    return result


def addition(vector_1, vector_2):
    # Adds two vectors
    result = [0, 0]
    result[0] = vector_1[0] + vector_2[0]
    result[1] = vector_1[1] + vector_2[1]

    return result


def scalar_multiply(vector_1, scalar):
    # Multiplies a vector by a scalar
    result = [0, 0]
    result[0] = vector_1[0] * scalar
    result[1] = vector_1[1] * scalar

    return result


def scalar_addition(vector_1, scalar):
    # Adds a scalar value each element of a vector
    result = [0, 0]
    result[0] = vector_1[0] + scalar
    result[1] = vector_1[1] + scalar

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

    return math.sqrt((pow((point_z[0] - point_x[0]), 2)) + (pow((point_z[1] - point_x[1]), 2)))


def max_which(number, vector):
    # Finds the position of elements that are less than the number
    # given and returns the largest position
    result = []

    for i in range(len(vector)):
        if number > vector[i]:
            result.append(i)
            
    return max(result)


#------------------------------------------------------------------------------#
#                                Path Operations                               #
#------------------------------------------------------------------------------#
def path_assemble(path_x, path_y):
    # Assemble path data structure
    path_segments = len(path_x) - 1
    path_distance = [0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(len(path_distance)):
        if i == 0:
            continue
        else:
            path_distance[i] = path_distance[i - 1] + distance([path_x[i - 1], path_y[i - 1]], [path_x[i], path_y[i]])

    path_param = [0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(len(path_distance)):
        path_param[i] = path_distance[i] / max(path_distance)
    
    assembled_path = {
        "x": path_x,
        "y": path_y,
        "distance": path_distance,
        "param": path_param,
        "segments": path_segments
    }

    return assembled_path


def get_param(character_path, character_position):
    # Find point on path closest to given position.  
    closest_distance = float('inf')
    closest_segment = 0 # index of closest segment
    closest_point = [0, 0] 
    path_segments = character_path["segments"]
    path_param = character_path["param"]
    path_x = character_path["x"]
    path_y = character_path["y"]

    for i in range(0, path_segments):
        endpoint_a = [path_x[i], path_y[i]]
        endpoint_b = [path_x[i + 1], path_y[i + 1]] 
        check_point = closest_point_segment(character_position, endpoint_a, endpoint_b)  
        check_distance = distance(character_position, check_point)  
        if (check_distance < closest_distance):
            closest_point = check_point  
            closest_distance = check_distance  
            closest_segment = i  
               
    # Calculate path parameter of closest point.  
    endpoint_a = [path_x[closest_segment], path_y[closest_segment]]  
    a_param = path_param[closest_segment]  
    endpoint_b = [path_x[closest_segment + 1], path_y[closest_segment + 1]]
    b_param = path_param[closest_segment + 1]  

    c_a = subtract(closest_point, endpoint_a)
    b_a = subtract(endpoint_b, endpoint_a)
    t_var = length(c_a) / length(b_a) 
    c_param = a_param + (t_var * (b_param - a_param))
    return(c_param)  


def get_position(path, target_param):
    # Calculate position on path, given target path parameter
    path_x = path["x"]
    path_y = path["y"]
    path_param = path["param"]
    position = [0, 0]

    i = (max_which(target_param, path_param))
    endpoint_a = [path_x[i], path_y[i]]
    endpoint_b = [path_x[i + 1], path_y[i + 1]]
    t_var = (target_param - path_param[i]) / (path_param[i + 1] - path_param[i])
    position = addition(endpoint_a, scalar_multiply(subtract(endpoint_b, endpoint_a), t_var))
    return(position)  
 

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
    linear_acceleration = character["linear_acceleration"]
    if not linear_acceleration:
        linear_acceleration = [0,0]
    orientation = character["orientation"]
    max_velocity = character["max_velocity"]
    character_id = character["id"]
    
    rotation = 0
    angular = [0,0]
    time = timestep

    # Update the position and orientation  
    position = addition(position, scalar_multiply(velocity, time)) 
    orientation += rotation * time  

    # Update the velocity and rotation  
    velocity = addition(velocity, scalar_multiply(linear_acceleration, time)) 
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
    if (character_id == 2701):
        character_5["position"] = position
        character_5["orientation"] = orientation
        character_5["velocity"] = velocity
        

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


def follow_path(character, character_path):
    # Calculate target to delegate to Seek
    character_position = character["position"] # Current position on the path, as a path parameter
    path_offset = character["path_offset"] # Distance farther along path to place target
	
	# Find current position on path
    currentParam = get_param(character_path, character_position)  
		
	# Offset it
    target_param = currentParam + path_offset
		
	# Get the target position
    target_position = get_position(character_path, target_param)
    target = {"position": target_position}
		
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
    "collision_status": "FALSE",
    "path_to_follow": 1,
    "path_offset": 0.04
}


path_x = [0, -20, 20, -40, 40, -60, 60, 0]
path_y = [90, 65, 40, 15, -10, -35, -60, -85]
character_path = path_assemble(path_x, path_y)


#------------------------------------------------------------------------------#
#                                 Main Method                                  #
#------------------------------------------------------------------------------#

# Print initial values
output_steering(character_5)

for x in range(125):

    # Increase timestep by 0.5
    character_5["timestep"] += timestep

    # Call the character's steering movement behavior
    character_5["linear_acceleration"] = follow_path(character_5, character_path)

    # Update the character's postion, orientation, and velocity
    steering_update(character_5)

    # Print updated values
    output_steering(character_5)
