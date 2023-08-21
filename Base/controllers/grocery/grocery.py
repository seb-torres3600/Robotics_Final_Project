"""grocery controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard, CameraRecognitionObject
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d 

#Initialization
print("=== Initializing Grocery Shopper...")
#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 14
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.0, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.0,0.0)
new_pos = [0.0, 0.0, 0.0, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.0,0.0]
robot_parts=[]

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# Enable display
display = robot.getDevice("display")

# Odometry
pose_x     = -12.3
pose_y     = 4.55
pose_theta = 1.57

pose_y = ((1 * (pose_y + 3.82)))
pose_x = ((-1 * (pose_x - 10)))

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis
map = None


# ------------------------------------------------------------------
# Helper Functions


#mode = 'manual_arm'
mode = 'manual_mapping'
#mode = 'show_map'
#mode = 'auto_mapping'
#map = np.zeros([3000,1610],float)
if mode == 'show_map':
    map = np.load("map.npy")
    #plt.imshow(map)
    #plt.show()
    Kernel = np.ones((12,12)) # Play with this number to find something suitable, the number corresponds to the # of pixels you want to cover
    Convolved_map = convolve2d(map, Kernel, mode='same') # You still have to threshold this convolved map
    new_map = np.where(Convolved_map < .5 ,0 , 1) #threshold similar to part 1.4
    new_map_1 = np.fliplr(new_map)
    new_map_1 = np.rot90(new_map_1)
    plt.imshow(new_map_1)
    #plt.show()
    
    for i in range(360):
        for j in range(360):
            if(new_map[i][j] == 1):
                #Convolved_map[i][j] = 1
                display.setColor(0xFFFFF)
                display.drawPixel(i,j)


map = np.zeros([360,360],float)
count = 0

#Visualization
green_locations = []
green_ids = []
target_position = []

# Main Loop
while robot.step(timestep) != -1:
    count += 1
    
    pose_y = gps.getValues()[2]
    pose_x = gps.getValues()[0]
    
    pose_y = ((100 * (pose_y + 3.82)))
    pose_x = ((-100 * (pose_x - 10)))
    #print(n_pose_y, n_pose_x)
    
    map_x = int(pose_x * .12)
    map_y = int(pose_y * .12 + 80)
        
    display.setColor(int(0xFF0000))
    display.drawPixel(map_y,map_x)
    
    n = compass.getValues()
    rad = -((math.atan2(n[0], n[2]))-1.57)
    pose_theta = rad
    
    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

    if count > 50: #let lidar stabilize, was getting weird initial noise
    
        for i, rho in enumerate(lidar_sensor_readings):
            alpha = lidar_offsets[i]

            if rho > LIDAR_SENSOR_MAX_RANGE:
                continue
            rho = rho 
        #The Webots coordinate system doesn't match the robot-centric axes we're used to
            rx = math.cos(alpha)*rho *100
            ry = -math.sin(alpha)*rho * 100

        # Convert detection from robot coordinates into world coordinates
            wy =  -(math.cos(pose_theta)*rx - math.sin(pose_theta)*ry) + pose_y
            wx =  math.sin(pose_theta)*rx + math.cos(pose_theta)*ry + pose_x
            
            map_x = int(wx * .12)
            map_y = int(wy * .12 + 80)
                
            if map_x >= 360:
                map_x = 359
            if map_y >= 360:
                map_y = 359
            if map_x < 0:
                map_x = 0
            if map_y < 0:
                map_y = 0
                   
            if map[map_y][map_x] < 1:
                map[map_y][map_x] += .005     
            
            if rho < LIDAR_SENSOR_MAX_RANGE:
                tmp_val = map[map_y][map_x] 
                g = int(tmp_val * 255) # converting [0,1] to grayscale intensity [0,255] 
                color = g*256**2+g*256+g
                display.setColor(color)
                display.drawPixel(map_y,map_x)
    
    #copied from lab 5
    if mode == 'manual_mapping':
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == keyboard.LEFT :
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
        elif key == ord('S'):
            # Part 1.4: Filter map and save to filesystem
            b = np.where(map < .2,0 , 1) #if under threshold convert to 0, else 1
            with open('map.npy', 'wb') as f: #save .npy
                np.save(f, b)
            print("Map file saved")
        elif key == ord('L'):
            # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
            map = np.load("map.npy")
            print("Map loaded")
        else: # slow down
            vL *= .75
            vR *= .75
            
    if mode == 'manual_arm':
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == keyboard.LEFT:
            if(new_pos[3] < 2.60):
                robot_parts[3].setPosition(float(new_pos[3] + .01))
                new_pos[3] = new_pos[3] + .01
            else:
                print("Can't go further left")
                continue
        if key == keyboard.RIGHT:
            if new_pos[3]-.01 > 0.1:
                robot_parts[3].setPosition(float(new_pos[3] - .01))
                new_pos[3] = new_pos[3] - .01
            else:
                print("Can't go further right")
                continue
        if key == keyboard.DOWN:
            if new_pos[6]-.01 > -0.4:
                robot_parts[6].setPosition(float(new_pos[6] - .01))
                new_pos[6] = new_pos[6] - .01
            else:
                print("Can't go further down")
                continue
        if key == keyboard.UP:
            if new_pos[6]-.01 < 2.2:
                robot_parts[6].setPosition(float(new_pos[6] + .01))
                new_pos[6] = new_pos[6] + .01
            else:
                print("Can't go further up")
                continue
        if key == ord('A'):
            if new_pos[5]+.02 < 1.4:
                robot_parts[5].setPosition(float(new_pos[5] + .02))
                new_pos[5] = new_pos[5] + .02
            else:
                print("Can't go further up")
                continue
        if key == ord('D'):
            if new_pos[5]-.02 > -3.4:
                robot_parts[5].setPosition(float(new_pos[5] - .02))
                new_pos[5] = new_pos[5] - .02
            else:
                print("Can't go further up")
                continue
        if key == ord('W'):
            if new_pos[4]+.02 < 1:
                robot_parts[4].setPosition(float(new_pos[4] + .02))
                new_pos[4] = new_pos[4] + .02
            else:
                print("Can't go further up")
                continue
        if key == ord('S'):
            if new_pos[4]-.02 > -1.5:
                robot_parts[4].setPosition(float(new_pos[4] - .02))
                new_pos[4] = new_pos[4] - .02
            else:
                print("Can't go further up")
                continue
        if key == ord('E'):
            if new_pos[12] - .01 > 0:
                robot_parts[12].setPosition(float(new_pos[12] - .01))
                robot_parts[13].setPosition(float(new_pos[13] - .01))
                new_pos[12] = new_pos[12] - .01
                new_pos[13] = new_pos[13] - .01
            else:
                print("Can't go further up")
                continue
        if key == ord('Q'):
            if new_pos[12] + .01 < .05:
                robot_parts[12].setPosition(float(new_pos[12] + .01))
                robot_parts[13].setPosition(float(new_pos[13] + .01))
                new_pos[12] = new_pos[12] + .01
                new_pos[13] = new_pos[13] + .01
            else:
                print("Can't go further up")
                continue
        if key == ord('Z'):
            if new_pos[7] + .02 < 2:
                robot_parts[7].setPosition(float(new_pos[7] + .02))
                new_pos[7] = new_pos[7] + .02
            else:
                print("Can't go further up")
                continue
        if key == ord('C'):
            if new_pos[7] - .02 > -2:
                robot_parts[7].setPosition(float(new_pos[7] - .02))
                new_pos[7] = new_pos[7] - .02
            else:
                print("Can't go further up")
                continue
                
        if key == ord('R'):
            if new_pos[8] + .02 < 1.4:
                robot_parts[8].setPosition(float(new_pos[8] + .01))
                new_pos[8] = new_pos[8] + .01
            else:
                print("Can't go further up")
                continue
        if key == ord('Y'):
            if new_pos[8] - .01 > -1.4:
                robot_parts[8].setPosition(float(new_pos[8] - .01))
                new_pos[8] = new_pos[8] - .01
            else:
                print("Can't go further up")
                continue
                
    if mode == "auto_mapping":
            map_x = int(pose_x * .12)
            map_y = int(pose_y * .12 + 80)
            vL = MAX_SPEED
            vR = MAX_SPEED
    
    #------------------
    #Visualization
    #------------------
    objects = camera.getRecognitionObjects() #Get objects
    
    green_range = [[0.38, 0.66, 0.09],[0.51 ,0.78, 0.19]] #[ Green range
    
    #Loop through objects
    for object in objects:
    
        #Get object information
        id = object.get_id() 
        position = object.get_position()
        color = object.get_colors()
        
        #Loop through objects, find green ones
        for i in range(0,3): 
        #Check if object is green
            if (color[i] >= green_range[0][i] and color[i] <= green_range[1][i]):
                
                if id not in green_ids: #If object is not already accounted for
                    green_ids.append(id) #Keep track of object id 
                    green_locations.append(position) #Add to list of goal locations
                    
    target_position = green_locations[0] #Pop off when retrieved
    
    print(green_locations)
    #---------------------------
    # Manipulation
    #---------------------------
            

    #print(vL,vR)
    pose_y += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_x -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    #print(pose_theta)
    #print("X: %f Y: %f Theta: %f" % (pose_x, pose_y, pose_theta))
    
    
    robot.getDevice("wheel_left_joint").setVelocity(vL)
    robot.getDevice("wheel_right_joint").setVelocity(vR)