"""image_with_camera controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
import cv2
import numpy as np

def get_image_with_cam(webots_cam):
    img = webots_cam.getImageArray()
    img = np.asarray(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img






# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
print(timestep)

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

CamLeft = robot.getDevice("CamLeft")
CamLeft.enable(timestep)

w, h = CamLeft.getWidth(), CamLeft.getHeight()
print(w, h)



i = 0
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # img = get_image_with_cam(CamLeft)
    # img = CamLeft.getImage()

    # Get the image from the camera
    img = CamLeft.getImage()

    # Decode the image data into a NumPy array
    img_np = np.frombuffer(img, dtype=np.uint8)

    # Reshape the NumPy array to get the image in the correct shape (height, width, channels)
    img_np = img_np.reshape((CamLeft.getHeight(), CamLeft.getWidth(), 4))

    # Extract the RGB channels (assuming 4 channels, where the fourth channel is often an alpha channel)
    img_rgb = img_np[:, :, :3]

    # Now you can use OpenCV to perform various image processing tasks
    # For example, displaying the image
    cv2.imshow("CamLeft Image", img_rgb)
    cv2.waitKey(1)  # This line is necessary for the OpenCV window to update



    if i == 1:
        print(type(img))
        print(len(img))
        print(len(img) / w / h)


        print(img[:10])
        int_values = [int(byte) for byte in img]
        # print(int_values)


    
    # cv2.imshow("img", img)
    # cv2.waitKey(1)
    i += 1





    # pass
    # Read the sensors:
    # Enter here functions to read sensor data, like:
     # val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
     # motor.setPosition(10.0)
# Enter here exit cleanup code.
