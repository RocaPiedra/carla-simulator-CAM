import os
debug = False # Shows the PIL images during the preprocessing steps
visualize = False # prints information during execution of layers and steps to debug
sendToGPU = True
paused = False
activate_deleter = False # global variable to close cleanly the application
fixed_delta_seconds = 0.05 # to maintain constant delta time in every client
imagenet_weights_path = f'{os.getcwd()}/utils/imagenet1000_clsid_to_human.pkl'
carla_logo = f'{os.getcwd()}/utils/Carla-Simulator-CAM-Logo.png'
gui_cam_logo = f'{os.getcwd()}/utils/GUI-CAM-Logo.png'
if os.name == 'nt':
    unreal_engine_path = r"C:\Users\pablo\CARLA_0.9.13\Carla\CarlaUE4.exe"
else:
    unreal_engine_path = '/opt/carla-simulator/CarlaUE4.sh'
WHITE = (255, 255, 255)
BLACK = (0,0,0)
BUTTON_COLOR = (169,169,169)
RED_BUTTON_COLOR = (139,0,0)
imagenet_relevant_classes = [444,479,475,757,468,569,575,670,671,705,751,817,829,864,919,920,671]