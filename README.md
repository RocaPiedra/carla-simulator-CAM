# carla-simulator-CAM
Application to test CNNs using Class Activation Mapping techniques in the Carla Simulator.
The idea behind this project is to create an application to increase the transparency of the DL models in the autonomous driving context.

## Features
The application is launched through the carla_CAM.py script and it launches all the background processes.
The app will launch the simulator if it is not running already, will deploy traffic and start the Pygame-based interactive window when you can visualize the sensors selected and interact with them.
The app will also manage the garbage collection and process termination if it is exited. It is possible to maintain the simulator running with the flag *--keepsim* in case you want to launch the app faster.

### Mid execution selection
The app offers interactive ways to select different parameters of the visualization any time. To do so, the pygame execution window reads "events" that are the interactions between the keyboard and mouse (input periphera(ls in general). It is possible to catch the type of event registered and create an option menu. We can differentiate two types of inputs: 
#### Keyboard inputs
- Key **SPACE**: Pauses the simulation and displays the saliency mask obtained for the selected technique, if the technique is not selected it will not stop the simulation and tell the user to select a technique. If the simulation is already paused, it resumes it.
- Key **M**: Stops the simulation and displays a menu to select the CAM technique. If it is pressed when the simulation is paused (showing the saliency mask), it will prompt the user the method menu again to select a different method and compare the resulting saliency masks.
- Key **N**: Stops the simulation and displays a menu to select the CNN architecture.
- Key **T**: Performs a forward pass in the model and returns the top 5 classes detected.
- Key **Q** and **ESCAPE**: Stops the execution of the app and the simulator (unless the *--keepsim* flag has been used during launch).

#### Mouse inputs
- With the simulation running: Allows the user to select the input sensor to visualize. The area of the sensor that you click will select the feed that provides the input image to evaluate.
- With a menu displayed: Allows the user to click and select an item from the menu.


### Compatible CNN architectures
- ResNet
- Alexnet
- VGGnet
- YOLOv5

### Compatible CAM Techniques
#### Gradient Based Techniques
- Grad-CAM
- Grad-CAM++
- XGrad-CAM
- FullGrad
#### Gradient Based Techniques
- Score-CAM
- Ablation-CAM
- Eigen-CAM



## References
This project would not be possible without the incredible work from:
- The developers behind [https://github.com/carla-simulator/carla](Carla Simulator)
- Jacob Gildenblat and his amazing repository to create CNN visualizations with PyTorch [https://github.com/jacobgil/pytorch-grad-cam](pytorch-grad-cam)


