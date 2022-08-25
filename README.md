# carla-simulator-CAM
Application to test CNNs using Class Activation Mapping techniques in the Carla Simulator.
The idea behind this project is to create an application to increase the transparency of the DL models in the autonomous driving context.

## Features
The application is launched through the carla_CAM.py script and it launches all the background processes.
The app will launch the simulator if it is not running already, will deploy traffic and start the Pygame-based interactive window when you can visualize the sensors selected and interact with them.
The app will also manage the garbage collection and process termination if it is exited. It is possible to maintain the simulator running with the flag *--keepsim* in case you want to launch the app faster.

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


