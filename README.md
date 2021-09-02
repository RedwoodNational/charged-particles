# Charged particles simulator
Charged particles interactions simulator. OpenGL Interoperability with CUDA.

## Demo
![Alt Text](https://github.com/curlysilk53/charged-particles/blob/main/particles_demo.gif)



## Controls

|    Control    |     Action    |
|:-------------:|:-------------:|
| W             | move forward  |
| A             | move left     |
| S             | move down     |
| D             | move right    |
|mouse left button     | shoot         |
|mouse cursor         | camera        |
| ESC           | exit          |

## Requirements

+ Visual Studio 2019
+ CUDA Toolkit 11.1
+ Compatible NVIDIA GPU

## Texture
Custom image can be used to change particles texture. To provide custom texture it is required for the image to be converted to a special format. First eight bytes of the file should contain texture dimensions (4 bytes for width and 4 bytes for height), the rest part should contain pixels themself (4 bytes for each pixel). Texture should be named as in.data and placed in the same folder as the exutable file. 

## Parameters
To change the behavior of the model it is possible to adjust internal constants that are responsible for physical properties. 
```C++
int a = 15;           // half of the side of the cube  
int n = 150;          // number of particles  
float eps = 1e-3;     // epsilon to avoid division by zero  
float wc = 0.99;      // deceleration factor  
float k = 50.0;       // Coulombs law constant
float dt = 0.01;      // integration step  
float g = 20;         // gravitational acceleration
float qi = 1;         // particle charges  
float qc = 30;        // camera charge
float qb = 50;        // bullet charge  
float vb = 30;        // bullet speed  
float shift_z = 0.75; // shift the tension map  
float radius = 1;     // particles radius
```
## Notes

+ CPU is used to compute particles behaviour
+ GPU is used to generate floors tension map
