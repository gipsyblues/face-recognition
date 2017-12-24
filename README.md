## Face Recognition using Eigenface method

The language I used for this face recognition project is C++. I used OpenCV library to read and write
images, and I used Eigen 3 library to performance matrix/vector arithmetic operations.

To run my code, you will first need to have OpenCV library installed because that is what I used to load
and display images. More information on how to install OpenCV in this link:
https://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html

Eigen 3 library has been included in the folder “Eigen3”, and I wrote the makefile to automatically
include it when I compile my code.

In the “face_recognition” folder, you can see an executable file named “face_recognition”, and this is the
program you want to run. There is no argument needed to run the program, as all training images and test
images will load automatically. The training images are located in “trainingg_images” folder, and test
images are located in “test_images” folder. Outputs images, such as eigenfaces, meanfaces, and
reconstructed faces, are located in the “output_images” folder.

To run the program, type:
```
./face_recognition
```

If you modify the code and want to recompile, you can type:
```
make
```

If it doesn’t work, you might want to modify the CMakeLists.txt first by doing:
```
cmake.
```

Then try “make” again.
