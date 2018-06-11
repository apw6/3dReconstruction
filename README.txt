In order to compile and run our code, it is required that you have both OpenCV and the SfM module available. 

We recommend building SfM with OpenCV, and instructions for installing the SfM module can be found at https://docs.opencv.org/3.4.1/db/db8/tutorial_sfm_installation.html.

Note the link to the 3.4.1 documentation as this is the version this program was developed under. Also note that CERES requires C++11 features and so it is helpful to use the  ENABLE_CXX11 option with cmake.

Once all the libraries are built, you can compile the program with a command like 
$ g++ Source.cpp -std=c++11 -I /usr/include/eigen3/ `pkg-config opencv --libs --cflags`

The program does not support a command line interface, but can be configured before compilation. This includes several directives for specifying what kind of visualizations to display, as well as constants for specifying input video, how many frames to process, and the location of output files.

After successfully running the program, the output can be passed to pmvs2 which can download from https://www.di.ens.fr/pmvs/documentation.html. As the documents state, pmvs2 would be executed as
$ ./pmvs2 ~/projectDir/root/ options.txt

This will output a .ply file to the ~/projectDir/root/models directory which can then be viewed and manipulated in a 3D modeler of your choice e.g. MeshLab (http://www.meshlab.net/).
