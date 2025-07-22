faces = directory containing our dataset
demoImages = folder containing an example demo image for each model
basicSVM.py = script for running face detection w/ a basic SVM model
ConvolutionalNeuralNetwork.py = script for running face detection w/ a convolution neural network
inprovedSVM.py = script for running face detection w/ improved SVM model

the following directories will be generated upon running each respective script for the first time:
basic_SVM_cache = directory for cached features, trained SVM classifier, and accuracy generated when running the basicSVM.py script

improved_SVM_cache = directory for cached features, trained SVM classifier, and accuracy generated after HNM for improvedSVM.py script

cnn_cache = directory for cached model and accuracy for cnn


Notes: 
1. To run the ConvolutionalNeuralNetwork.py script, you will need to create a virtual environment for python 3.10, as this is the 
python version that tensorflow and keras are compatible with. Then, just install the dependencies in the import sections
using pip.
The install dependencies should be:
pip install opencv-python
pip install pillow
pip install tensorflow
pip install scikit-learn
pip install joblib
pip install matplotlib
pip install keras


2. I tried to include the cache directories in the github repository, but the files were too big to add to github.
This means that all scripts will have to be run from scratch and go through the training/testing phase, which can take a pretty
long time. However, once they are ran for the first time, they run much more quickly after that. I also tried to submit as a zip file to canvas, which
included both the caches and the virtual environment folder, but it was too big to submit to canvas. 
Also, a zip containing only source code and virtual environment, as well as a zip containing only source code and caches were both also too big to submit to canvas.


3. Even after creating the virtual environment and installing dependencies, vs code may still give a 'could not resolve' indicator for keras,
but the scripts should run fine.
