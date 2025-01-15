
## Learning MNIST dataset with 1 hidden layer fully connected network "by hand", 
### that is using only Google JAX's numpy-equivalent for GPU 
### acceleration. Forward propagation and backpropagation both          
### explicitly calculated in terms of the appropriate matrices.         
### Achieving accuracy >95% in ~100 epochs (~1min) on my GPU laptop     

Usage: You should have JAX+CUDA installed to run this smoothly with Python3. 
Without cuda support, JAX will use CPU, so it will be much slower.
If you don't have or don't know JAX, you can easily modify the code to be written 
in numpy (replace "jn" with "np" + debug), but this, again, will slow it down.

CC-BY-NC-4.0                                                       
