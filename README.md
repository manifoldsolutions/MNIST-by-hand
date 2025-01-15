
### Learning MNIST dataset with 1-hidden-layer fully-connected network with sigmoid activations "by hand", 
#### that is by calculating all matrices and derivatives explicitly using only the jax.numpy library (Google JAX's version of numpy which is like numpy but with GPU acceleration). 
##### Achieving accuracy >95% in ~100 epochs (~1min) on my GPU laptop     

Only cheating when using JAX to calculate the gradient of cross-entropy.

Usage: You should have JAX+CUDA installed to run this smoothly with Python3. 
Without cuda support, JAX will use CPU, so it will be much slower.
If you don't have or don't know JAX, you can easily modify the code to be written 
in numpy (replace "jn" with "np" + debug), but this, again, will slow it down.

CC-BY-NC-4.0                                                       
