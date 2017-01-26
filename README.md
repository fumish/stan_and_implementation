# stan_and_implementation
For stan codes and implementation of them

# Contents
## 1.neural_net.stan
3-layer neural net,
input, middel, output layer.  
Thus let y,x be output and input,  
then y = f(x,w) + e, where e ~ N(0,sigma), and f(x,w)=sigmoid(a \* sigmoid(b \* x)).  
Parameter a and b are sampled from this code, and in order to do it,  
put the list(x,y,n,M,H,sigma), where x is input with [n,M] array,
y is output with [n] array, n is sample number, M is dimension of input,  
H is number of middle layers, sigma is std of learning model.

This code assumes that prior for parameter is ridge.

## 2.nmf_poisson.stan
