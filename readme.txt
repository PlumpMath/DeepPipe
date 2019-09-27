Deep Learning with elixir
This project is called Deep Pipe(DP)

Network example (See test.ex)

defnetwork init_network2(_x) do
  _x |> f(5,5) |> flatten
  |> w(576,100) |> b(100) |> sigmoid
  |> w(100,10) |> b(10) |> sigmoid
end

Install:
  sudo apt install build-essential
  sudo apt-get install build-essential erlang-dev libatlas-base-dev
  mix deps.get


Usage:
  iex -S mix

module DP is Deep Pipe(DP) module
module CTensor is code for CNN data operation
module Cmatrix is code for Matrix
module MNIST is code for MNIST data set
module Time is time/1 for measure execution time

I implemented backpropagation and numerical-gradient
Now I'm testing small data set.


expample:
iex(1)> require Time

iex(1)> require(Time)
Time
iex(2)> Time.time(Test.momentum(30,100))
preparing data
ready
2.866287227629866
2.5600212240059506
...
0.04318082027257467
0.029026173275906994
0.03131037967594155
0.06550367669302301
accuracy rate = 0.879
"time: 55248299 micro second"
"-------------"
:ok

specification:

data structure
matrix  data structure of Matrex(CBLAS) e.g. m[2*3]
row_vector data structure of Matrex(CBLAS)  e.g. m[1*3]
tensor e.g. [m[2*3],n[2*3]]
network e.g. [{:weight,w,lr,v},{:bias,b,lr},{:function,f,g}]
weight {:weight,w,lr,v} w is matrix, lr is learning rate, v is for momentum,adagrad,adam
bias   {:bias,b,lr,v} b is row vector
filter {:filter,w,lr,st,v} st is stradd for convolution
pad    {:pad,n} n is size of padding
pool   {:pool,st} st is stradd
function {:function,f,g,h} f is function, g is differential function, h is function name
softmax {:softmax,f,_} f is function, only output layer softmax is set with cross_entropy

module macros
defnetwork is macros to describe network
argument must have under bar to avoid warning message

w(m,n)  weight matrix size(m,n). elements are Gaussian distribution random float
w(m,n,lr) lr is learning rate (default is 0.1)
w(m,n,lr,z) z is multiple for Gaussian distribution random float. (default is 0.1)

b(n) bias row_vector size(n). elements are all zero
b(n,lr) lr is learning rate (default is 0.1)

function sigmoid,relu,ident,softmax

f(m,n) filter matrix size(m,n). elements are Gaussian distribution random float
f(m,n,lr) lr is learning rate
f(m,n,lr,z) z is multiple for Gaussian distribution random float.(default is 0.1)
f(m,n,lr,z,st) st is stradd

pad(n) padding n is size of padding

pool(st) pooling stradd size is st

module DP
forward/2 forward calculation for batch data
forward(x,network) x is data(matrix or tensor) , network

numerical_gradient/3 calculate gradient by numerical differentiation
numerical_gradient(x,network,t) x is data, t is train data, loss function is mean_square
numerical_gradient(x,network,t,:cross) loss function is cross_entropy
numerical_gradient(x,network,t,:square) loss function is mean_square

gradient/3 caluculate gradient by backpropagation
gradient(x,network,t)  x is data, t is train data

learning/2, /3 update network with gradient
learning(network,gradient)  update with sgd
learning(network,gradient,:sgd)
learning(network,gradient,:momentum) update with momentum method
learning(network,gradient,:adagrad) update with adagrad method
learning(network,gradient,:adam) update with adam method

print/1 print data
newline/0 print LF
save/2 save network data to file
save(filename,network)
load/1 load network data from file
load(filename)

mean_square/2 loss function
mean_square(y,t) y is row vector of result and t is row vector of train
cross_entropy/2 loss function
cross_entropy(y,t) y is row vector of result and t is row vector of train

module MNIST
train_image(n)  get train image data size of n. Each data is 28*28 matrix
train_image(n,:flatten) get train image data size of n. Each data is 784 row_vector
train_label_onehot(n) get train label data aas onehot row_vector

test_image(n)  get test image data size of n. Each data is 28*28 matrix
test_image(n,:flatten) get test image data size of n. Each data is 784 row_vector
test_label_onehot(n) get test label data aas onehot row_vector

module time
time/1
time(func) e.g. time(1+2)
