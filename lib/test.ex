defmodule Test do
  import Network

  # for DNN test
  defnetwork init_network1(_x) do
    _x |> w(784,300) |> b(300) |> relu
    |> w(300,100) |> b(100) |> relu
    |> w(100,10) |> b(10) |> softmax
  end

  # for sgd test
  defnetwork init_network2(_x) do
    _x |> f(5,5) |> flatten
    |> w(576,300) |> b(300) |> relu
    |> w(300,100) |> b(100) |> relu
    |> w(100,10) |> b(10) |> softmax
  end

  # for momentum test
  defnetwork init_network3(_x) do
    _x |> f(5,5,0.03) |> flatten
    |> w(576,300,0.03) |> b(300,0.03) |> relu
    |> w(300,100,0.03) |> b(100,0.03) |> relu
    |> w(100,10,0.03) |> b(10,0.03) |> softmax
  end

  # for adagrad test
  defnetwork init_network4(_x) do
    _x |> f(10,10) |> flatten
    |> w(361,300) |> b(300) |> relu
    |> w(300,100) |> b(100) |> relu
    |> w(100,10) |> b(10) |> softmax
  end

  # for adam test
  defnetwork init_network5(_x) do
    _x |> f(5,5,0.005) |> flatten
    |> w(576,300,0.005) |> b(300,0.005) |> relu
    |> w(300,100,0.005) |> b(100,0.005) |> relu
    |> w(100,10,0.005) |> b(10,0.005) |> softmax
  end

  def dnn(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000,:flatten)
    label = MNIST.train_label_onehot(3000)
    network = init_network1(0)
    test_image = MNIST.test_image(1000,:flatten) |> Cmatrix.to_matrex
    test_label = MNIST.test_label(1000)
    IO.puts("ready")
    network1 = dnn1(image,network,label,m,n)
    correct = DP.accuracy(test_image,network1,test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct / 1000)
  end

  def dnn1(_,network,_,_,0) do network end
  def dnn1(image,network,train,m,n) do
    {image1,train1} = DP.random_select(image,train,m,2000,:flatten)
    network1 = DP.gradient(image1,network,train1)
    network2 = DP.learning(network,network1)
    y = DP.forward(image1,network2)
    loss = DP.loss(y,train1,:cross)
    DP.print(loss)
    DP.newline()
    dnn1(image,network2,train,m,n-1)
  end


  def sgd(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000) |> Ctensor.to_matrex
    label = MNIST.train_label_onehot(3000)
    network = init_network2(0)
    test_image = MNIST.test_image(1000) |> Ctensor.to_matrex
    test_label = MNIST.test_label(1000)
    IO.puts("ready")
    network1 = sgd1(image,network,label,m,n)
    correct = DP.accuracy(test_image,network1,test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct / 1000)
  end

  def sgd1(_,network,_,_,0) do network end
  def sgd1(image,network,train,m,n) do
    {image1,train1} = DP.random_select(image,train,m,2000)
    network1 = DP.gradient(image1,network,train1)
    network2 = DP.learning(network,network1)
    y = DP.forward(image1,network2)
    loss = DP.loss(y,train1,:cross)
    DP.print(loss)
    DP.newline()
    sgd1(image,network2,train,m,n-1)
  end

  def momentum(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000) |> Ctensor.to_matrex
    label = MNIST.train_label_onehot(3000)
    network = init_network3(0)
    test_image = MNIST.test_image(10000) |> Ctensor.to_matrex
    test_label = MNIST.test_label(10000)
    IO.puts("ready")
    network1 = momentum1(image,network,label,m,n)
    correct = DP.accuracy(test_image,network1,test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct / 10000)
  end

  def momentum1(_,network,_,_,0) do network end
  def momentum1(image,network,train,m,n) do
    {image1,train1} = DP.random_select(image,train,m,2000)
    network1 = DP.gradient(image1,network,train1)
    network2 = DP.learning(network,network1,:momentum)
    y = DP.forward(image1,network2)
    loss = DP.loss(y,train1,:cross)
    DP.print(loss)
    DP.newline()
    momentum1(image,network2,train,m,n-1)
  end

  def adagrad(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000) |> Ctensor.to_matrex
    label = MNIST.train_label_onehot(3000)
    network = init_network3(0)
    test_image = MNIST.test_image(10000) |> Ctensor.to_matrex
    test_label = MNIST.test_label(10000)
    IO.puts("ready")
    network1 = adagrad1(image,network,label,m,n)
    correct = DP.accuracy(test_image,network1,test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct / 10000)
  end

  def adagrad1(_,network,_,_,0) do network end
  def adagrad1(image,network,train,m,n) do
    {image1,train1} = DP.random_select(image,train,m,2000)
    network1 = DP.gradient(image1,network,train1)
    network2 = DP.learning(network,network1,:adagrad)
    y = DP.forward(image1,network2)
    loss = DP.loss(y,train1,:cross)
    DP.print(loss)
    DP.newline()
    adagrad1(image,network2,train,m,n-1)
  end


  # under constructing
  def adam(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(3000) |> Ctensor.to_matrex
    label = MNIST.train_label_onehot(3000)
    network = init_network4(0)
    test_image = MNIST.test_image(1000) |> Ctensor.to_matrex
    test_label = MNIST.test_label(1000)
    IO.puts("ready")
    network1 = adam1(image,network,label,m,n)
    correct = DP.accuracy(test_image,network1,test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct / 1000)
  end

  def adam1(_,network,_,_,0) do network end
  def adam1(image,network,train,m,n) do
    {image1,train1} = DP.random_select(image,train,m,2000)
    network1 = DP.gradient(image1,network,train1)
    network2 = DP.learning(network,network1,:adam)
    y = DP.forward(image1,network2)
    loss = DP.loss(y,train1,:cross)
    DP.print(loss)
    DP.newline()
    adam1(image,network2,train,m,n-1)
  end

  def all(m,n) do
    IO.puts("preparing data")
    image = MNIST.train_image(60000) |> Ctensor.to_matrex
    label = MNIST.train_label_onehot(60000)
    network = init_network4(0)
    test_image = MNIST.test_image(10000) |> Ctensor.to_matrex
    test_label = MNIST.test_label(10000)
    IO.puts("ready")
    network1 = all1(image,network,label,m,n,test_image,test_label)
    correct = DP.accuracy(test_image,network1,test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct / 10000)
  end


  def all1(_,network,_,_,0,_,_) do network end
  def all1(image,network,train,m,n,test_image,test_label) do
    network1 = all2(image,network,train,m)
    image1 = Enum.take(image,100)
    train1 = Enum.take(train,100) |> Cmatrix.to_matrex
    y = DP.forward(image1,network1)
    loss = DP.loss(y,train1,:cross)
    DP.print(loss)
    DP.newline()
    correct = DP.accuracy(test_image,network1,test_label)
    IO.write("accuracy rate = ")
    IO.puts(correct / 10000)
    all1(image,network1,train,m,n-1,test_image,test_label)
  end

  def all2(image,network,train,size) do
    if length(image) <= size do
      train1 = train |> Cmatrix.to_matrex
      network1 = DP.gradient(image,network,train1)
      network2 = DP.learning(network,network1,:adagrad)
      IO.puts(".")
      network2
    else
      train1 = Enum.take(train,size) |> Cmatrix.to_matrex
      network1 = DP.gradient(Enum.take(image,size),network,train1)
      network2 = DP.learning(network,network1,:adagrad)
      IO.write(".")
      all2(Enum.drop(image,size),network2,Enum.drop(train,size),size)
    end
  end


  defnetwork check_network(_x) do
    _x |> cf([[0.1,0.2],[0.3,0.4]])
    |> cf([[0.2,0.1],[0.2,0.3]])
    |> flatten
    |> cw([[0.1,0.2],
              [0.3,0.2],
              [0.2,0.2],
              [0.4,0.2]]) |> cb([[0,0]]) |> sigmoid
    |> cw([[0.1,0.2],[0.2,0.1]]) |> cb([[0,0]]) |> softmax
  end
  def check() do
    dt = Ctensor.to_matrex([[[0.1,0.2,0.03,0.4],
                             [0.04,0.5,0.3,0.1],
                             [0.2,0.03,0.04,0.5],
                             [0.5,0.6,0.4,0.3]],
                            [[0.2,0.3,0.2,0.4],
                             [0.1,0.2,0.1,0.5],
                             [0.3,0.3,0.2,0.4],
                             [0.1,0.2,0.1,0.3]],
                            [[0.2,0.1,0.1,0.5],
                             [0.2,0.1,0.4,0.2],
                             [0.2,0.1,0.3,0.3],
                             [0.2,0.6,0.4,0.3]]])
    tt = Cmatrix.to_matrex([[0,1],[1,0],[1,0]])
    network = check_network(0)
    network1 = DP.numerical_gradient(dt,network,tt,:cross)
    network2 = DP.gradient(dt,network,tt)
    IO.inspect(network1)
    IO.inspect(network2)
    true
  end

  defnetwork check_network1(_x) do
    _x |> f(2,2) |> flatten
    |> w(2,2) |> b(2) |> relu
  end
  def check_io() do
    network = init_network1(0)
    DP.save("test.dp",network)
    network1 = DP.load("test.dp")
    network1
  end

  defnetwork rnntest(_x) do
    _x |> rnn(5,4,3) |> softmax
  end

end
