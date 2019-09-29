#--------------
# Deep Pipe for CBLAS Matrex
defmodule DP do
@moduledoc """
A module that implements functions for deep learning 
"""


  def stop() do
    raise("stop")
  end

  def print(x) do
    :io.write(x)
  end

  def newline() do
    IO.puts("")
  end

  # activate function
  def sigmoid(x) do
    cond do
      x > 100 -> 1
      x < -100 -> 0
      true ->  1 / (1+:math.exp(-x))
    end
  end

  def dsigmoid(x) do
    (1 - sigmoid(x)) * sigmoid(x)
  end

  def step(x) do
    if x > 0 do 1 else 0 end
  end

  def relu(x) do
    max(0,x)
  end

  def drelu(x) do
    if x > 0 do 1 else 0 end
  end

  def ident(x) do
    x
  end

  def dident(_) do
    1
  end

  def tanh(x) do
    :math.tanh(x)
  end

  def dtanh(x) do
    1 / (:math.cosh(x) * :math.cosh(x))
  end

  def softmax([x]) do
    softmax(x)
  end
  def softmax(x) do
    c = Enum.max(x)
    x1 = Enum.map(x,fn(y) -> y-c end)
    sum = listsum(x1)
    Enum.map(x1, fn(y) -> :math.exp(y)/sum end)
  end

  def listsum([]) do 0 end
  def listsum([x|xs]) do
    :math.exp(x) + listsum(xs)
  end

  # dummy
  def dsoftmax(x) do
    x
  end

  #error function
  def cross_entropy([x],[y]) do
    cross_entropy1(x,y)
  end
  def cross_entropy1([],[]) do 0 end
  def cross_entropy1([y|ys],[t|ts]) do
    delta = 1.0e-7
    -(t * :math.log(y+delta)) + cross_entropy1(ys,ts)
  end

  def mean_square([x],[t]) do
    mean_square1(x,t) / 2
  end

  def mean_square1([],[]) do 0 end
  def mean_square1([x|xs],[t|ts]) do
    square(x-t) + mean_square1(xs,ts)
  end

  def square(x) do
    x*x
  end

  # apply functin for matrex
  def apply_function(x,f) do
    Matrex.apply(x,f)
  end

  # y=result data t=train_data f=error function
  def loss(y,t,:cross) do
    y1 = Matrex.to_list_of_lists(y)
    t1 = Matrex.to_list_of_lists(t)
    batch_error(y1,t1,fn(x,y) -> cross_entropy(x,y) end)
  end
  def loss(y,t,:square) do
    y1 = Matrex.to_list_of_lists(y)
    t1 = Matrex.to_list_of_lists(t)
    batch_error(y1,t1,fn(x,y) -> mean_square(x,y) end)
  end
  def loss(y,t) do
    y1 = Matrex.to_list_of_lists(y)
    t1 = Matrex.to_list_of_lists(t)
    batch_error(y1,t1,fn(x,y) -> mean_square(x,y) end)
  end

  defp batch_error(y,t,f) do
    batch_error1(y,t,f,0) / length(y)
  end

  defp batch_error1([],[],_,res) do res end
  defp batch_error1([y|ys],[t|ts],f,res) do
    batch_error1(ys,ts,f,f.([y],[t])+res)
  end

  # forward
  def is_tensor(x) do
    is_list(x)
  end

  def is_matrix(x) do
    !is_list(x)
  end

  def forward(x,[]) do x end
  def forward(x,[{:weight,w,_,_}|rest]) do
    x1 = Cmatrix.mult(x,w)
    forward(x1,rest)
  end
  def forward(x,[{:bias,b,_,_}|rest]) do
    {r,_} = x[:size]
    b1 = Cmatrix.expand(b,r)
    x1 = Cmatrix.add(x,b1)
    forward(x1,rest)
  end
  def forward(x,[{:function,f,_,_}|rest]) do
    if is_tensor(x) do
      x1 = Ctensor.apply_function(x,f)
      forward(x1,rest)
    else
      x1 = apply_function(x,f)
      forward(x1,rest)
    end
  end
  def forward(x,[{:softmax,f,_}|rest]) do
    x1 = Cmatrix.apply_row(x,f)
    forward(x1,rest)
  end
  def forward(x,[{:filter,w,st,_,_}|rest]) do
    x1 = Ctensor.convolute(x,w,st)
    forward(x1,rest)
  end
  def forward(x,[{:padding,st}|rest]) do
    x1 = Ctensor.pad(x,st)
    forward(x1,rest)
  end
  def forward(x,[{:pooling,st}|rest]) do
    x1 = Ctensor.pool(x,st)
    forward(x1,rest)
  end
  def forward(x,[{:flatten}|rest]) do
    x1 = Ctensor.flatten(x)
    forward(x1,rest)
  end
  def forward(x,[{:rnn,h,e,_}|rest]) do
    x1 = Rnn.forward(x,h,e,[])
    forward(x1,rest)
  end
  def forward(x,[{:lstm,c,e,_}|rest]) do
    {r,c} = c[:size]
    h = Cmatrix.zeros(r,c)
    x1 = Lstm.forward(x,h,c,e,[])
    forward(x1,rest)
  end

  # forward for backpropagation
  # this store all middle data
  def forward_for_back(_,[],res) do res end
  def forward_for_back(x,[{:weight,w,_,_}|rest],res) do
    x1 = Cmatrix.mult(x,w)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:bias,b,_,_}|rest],res) do
    {r,_} = x[:size]
    b1 = Cmatrix.expand(b,r)
    x1 = Cmatrix.add(x,b1)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:function,f,_,_}|rest],res) do
    if is_tensor(x) do
      x1 = Ctensor.apply_function(x,f)
      forward_for_back(x1,rest,[x1|res])
    else
      x1 = Cmatrix.apply_function(x,f)
      forward_for_back(x1,rest,[x1|res])
    end
  end
  def forward_for_back(x,[{:softmax,f,_}|rest],res) do
    x1 = Cmatrix.apply_row(x,f)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:filter,w,st,_,_}|rest],res) do
    x1 = Ctensor.convolute(x,w,st)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:padding,st}|rest],res) do
    x1 = Ctensor.pad(x,st)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:pooling,st}|rest],[_|res]) do
    x1 = Ctensor.pool(x,st)
    x2 = Ctensor.sparse(x,st)
    forward_for_back(x1,rest,[x1,x2|res])
  end
  def forward_for_back(x,[{:flatten}|rest],res) do
    x1 = Ctensor.flatten(x)
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:rnn,h,e,_}|rest],res) do
    x1 = Rnn.forward_for_back(x,h,e,[])
    forward_for_back(x1,rest,[x1|res])
  end
  def forward_for_back(x,[{:lstm,c,h,e,_}|rest],res) do
    x1 = Lstm.forward_for_back(x,c,h,e,[])
    forward_for_back(x1,rest,[x1|res])
  end

  # numerical gradient
  def numerical_gradient(x,network,t,:square) do
    numerical_gradient(x,network,t)
  end
  def numerical_gradient(x,network,t,:cross) do
    numerical_gradient1(x,network,t,[],[],:cross)
  end
  def numerical_gradient(x,network,t) do
    numerical_gradient1(x,network,t,[],[])
  end

  defp numerical_gradient1(_,[],_,_,res) do
    Enum.reverse(res)
  end
  defp numerical_gradient1(x,[{:filter,w,st,lr,v}|rest],t,before,res) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:filter,w,st,lr,v},rest)
    numerical_gradient1(x,rest,t,[{:filter,w,st,lr,v}|before],[{:filter,w1,st,lr,v}|res])
  end
  defp numerical_gradient1(x,[{:weight,w,lr,v}|rest],t,before,res) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:weight,w,lr,v},rest)
    numerical_gradient1(x,rest,t,[{:weight,w1,lr,v}|before],[{:weight,w1,lr,v}|res])
  end
  defp numerical_gradient1(x,[{:bias,w,lr,v}|rest],t,before,res) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:bias,w,lr,v},rest)
    numerical_gradient1(x,rest,t,[{:bias,w,lr,v}|before],[{:bias,w1,lr,v}|res])
  end
  defp numerical_gradient1(x,[y|rest],t,before,res) do
    numerical_gradient1(x,rest,t,[y|before],[y|res])
  end
  # calc numerical gradient of filter,weigth,bias matrix
  defp numerical_gradient_matrix(x,w,t,before,now,rest) do
    {r,c} = w[:size]
    Enum.map(1..r,
      fn(x1) -> Enum.map(1..c,
                  fn(y1) -> numerical_gradient_matrix1(x,t,x1,y1,before,now,rest) end) end)
    |> Matrex.new()
  end
  defp numerical_gradient_matrix1(x,t,r,c,before,{type,w,lr,v},rest) do
    h = 0.0001
    w1 = Cmatrix.diff(w,r,c,h)
    network0 = Enum.reverse(before) ++ [{type,w,lr,v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type,w1,lr,v}] ++ rest
    y0 = forward(x,network0)
    y1 = forward(x,network1)
    (loss(y1,t,:square) - loss(y0,t,:square)) / h
  end
  defp numerical_gradient_matrix1(x,t,r,c,before,{type,w,st,lr,v},rest) do
    h = 0.0001
    w1 = Cmatrix.diff(w,r,c,h)
    network0 = Enum.reverse(before) ++ [{type,w,st,lr,v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type,w1,st,lr,v}] ++ rest
    y0 = forward(x,network0)
    y1 = forward(x,network1)
    (loss(y1,t,:square) - loss(y0,t,:square)) / h
  end

  defp numerical_gradient1(_,[],_,_,res,:cross) do
    Enum.reverse(res)
  end
  defp numerical_gradient1(x,[{:filter,w,st,lr,v}|rest],t,before,res,:cross) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:filter,w,st,lr,v},rest)
    numerical_gradient1(x,rest,t,[{:filter,w,st,lr,v}|before],[{:filter,w1,st,lr,v}|res],:cross)
  end
  defp numerical_gradient1(x,[{:weight,w,lr,v}|rest],t,before,res,:cross) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:weight,w,lr,v},rest,:cross)
    numerical_gradient1(x,rest,t,[{:weight,w,lr,v}|before],[{:weight,w1,lr,v}|res],:cross)
  end
  defp numerical_gradient1(x,[{:bias,w,lr,v}|rest],t,before,res,:cross) do
    w1 = numerical_gradient_matrix(x,w,t,before,{:bias,w,lr,v},rest,:cross)
    numerical_gradient1(x,rest,t,[{:bias,w,lr,v}|before],[{:bias,w1,lr,v}|res],:cross)
  end
  defp numerical_gradient1(x,[y|rest],t,before,res,:cross) do
    numerical_gradient1(x,rest,t,[y|before],[y|res],:cross)
  end
  # calc numerical gradient of filter,weigth,bias matrix
  defp numerical_gradient_matrix(x,w,t,before,now,rest,:cross) do
    {r,c} = w[:size]
    Enum.map(1..r,
      fn(x1) -> Enum.map(1..c,
                  fn(y1) -> numerical_gradient_matrix1(x,t,x1,y1,before,now,rest,:cross) end) end)
    |> Matrex.new()
  end
  defp numerical_gradient_matrix1(x,t,r,c,before,{type,w,lr,v},rest,:cross) do
    h = 0.0001
    w1 = Cmatrix.diff(w,r,c,h)
    network0 = Enum.reverse(before) ++ [{type,w,lr,v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type,w1,lr,v}] ++ rest
    y0 = forward(x,network0)
    y1 = forward(x,network1)
    (loss(y1,t,:cross) - loss(y0,t,:cross)) / h
  end
  defp numerical_gradient_matrix1(x,t,r,c,before,{type,w,st,lr,v},rest,:cross) do
    h = 0.0001
    w1 = Cmatrix.diff(w,r,c,h)
    network0 = Enum.reverse(before) ++ [{type,w,st,lr,v}] ++ rest
    network1 = Enum.reverse(before) ++ [{type,w1,st,lr,v}] ++ rest
    y0 = forward(x,network0)
    y1 = forward(x,network1)
    (loss(y1,t,:cross) - loss(y0,t,:cross)) / h
  end


  # gradient with backpropagation
  def gradient(x,network,t) do
    x1 = forward_for_back(x,network,[x])
    loss = Cmatrix.sub(hd(x1),t)
    network1 = Enum.reverse(network)
    backpropagation(loss,network1,tl(x1),[])
  end

  #backpropagation
  defp backpropagation(_,[],_,res) do res end
  defp backpropagation(l,[{:function,f,g,h}|rest],[u|us],res) do
    if is_tensor(u) do
      l1 = Ctensor.emult(l,Ctensor.apply_function(u,g))
      backpropagation(l1,rest,us,[{:function,f,g,h}|res])
    else
      l1 = Cmatrix.emult(l,apply_function(u,g))
      backpropagation(l1,rest,us,[{:function,f,g,h}|res])
    end
  end
  defp backpropagation(l,[{:softmax,f,g}|rest],[_|us],res) do
    #l1 = Matrex.apply(l,fn(x) -> 10*x end)
    backpropagation(l,rest,us,[{:softmax,f,g}|res])
  end
  defp backpropagation(l,[{:bias,_,lr,v}|rest],[_|us],res) do
    {n,_} = l[:size]
    b1 = Cmatrix.reduce(l) |> apply_function(fn(x) -> x/n end)
    backpropagation(l,rest,us,[{:bias,b1,lr,v}|res])
  end
  defp backpropagation(l,[{:weight,w,lr,v}|rest],[u|us],res) do
    {n,_} = l[:size]
    w1 = Cmatrix.mult(Cmatrix.transpose(u),l) |> apply_function(fn(x) -> x/n end)
    l1 = Cmatrix.mult(l,Cmatrix.transpose(w))
    backpropagation(l1,rest,us,[{:weight,w1,lr,v}|res])
  end
  defp backpropagation(l,[{:filter,w,st,lr,v}|rest],[u|us],res) do
    w1 = Ctensor.gradient_filter(u,w,l) |> Ctensor.average
    l1 = Ctensor.deconvolute(u,w,l,st)
    backpropagation(l1,rest,us,[{:filter,w1,st,lr,v}|res])
  end
  defp backpropagation(l,[{:pooling,st}|rest],[u|us],res) do
    l1 = Ctensor.restore(u,l,st)
    backpropagation(l1,rest,us,[{:pooling,st}|res])
  end
  defp backpropagation(l,[{:padding,st}|rest],[_|us],res) do
    l1 = Ctensor.remove(l,st)
    backpropagation(l1,rest,us,[{:padding,st}|res])
  end
  defp backpropagation(l,[{:flatten}|rest],[u|us],res) do
    {r,c} = hd(u)[:size]
    l1 = Ctensor.structure(l,r,c)
    backpropagation(l1,rest,us,[{:flatten}|res])
  end
  # c=c(t), e=axpand data of lstm
  defp backpropagation(l,[{:lstm,c,e,lr}|rest],[u|us],res) do
    l1 = Lstm.backpropagation(l,e,u,[])
    backpropagation(l1,rest,us,[{:lstm,c,e,lr}|res])
  end

  #--------sgd----------
  def learning([],_) do [] end
  def learning([{:weight,w,lr,v}|rest],[{:weight,w1,_,_}|rest1]) do
    [{:weight,Cmatrix.update(w,w1,lr),lr,v}|learning(rest,rest1)]
  end
  def learning([{:bias,w,lr,v}|rest],[{:bias,w1,_,_}|rest1]) do
    [{:bias,Cmatrix.update(w,w1,lr),lr,v}|learning(rest,rest1)]
  end
  def learning([{:filter,w,st,lr,v}|rest],[{:filter,w1,st,_,_}|rest1]) do
    [{:filter,Cmatrix.update(w,w1,lr),st,lr,v}|learning(rest,rest1)]
  end
  def learning([network|rest],[_|rest1]) do
    [network|learning(rest,rest1)]
  end
  #--------momentum-------------
  def learning([],_,:momentum) do [] end
  def learning([{:weight,w,lr,v}|rest],[{:weight,w1,_,_}|rest1],:momentum) do
    v1 = Cmatrix.momentum(v,w1,lr)
    [{:weight,Cmatrix.add(w,v1),lr,v1}|learning(rest,rest1,:momentum)]
  end
  def learning([{:bias,w,lr,v}|rest],[{:bias,w1,_,_}|rest1],:momentum) do
    v1 = Cmatrix.momentum(v,w1,lr)
    [{:bias,Cmatrix.add(w,v1),lr,v1}|learning(rest,rest1,:momentum)]
  end
  def learning([{:filter,w,st,lr,v}|rest],[{:filter,w1,st,_,_}|rest1],:momentum) do
    v1 = Cmatrix.momentum(v,w1,lr)
    [{:filter,Cmatrix.add(w,v1),st,lr,v1}|learning(rest,rest1,:momentum)]
  end
  def learning([network|rest],[_|rest1],:momentum) do
    [network|learning(rest,rest1,:momentum)]
  end
  #--------AdaGrad--------------
  def learning([],_,:adagrad) do [] end
  def learning([{:weight,w,lr,h}|rest],[{:weight,w1,_,_}|rest1],:adagrad) do
    h1 = Cmatrix.add(h,Cmatrix.emult(w1,w1))
    [{:weight,Cmatrix.adagrad(w,w1,h1,lr),lr,h1}|learning(rest,rest1,:adagrad)]
  end
  def learning([{:bias,w,lr,h}|rest],[{:bias,w1,_,_}|rest1],:adagrad) do
    h1 = Cmatrix.add(h,Cmatrix.emult(w1,w1))
    [{:bias,Cmatrix.adagrad(w,w1,h1,lr),lr,h1}|learning(rest,rest1,:adagrad)]
  end
  def learning([{:filter,w,st,lr,h}|rest],[{:filter,w1,st,_,_}|rest1],:adagrad) do
    h1 = Cmatrix.add(h,Cmatrix.emult(w1,w1))
    [{:filter,Cmatrix.adagrad(w,w1,h1,lr),st,lr,h1}|learning(rest,rest1,:adagrad)]
  end
  def learning([network|rest],[_|rest1],:adagrad) do
    [network|learning(rest,rest1,:adagrad)]
  end
  #--------Adam--------------
  def learning([],_,:adam) do [] end
  def learning([{:weight,w,lr,mv}|rest],[{:weight,w1,_,_}|rest1],:adam) do
    mv1 = Cmatrix.adammv(mv,w1)
    [{:weight,Cmatrix.adam(w,mv1,lr),lr,mv1}|learning(rest,rest1,:adam)]
  end
  def learning([{:bias,w,lr,mv}|rest],[{:bias,w1,_,_}|rest1],:adam) do
    mv1 = Cmatrix.adammv(mv,w1)
    [{:bias,Cmatrix.adam(w,mv1,lr),lr,mv1}|learning(rest,rest1,:adam)]
  end
  def learning([{:filter,w,st,lr,mv}|rest],[{:filter,w1,st,_,_}|rest1],:adam) do
    mv1 = Cmatrix.adammv(mv,w1)
    [{:filter,Cmatrix.adam(w,mv1,lr),st,lr,mv1}|learning(rest,rest1,:adam)]
  end
  def learning([network|rest],[_|rest1],:adam) do
    [network|learning(rest,rest1,:adam)]
  end


  # print predict of test data
  def accuracy(image,network,label) do
    forward(image,network) |> Cmatrix.to_list() |> score(label,0)
  end

  defp score([],[],correct) do correct end
  defp score([x|xs],[l|ls],correct) do
    if MNIST.onehot_to_num(x) == l do
      score(xs,ls,correct+1)
    else
      score(xs,ls,correct)
    end
  end


  # select randome data size of m , range from 0 to n
  def random_select(image,train,m,n) do
    random_select1(image,train,[],[],m,n)
  end

  defp random_select1(_,_,res1,res2,0,_) do {res1,res2 |> Matrex.new()} end
  defp random_select1(image,train,res1,res2,m,n) do
    i = :rand.uniform(n)
    image1 = Enum.at(image,i)
    train1 = Enum.at(train,i)
    random_select1(image,train,[image1|res1],[train1|res2],m-1,n)
  end

  # for DNN
  def random_select(image,train,m,n,:flatten) do
    random_select1(image,train,[],[],m,n,:flatten)
  end

  defp random_select1(_,_,res1,res2,0,_,:flatten) do {res1 |> Matrex.new(),res2 |> Matrex.new()} end
  defp random_select1(image,train,res1,res2,m,n,:flatten) do
    i = :rand.uniform(n)
    image1 = Enum.at(image,i)
    train1 = Enum.at(train,i)
    random_select1(image,train,[image1|res1],[train1|res2],m-1,n,:flatten)
  end

  def save(file,network) do
    network1 = save1(network)
    File.write(file,inspect(network1,limit: :infinity))
  end

  def save1([]) do [] end
  def save1([{:weight,w,lr,v}|rest]) do
    [{:weight,Cmatrix.to_list(w),lr,Cmatrix.to_list(v)}|save1(rest)]
  end
  def save1([{:bias,w,lr,v}|rest]) do
    [{:bias,Cmatrix.to_list(w),lr,Cmatrix.to_list(v)}|save1(rest)]
  end
  def save1([{:filter,w,st,lr,v}|rest]) do
    [{:filter,Cmatrix.to_list(w),st,lr,Cmatrix.to_list(v)}|save1(rest)]
  end
  def save1([{:function,_,_,:sigmoid}|rest]) do
    [{:sigmoid}|save1(rest)]
  end
  def save1([{:function,_,_,:relu}|rest]) do
    [{:relu}|save1(rest)]
  end
  def save1([{:softmax,_,_}|rest]) do
    [{:softmax}|save1(rest)]
  end
  def save1([network|rest]) do
    [network|save1(rest)]
  end


  def load(file) do
    Code.eval_file(file) |> elem(0) |> load1
  end

  def load1([]) do [] end
  def load1([{:weight,w,lr,v}|rest]) do
    [{:weight,Cmatrix.to_matrex(w),lr,Cmatrix.to_matrex(v)}|load1(rest)]
  end
  def load1([{:bias,w,lr,v}|rest]) do
    [{:bias,Cmatrix.to_matrex(w),lr,Cmatrix.to_matrex(v)}|load1(rest)]
  end
  def load1([{:filter,w,st,lr,v}|rest]) do
    [{:filter,Cmatrix.to_matrex(w),st,lr,Cmatrix.to_matrex(v)}|load1(rest)]
  end
  def load1([{:sigmoid}|rest]) do
    [{:sigmoid,fn(x) -> DP.sigmoid(x) end,fn(x) -> DP.dsigmoid(x) end}|load1(rest)]
  end
  def load1([{:relu}|rest]) do
    [{:relu,fn(x) -> DP.relu(x) end,fn(x) -> DP.relu(x) end}|load1(rest)]
  end
  def load1([{:softmax}|rest]) do
    [{:softmax,fn(x) -> DP.softmax(x) end,fn(x) -> DP.dsoftmax(x) end}|load1(rest)]
  end
  def load1([network|rest]) do
    [network|load1(rest)]
  end



end
