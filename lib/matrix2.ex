#---------Tensor for DPBLAS ---------------
defmodule Ctensor do
  def to_matrex([]) do [] end
  def to_matrex([x|xs]) do
    [Matrex.new(x)|to_matrex(xs)]
  end


  def average(x) do
    n = length(x)
    sum(x) |> Matrex.apply(fn(y) -> y/n end)
  end

  def sum([x]) do x end
  def sum([x|xs]) do
    Matrex.add(x,sum(xs))
  end

  def gradient_filter([],_,_) do [] end
  def gradient_filter([u|us],w,[l|ls]) do
    [Cmatrix.gradient_filter(u,w,l)|gradient_filter(us,w,ls)]
  end

  def convolute([],_) do [] end
  def convolute([x|xs],y) do
    [Cmatrix.convolute(x,y)|convolute(xs,y)]
  end
  def convolute([],_,_) do [] end
  def convolute([x|xs],y,s) do
    [Cmatrix.convolute(x,y,s)|convolute(xs,y,s)]
  end

  def deconvolute([],_,_,_) do [] end
  def deconvolute([u|us],filter,[l|ls],st) do
    [Cmatrix.deconvolute(u,filter,l,st)|deconvolute(us,filter,ls,st)]
  end


  def apply_function([],_) do [] end
  def apply_function([x|xs],f)  do
    [Cmatrix.apply_function(x,f)|apply_function(xs,f)]
  end

  def flatten(x) do
    flatten1(x) |> Matrex.new()
  end

  def flatten1([]) do [] end
  def flatten1([x|xs]) do
    [flatten2(x)|flatten1(xs)]
  end

  def flatten2(x) do
    Matrex.to_list_of_lists(x) |> Cmatrix.flatten2()
  end

  def pad([],_) do [] end
  def pad([x|xs],n) do
    [Cmatrix.pad(x,n)|pad(xs,n)]
  end

  def pool([],_) do [] end
  def pool([x|xs],s) do
    [Cmatrix.pool(x,s)|pool(xs,s)]
  end

  def sparse([],_) do [] end
  def sparse([x|xs],s) do
    [Cmatrix.sparse(x,s)|sparse(xs,s)]
  end

  # emult for tensor
  def emult([],[]) do [] end
  def emult([x|xs],[y|ys]) do
    [Cmatrix.emult(x,y)|emult(xs,ys)]
  end

  def structure(x,r,c) do
    Matrex.to_list_of_lists(x) |> structure1(r,c)
  end

  def structure1([],_,_) do [] end
  def structure1([l|ls],r,c) do
    [Cmatrix.structure(l,r,c)|structure1(ls,r,c)]
  end

  def restore([],_,_) do [] end
  def restore([u|us],l,st) do
    [Cmatrix.restore(u,l,st)|restore(us,l,st)]
  end

  def remove([],_) do [] end
  def remove([l|ls],st) do
    [Cmatrix.remove(l,st)|remove(ls,st)]
  end


end

#---------Matrix for DPBLAS ----------------

defmodule Cmatrix do
  def apply_row(x,f) do
    Matrex.to_list_of_lists(x) |> Enum.map(f) |> Matrex.new()
  end

  #list -> matrex data
  def to_matrex(l) do
    Matrex.new(l)
  end

  #matrex -> list
  def to_list(x) do
    Matrex.to_list_of_lists(x)
  end


  def apply_function(m,f) do
    Matrex.apply(m,f)
  end

  def add(x,y) do
    Matrex.add(x,y)
  end

  def sub(x,y) do
    Matrex.subtract(x,y)
  end

  def mult(x,y) do
    Matrex.dot(x,y)
  end

  # y is matrex or scalar
  def emult(x,y) do
    Matrex.multiply(x,y)
  end

  def ediv(m,x) do
    Matrex.divide(m,x)
  end

  def elem(m,r,c) do
    m[r][c]
  end

  def new(r,c) do
    Matrex.zeros(r,c) |> Matrex.apply(fn(_) -> Dmatrix.box_muller() end)
  end
  def new(r,c,x) do
    Matrex.zeros(r,c) |> Matrex.apply(fn(_) -> x end)
  end


  def zeros(r,c) do
    Matrex.zeros(r,c)
  end

  def transpose(x) do
    Matrex.transpose(x)
  end

  def max(x) do
    Matrex.max(x)
  end

  def sum(x) do
    Matrex.sum(x)
  end

  def flatten(x) do
    Matrex.to_list_of_lists(x) |> flatten1() |> Matrex.new()
  end
  def flatten1(x) do
    [flatten2(x)]
  end
  def flatten2([]) do [] end
  def flatten2([x|xs]) do
    x ++ flatten2(xs)
  end

  def structure(x,r,c) do
    structure1(x,r,c) |> Matrex.new()
  end

  def structure1(_,0,_) do [] end
  def structure1(x,r,c) do
    [Enum.take(x,c)|structure1(Enum.drop(x,c),r-1,c)]
  end


  def reduce(x) do
    Matrex.to_list_of_lists(x) |> reduce1() |> Matrex.new()
  end
  def reduce1([x]) do [x] end
  def reduce1([x|xs]) do
    Matrix.add([x],reduce1(xs))
  end

  def expand(x,n) do
    Matrex.to_list(x) |> expand1(n) |> Matrex.new()
  end
  def expand1(x,1) do [x] end
  def expand1(x,n) do
    [x|expand1(x,n-1)]
  end

  # r and c are 1 base
  def diff(x,r,c,d) do
    Matrex.set(x,r,c,Matrex.at(x,r,c)+d)
  end


  # index is 1 base
  def part(x,tr,tc,m,n) do
    s1 = tr
    e1 = tr+m-1
    s2 = tc
    e2 = tc+n-1
    Matrex.submatrix(x,s1..e1,s2..e2)
  end

  # sparse for matrix (use backpropagation)
  def sparse(x,s) do
    {r,c} = x[:size]
    if rem(r,s) != 0 or rem(c,s) != 0 do
      :error
    else
      sparse1(x,r,c,1,s)
    end
  end

  def sparse1(x,r,_,m,_) when m > r do x end
  def sparse1(x,r,c,m,s) do
    sparse2(x,r,c,m,1,s) |> sparse1(r,c,m+s,s)
  end

  def sparse2(x,_,c,_,n,_) when n > c do x end
  def sparse2(x,r,c,m,n,s) do
    x1 = part(x,m,n,s,s)
    max_element = max(x1)
    sparse3(x,m,n,m+s-1,n+s-1,max_element) |> sparse2(r,c,m,n+s,s)
  end

  def sparse3(x,i,_,e1,_,_) when i > e1 do x end
  def sparse3(x,i,j,e1,e2,max_element) do
    sparse4(x,i,j,e1,e2,max_element) |> sparse3(i+1,j,e1,e2,max_element)
  end

  def sparse4(x,_,j,_,e2,_) when j > e2 do x end
  def sparse4(x,i,j,e1,e2,max_element) do
    elt = x[i][j]
    elt1 = if elt == max_element do elt else 0 end
    Matrex.set(x,i,j,elt1) |> sparse4(i,j+1,e1,e2,max_element)
  end


  def convolute(x,y) do
    {r1,c1} = x[:size]
    {r2,c2} = y[:size]
    s = 1
    if rem(r1-r2,s) == 0 and  rem(c1-c2,s) == 0 do
      convolute1(x,y,r1,c1,r2,c2,1,1,s) |> Matrex.new()
    else
      :error
    end
  end

  def convolute(x,y,s) do
    {r1,c1} = x[:size]
    {r2,c2} = y[:size]
    if rem(r1-r2,s) == 0 and  rem(c1-c2,s) == 0 do
      convolute1(x,y,r1,c1,r2,c2,1,1,s) |> Matrex.new()
    else
      :error
    end
  end


  def convolute1(_,_,r1,_,r2,_,m,_,_) when m+r2 > r1+1 do [] end
  def convolute1(x,y,r1,c1,r2,c2,m,n,s) do
    [convolute2(x,y,r1,c1,r2,c2,m,n,s)|convolute1(x,y,r1,c1,r2,c2,m+s,n,s)]
  end

  def convolute2(_,_,_,c1,_,c2,_,n,_) when n+c2 > c1+1 do [] end
  def convolute2(x,y,r1,c1,r2,c2,m,n,s) do
    [convolute_mult_sum(x,y,m,n)|convolute2(x,y,r1,c1,r2,c2,m,n+s,s)]
  end

  def convolute_mult_sum(x,y,m,n) do
    {r,c} = y[:size]
    x1 = part(x,m,n,r,c)
    emult(x1,y) |> Matrex.sum()
  end

  def pad(x,n) do
    Matrex.to_list_of_lists(x) |> pad1(n) |> Matrex.new()
  end

  def pad1(x,0) do x end
  def pad1(x,n) do
    {_,c} = Matrix.size(x)
    zero1 = Matrix.zeros(n,c+n*2)
    zero2 = Matrix.zeros(1,n)
    x1 = Enum.map(x,fn(y) -> hd(zero2) ++ y ++ hd(zero2) end)
    zero1 ++ x1 ++ zero1
  end

  #remove ,-> padding
  def remove(x,n) do
    Matrex.to_list(x) |> remove1(n) |> Matrex.new()
  end

  def remove1(x,0) do x end
  def remove1(x,n) do
    x1 = Enum.drop(Enum.reverse(Enum.drop(Enum.reverse(x),n)),n)
    Enum.map(x1,fn(y) -> Enum.drop(Enum.reverse(Enum.drop(Enum.reverse(y),n)),n) end)
  end

  # poolong
  def pool(x,s) do
    {r,c} = x[:size]
    if rem(r,s) != 0 or rem(c,s) != 0 do
      IO.puts("Bad argment pooling")
      :error
    else
      pool1(x,r,c,1,s) |> Matrex.new()
    end
  end

  def pool1(_,r,_,m,_) when m > r do [] end
  def pool1(x,r,c,m,s) do
    [pool2(x,r,c,m,1,s)|pool1(x,r,c,m+s,s)]
  end

  def pool2(_,_,c,_,n,_) when n > c do [] end
  def pool2(x,r,c,m,n,s) do
    x1 = part(x,m,n,s,s)
    [Matrex.max(x1)|pool2(x,r,c,m,n+s,s)]
  end

  def rotate180(x) do
    x1 = Matrex.to_list_of_lists(x)
    Enum.reverse(Enum.map(x1,fn(y) -> Enum.reverse(y) end)) |>
    Matrex.new()
  end

  def deconvolute(u,filter,loss,st) do
    {r,_} = filter[:size]
    loss |> pad(r-1) |> convolute(rotate180(filter),st) |> emult(u)
  end

  def gradient_filter(u,filter,loss) do
    {r,c} = filter[:size]
    {m,n} = loss[:size]
    Enum.map(1..r,
      fn(x1) -> Enum.map(1..c,
                  fn(y1) -> gradient_filter1(u,loss,x1,y1,m,n) end) end)
    |> Matrex.new()
  end

  def gradient_filter1(u,error,x1,y1,m,n) do
    p = part(u,x1,y1,m,n)
    p |> Cmatrix.emult(error)
    |> sum
  end

  # restore <-> pool
  def restore(u,los,st) do
    los1 = Matrex.to_list_of_lists(los) |> increase(st) |> Matrex.new()
    Cmatrix.emult(u,los1)
  end

  # e.g.  increase(x,2)   [[1,1,2,2,],
  #        [[1,2],         [1,1,2,2],
  #         [3,4]]     ->  [3,3,4,4],
  #                        [3,3,4,4]]
  def increase([],_) do [] end
  def increase([x|xs],st) do
    x1 = increase1(x,st) |> increase2(st)
    x1 ++ increase(xs,st)
  end

  def increase1([],_) do [] end
  def increase1([x|xs],st) do
    increase3(x,st) ++ increase1(xs,st)
  end

  def increase2(_,0) do [] end
  def increase2(x,st) do
    [x|increase2(x,st-1)]
  end

  def increase3(x,1) do [x] end
  def increase3(x,st) do
    [x|increase3(x,st-1)]
  end

  # for learning
  # each element x-y*lr
  def update(x,y,lr) do
    Matrex.apply(x,y,fn(x,y) -> x - y*lr end)
  end

  def momentum(v,g,lr) do
    Matrex.apply(v,g,fn(v,g) -> 0.5*v - lr*g end)
  end

  def adagrad(w,g,h,lr) do
    Cmatrix.sub(w,Matrex.apply(g,h,fn(g,h) -> lr*(1 / adagrad_sqrt(h))*g end))
  end


  def adagrad_sqrt(x) do
    if x != 0 do
      :math.sqrt(x)
    else
      1
    end
  end

  """
  def adam_init(w) do
    if !is_list(w) do
      {r,c} = w[:size]
      [Cmatrix.zeros(r,c),Cmatrix.zeros(r,c)]
    else
      w
    end
  end

  def adammv(mv,grad) do
    mv1 = adam_init(mv)
  end

  def adammv1([],[]) do [] end
  def adammv1([mv|mvs],[g|gs]) do
    [adammv2(mv,g)|adammv1(mvs,gs)]
  end

  def adammv2([],[]) do [] end
  def adammv2([mv|mvs],[g|gs]) do
    beta1 = 0.9
    beta2 = 0.999
    [m,v] = mv
    m1 = beta1*m+(1-beta2)*g
    v1 = beta2*v+(1-beta2)*(g*g)
    [[m1,v1]|adammv2(mvs,gs)]
  end

  def adam([],[],_) do [] end
  def adam([w|ws],[mv|mvs],lr) do
    [adam1(w,mv,lr)|adam(ws,mvs,lr)]
  end

  def adam1([],[],_) do [] end
  def adam1([w|ws],[mv|mvs],lr) do
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 10.0e-8
    [m,v] = mv
    m1 = m/(1-beta1)
    v1 = v/(1-beta2)
    [w-lr/(:math.sqrt(v1)+epsilon)*m1|adam1(ws,mvs,lr)]
  end
  """
end
