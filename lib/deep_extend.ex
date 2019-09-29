defmodule Nat do
  def corpus(str) do
    str |> String.replace("."," .") |> String.split(" ")
  end

  def wordvec(x) do
    x
    |> String.to_charlist
    |> Enum.map(fn(x) -> (x-96)/26 end)
  end

end

defmodule Rnn do
  def forward(_,_,[],res) do res end
  def forward([x|xs],h,[wx,wh,b|ls],res) do
    dt = Cmatrix.add(Cmatrix.add(Cmatrix.mult(x,wx),Cmatrix.mult(h,wh)),b)
    dt1 = Cmatrix.apply_function(dt,fn(x) -> DP.tanh(x) end)
    forward(xs,dt1,ls,[dt1|res])
  end

  def forward_for_back(_,_,[],res) do res end
  def forward_for_back([x|xs],h,[wx,wh,b|ls],res) do
    dt = Cmatrix.add(Cmatrix.add(Cmatrix.mult(x,wx),Cmatrix.mult(h,wh)),b)
    dt1 = Cmatrix.apply_function(dt,fn(x) -> DP.tanh(x) end)
    forward_for_back(xs,dt1,ls,[dt1|res])
  end
end

#LSTM
defmodule Lstm do
  def forward(_,_,_,[],res) do res end
  def forward([x|xs],h,c,[wx,wh,b|ls],res) do
    dt = Cmatrix.add(Cmatrix.add(Cmatrix.mult(x,wx),Cmatrix.mult(h,wh)),b)
    f = dt |> partf |> Cmatrix.apply_function(fn(x) -> DP.sigmoid(x) end)
    g = dt |> partg |> Cmatrix.apply_function(fn(x) -> DP.tanh(x) end)
    i = dt |> parti |> Cmatrix.apply_function(fn(x) -> DP.sigmoid(x) end)
    o = dt |> parto |> Cmatrix.apply_function(fn(x) -> DP.sigmoid(x) end)
    c1 = Cmatrix.add(Cmatrix.emult(c,f),Cmatrix.emult(g,i))
    h1 = Cmatrix.emult(o,Cmatrix.apply_function(c1,fn(x) -> DP.tanh(x) end))
    forward(xs,h,c1,ls,[h1|res])
  end

  def forward_for_back(_,_,_,[],res) do res end
  def forward_for_back([x,x1|xs],h,c,[wx,wh,b|ls],res) do
    dt = Cmatrix.add(Cmatrix.add(Cmatrix.mult(x,wx),Cmatrix.mult(h,wh)),b)
    f = dt |> partf |> Cmatrix.apply_function(fn(x) -> DP.sigmoid(x) end)
    g = dt |> partg |> Cmatrix.apply_function(fn(x) -> :math.tanh(x) end)
    i = dt |> parti |> Cmatrix.apply_function(fn(x) -> DP.sigmoid(x) end)
    o = dt |> parto |> Cmatrix.apply_function(fn(x) -> DP.sigmoid(x) end)
    c1 = Cmatrix.add(Cmatrix.emult(c,f),Cmatrix.emult(g,i))
    h1 = Cmatrix.emult(o,Cmatrix.apply_function(c1,fn(x) -> :math.tanh(x) end))
    forward_for_back([x1|xs],h,c1,ls,[[c1,h1,x1,f,g,i,o]|res])
  end

  #LSTM backpropagation
  # l = loss vector
  # e = list of expanded LSTM. Each element is [wx,wh,b]
  # 3rd argument is saved middle predict data
  # 4th argument is gradient of [wx,wh,b] list
  def backpropagation(_,[],_,res) do res end
  def backpropagation(l,[[_,wh,_]|es],[[c1,_,_,f1,g1,i1,o1],[c2,h2,x2,f2,g2,i2,o2]|us],res) do
    dc = l |> Cmatrix.emult(Cmatrix.apply_function(o1,fn(x) -> DP.dtanh(x) end))
    df = dc |> Cmatrix.emult(f1) |> Cmatrix.apply_function(fn(x) -> DP.dsigmoid(x) end)
    dg = dc |> Cmatrix.emult(i1) |> Cmatrix.apply_function(fn(x) -> DP.dtanh(x) end)
    di = dc |> Cmatrix.emult(g1) |> Cmatrix.apply_function(fn(x) -> DP.dsigmoid(x) end)
    do0 = l |> Cmatrix.emult(Cmatrix.apply_function(c1,fn(x) -> DP.tanh(x) end))
            |> Cmatrix.apply_function(fn(x) -> DP.dsigmoid(x) end)
    d = Cmatrix.stick(df,dg,di,do0)
    b1 = d
    wh1 = Cmatrix.mult(Cmatrix.transpose(h2),d)
    l1 = Cmatrix.mult(d,Cmatrix.transpose(wh))
    wx1 = Cmatrix.mult(Cmatrix.transpose(x2),d)
    backpropagation(l1,es,[[c2,h2,x2,f2,g2,i2,o2]|us],[[wx1,wh1,b1]|res])
  end

  # divide matrex_dt to four parts
  def partf(m) do
    {r,c} = m[:size]
    size = div(c,4)
    index = 1
    Cmatrix.part(m,1,index,r,size)
  end

  def partg(m) do
    {r,c} = m[:size]
    size = div(c,4)
    index = size+1
    Cmatrix.part(m,1,index,r,size)
  end

  def parti(m) do
    {r,c} = m[:size]
    size = div(c,4)
    index = size*2+1
    Cmatrix.part(m,1,index,r,size)
  end

  def parto(m) do
    {r,c} = m[:size]
    size = div(c,4)
    index = size*3+1
    Cmatrix.part(m,1,index,r,size)
  end

end
