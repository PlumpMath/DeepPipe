defmodule Time do
  defmacro time(exp) do
    quote do
      {time, dict} = :timer.tc(fn -> unquote(exp) end)
      IO.inspect("time: #{time} micro second")
      IO.inspect("-------------")
      dict
    end
  end
end

defmodule Network do
  defmacro defnetwork(name, do: body) do
    {_, _, [{arg, _, _}]} = name
    body1 = parse(body, arg)

    quote do
      def unquote(name) do
        unquote(body1)
      end
    end
  end

  # filter
  def parse({:f, _, [x, y]}, _) do
    quote do
      {:filter, Cmatrix.new(unquote(x), unquote(y), 0.1), 1, 0.1,
       Cmatrix.zeros(unquote(x), unquote(y))}
    end
  end

  def parse({:f, _, [x, y, lr]}, _) do
    quote do
      {:filter, Cmatrix.new(unquote(x), unquote(y), 0.1), 1, unquote(lr),
       Cmatrix.zeros(unquote(x), unquote(y))}
    end
  end

  def parse({:f, _, [x, y, lr, z]}, _) do
    quote do
      {:filter, Cmatrix.new(unquote(x), unquote(y), unquote(z)), 1, unquote(lr),
       Cmatrix.zeros(unquote(x), unquote(y))}
    end
  end

  def parse({:f, _, [x, y, lr, z, st]}, _) do
    quote do
      {:filter, Cmatrix.new(unquote(x), unquote(y), unquote(z)), unquote(st), unquote(lr),
       Cmatrix.zeros(unquote(x), unquote(y))}
    end
  end

  # pooling
  def parse({:pool, _, [x]}, _) do
    quote do
      {:pooling, unquote(x)}
    end
  end

  # padding
  def parse({:pad, _, [x]}, _) do
    quote do
      {:padding, unquote(x)}
    end
  end

  # constant weight for test
  def parse({:cw, _, [x]}, _) do
    quote do
      {:weight, Matrex.new(unquote(x)), 0.1, 0}
    end
  end

  # constant filter for test
  def parse({:cf, _, [x]}, _) do
    quote do
      {:filter, Matrex.new(unquote(x)), 1, 0.1, 0}
    end
  end

  # constant bias for test
  def parse({:cb, _, [x]}, _) do
    quote do
      {:bias, Matrex.new(unquote(x)), 0.1, 0}
    end
  end

  # weight
  def parse({:w, _, [x, y]}, _) do
    quote do
      {:weight, Cmatrix.new(unquote(x), unquote(y), 0.1), 0.1,
       Cmatrix.zeros(unquote(x), unquote(y))}
    end
  end

  def parse({:w, _, [x, y, lr]}, _) do
    quote do
      {:weight, Cmatrix.new(unquote(x), unquote(y), 0.1), unquote(lr),
       Cmatrix.zeros(unquote(x), unquote(y))}
    end
  end

  def parse({:w, _, [x, y, lr, z]}, _) do
    quote do
      {:weight, Cmatrix.new(unquote(x), unquote(y), unquote(z)), unquote(lr),
       Cmatrix.zeros(unquote(x), unquote(y))}
    end
  end

  # rnn
  def parse({:rnn, _, [x, y, r]}, _) do
    quote do
      {:rnn, Cmatrix.zeros(unquote(x), unquote(y)), unquote(gen_rnn(x, y, r))}
    end
  end

  def parse({:rnn, _, [x, y, r, lr]}, _) do
    quote do
      {:rnn, Cmatrix.zeros(unquote(x), unquote(y)), unquote(gen_rnn(x, y, r)), unquote(lr)}
    end
  end

  def parse({:rnn, _, [x, y, r, lr, z]}, _) do
    quote do
      {:rnn, Cmatrix.zeros(unquote(x), unquote(y)), unquote(gen_rnn(x, y, z, r)), unquote(lr)}
    end
  end

  # LSTM {:lstm, c_init,[]}
  def parse({:lstm, _, [x, y, r]}, _) do
    quote do
      {:lstm, Cmatrix.new(unquote(x), unquote(y)), Cmatrix.zeros(unquote(x), unquote(y)),
       unquote(gen_lstm(x, y, r)), 0.1}
    end
  end

  def parse({:lstm, _, [x, y, r, lr]}, _) do
    quote do
      {:lstm, Cmatrix.new(unquote(x), unquote(y)), Cmatrix.zeros(unquote(x), unquote(y)),
       unquote(gen_lstm(x, y, r)), unquote(lr)}
    end
  end

  def parse({:lstm, _, [x, y, r, lr, z]}, _) do
    quote do
      {:lstm, Cmatrix.new(unquote(x), unquote(y)), Cmatrix.zeros(unquote(x), unquote(y)),
       unquote(gen_lstm(x, y, z, r)), unquote(lr)}
    end
  end

  # bias
  def parse({:b, _, [x]}, _) do
    quote do
      {:bias, Cmatrix.zeros(1, unquote(x)), 0.1, Cmatrix.zeros(1, unquote(x))}
    end
  end

  def parse({:b, _, [x, lr]}, _) do
    quote do
      {:bias, Cmatrix.zeros(1, unquote(x)), unquote(lr), Cmatrix.zeros(1, unquote(x))}
    end
  end

  # sigmoid
  def parse({:sigmoid, _, nil}, _) do
    quote do
      {:function, fn x -> DP.sigmoid(x) end, fn x -> DP.dsigmoid(x) end, :sigmoid}
    end
  end

  # identity
  def parse({:ident, _, nil}, _) do
    quote do
      {:function, fn x -> DP.ident(x) end, fn x -> DP.dident(x) end, :ident}
    end
  end

  # relu
  def parse({:relu, _, nil}, _) do
    quote do
      {:function, fn x -> DP.relu(x) end, fn x -> DP.drelu(x) end, :relu}
    end
  end

  # softmax
  def parse({:softmax, _, nil}, _) do
    quote do
      {:softmax, fn x -> DP.softmax(x) end, fn x -> DP.dsoftmax(x) end}
    end
  end

  # flatten
  def parse({:flatten, _, nil}, _) do
    quote do
      {:flatten}
    end
  end

  def parse({x, _, nil}, _) do
    x
  end

  def parse({:|>, _, exp}, arg) do
    parse(exp, arg)
  end

  def parse([{arg, _, nil}, exp], arg) do
    [parse(exp, arg)]
  end

  def parse([exp1, exp2], arg) do
    Enum.reverse([parse(exp2, arg)] ++ Enum.reverse(parse(exp1, arg)))
  end

  def parse(x, _) do
    :io.write(x)
    IO.puts("Syntax error in defnetwork")
  end

  def gen_rnn(_, _, 0) do
    []
  end

  def gen_rnn(x, y, r) do
    quote do
      [
        {Cmatrix.new(unquote(x), unquote(y)), Cmatrix.new(unquote(x), unquote(y)),
         Cmatrix.zeros(1, unquote(y))}
        | unquote(gen_rnn(x, y, r - 1))
      ]
    end
  end

  # [{wx1,wh1,b1},..{wxr,whr,br}]
  def gen_rnn(_, _, _, 0) do
    []
  end

  def gen_rnn(x, y, z, r) do
    quote do
      [
        {Cmatrix.new(unquote(x), unquote(y), unquote(z)),
         Cmatrix.new(unquote(x), unquote(y), unquote(z)), Cmatrix.zeros(1, unquote(y))}
        | unquote(gen_rnn(x, y, z, r - 1))
      ]
    end
  end

  def gen_lstm(_, _, 0) do
    []
  end

  def gen_lstm(x, y, r) do
    quote do
      [
        {Cmatrix.new(unquote(x), unquote(y * 4)), Cmatrix.new(unquote(x), unquote(y * 4)),
         Cmatrix.zeros(unquote(x), unquote(y * 4))}
        | unquote(gen_rnn(x, y, r - 1))
      ]
    end
  end

  # wx = wx(f),wx(g),wx(i),wx(o)
  # wh = wh(f),wh(g),wh(i),wh(o)
  # [{wx1,wh1,b1},...,{wxr,whr,br}]
  def gen_lstm(_, _, _, 0) do
    []
  end

  def gen_lstm(x, y, z, r) do
    quote do
      [
        {Cmatrix.new(unquote(x), unquote(y * 4), unquote(z)),
         Cmatrix.new(unquote(x), unquote(y * 4), unquote(z)),
         Cmatrix.zeros(unquote(x), unquote(y * 4))}
        | unquote(gen_rnn(x, y, r - 1))
      ]
    end
  end
end
