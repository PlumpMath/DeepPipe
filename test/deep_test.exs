defmodule DLTest do
  use ExUnit.Case

  test "error function" do
    # assert DL.test1() == [[0.7043825919854788, 0.7043825919854788]]
    assert DP.cross_entropy([[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]], [
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
           ]) == 0.510825457099338

    assert DP.cross_entropy([[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]], [
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
           ]) == 2.302584092994546
  end

  test "BLAS" do
    a = Cmatrix.to_matrex([[1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1]])
    b = Cmatrix.to_matrex([[2, 0, 1], [0, 1, 2], [1, 0, 2]])
    c = Cmatrix.to_matrex([[1, 2, 3], [2, 3, 4]])
    d = Cmatrix.to_matrex([[1, 2, 1, 0], [0, 1, 2, 3], [3, 0, 1, 2], [2, 4, 0, 1]])
    e = Cmatrix.to_matrex([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert Cmatrix.part(a, 2, 2, 2, 2) |> Cmatrix.to_list() == [[1, 2], [0, 1]]

    assert Cmatrix.sparse(a, 2) |> Cmatrix.to_list() == [
             [0, 2, 3, 0],
             [0, 0, 0, 3],
             [3, 0, 0, 2],
             [0, 3, 0, 0]
           ]

    assert Cmatrix.convolute(a, b) |> Cmatrix.to_list() == [[15, 16], [6, 15]]

    assert Cmatrix.pad(c, 1) |> Cmatrix.to_list() == [
             [0, 0, 0, 0, 0],
             [0, 1, 2, 3, 0],
             [0, 2, 3, 4, 0],
             [0, 0, 0, 0, 0]
           ]

    assert Cmatrix.pad(c, 0) |> Cmatrix.to_list() == [[1, 2, 3], [2, 3, 4]]
    assert Cmatrix.pool(d, 2) |> Cmatrix.to_list() == [[2, 3], [4, 2]]
    assert Cmatrix.rotate180(e) |> Cmatrix.to_list() == [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
  end
end
