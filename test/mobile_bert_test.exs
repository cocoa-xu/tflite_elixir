defmodule TFLiteElixir.MobileBertTest do
  use ExUnit.Case

  alias TFLiteElixir.MobileBert

  # init/1 fed whatever get_associated_file/2 answered straight into
  # String.split/2, so a model carrying no vocab.txt raised out of String,
  # naming neither the model nor the file it was looking for.
  test "a model with no vocab.txt is refused by name" do
    without_vocab = "test/test_data/mobilenet_v2_1.0_224_inat_bird_quant.tflite"

    error =
      assert_raise ArgumentError, fn ->
        MobileBert.init(without_vocab)
      end

    assert error.message =~ without_vocab
    assert error.message =~ "vocab.txt"
  end
end
