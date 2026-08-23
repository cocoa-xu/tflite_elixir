defmodule TFLiteElixir.Tokenizer.WordpieceTokenizer.Test do
  use ExUnit.Case

  alias TFLiteElixir.Tokenizer.WordpieceTokenizer

  @mini_vocab %{
    "hello" => 0,
    "world" => 1,
    "una" => 2,
    "##ffa" => 3,
    "##ble" => 4
  }

  defp mini_vocab, do: @mini_vocab

  test "wordpiece tokenizer" do
    assert ["una", "##ffa", "##ble"] == WordpieceTokenizer.tokenize("unaffable", mini_vocab())
    assert ["hello", "world"] == WordpieceTokenizer.tokenize("hello world", mini_vocab())
    assert ["[UNK]", "[UNK]"] == WordpieceTokenizer.tokenize("not exists", mini_vocab())
  end

  test "wordpiece tokenizer, more than 200 letters in a single word" do
    # An over-long word is named unknown rather than dropped. Returning nothing
    # at all, which is what this did up to tflite_beam 0.4.0-rc5, lost the word
    # without saying so, and a caller lining tokens up against their input had
    # no way to tell.
    assert ["[UNK]"] == WordpieceTokenizer.tokenize(String.duplicate("a", 201), mini_vocab())
  end
end
