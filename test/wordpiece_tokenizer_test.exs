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
    # Pinned to what tflite_beam 0.4.0-rc5 does, which is to drop the word and
    # return nothing at all. rc6 reports it as ["[UNK]"] instead, because
    # silently losing input is worse than naming it unknown. This assertion is
    # the one that has to change in the same commit as the dependency bump, and
    # it will fail loudly at that moment rather than let the old behaviour
    # through.
    assert [] == WordpieceTokenizer.tokenize(String.duplicate("a", 201), mini_vocab())
  end
end
