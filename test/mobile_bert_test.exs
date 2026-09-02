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

  # A zip carrying only vocab.txt stands in for a model here: init/1 reads the
  # vocabulary before it builds the interpreter, so the vocabulary checks are
  # reached and what stops it afterwards is that a zip is not a model.
  defp vocab_only_model(dir, name, vocab) do
    path = Path.join(dir, name)
    {:ok, {_, bytes}} = :zip.create(~c"model.zip", [{~c"vocab.txt", vocab}], [:memory])
    File.write!(path, bytes)
    path
  end

  @tag :tmp_dir
  test "a vocabulary missing a special token is refused by name", %{tmp_dir: dir} do
    path = vocab_only_model(dir, "no_cls.tflite", "[SEP]\n[UNK]\nhello\n")

    error = assert_raise ArgumentError, fn -> MobileBert.init(path) end
    assert error.message =~ path
    assert error.message =~ "[CLS]"
  end

  @tag :tmp_dir
  test "a CRLF vocabulary reads the same as an LF one", %{tmp_dir: dir} do
    path = vocab_only_model(dir, "crlf.tflite", "[CLS]\r\n[SEP]\r\n[UNK]\r\nhello\r\n")

    error = assert_raise ArgumentError, fn -> MobileBert.init(path) end
    assert error.message =~ "could not be loaded"
  end

  # Enum.take/2 with a negative count takes from the end, so a query past the
  # limit kept the tail of the content, overran the sequence, and failed later
  # in set_data as a MatchError.
  test "a query that leaves no room for content is refused rather than run on the tail" do
    vocab = Map.new(Enum.with_index(["[CLS]", "[SEP]", "[UNK]", "a", "b"]))
    long_query = String.duplicate("a ", 400)

    error =
      assert_raise ArgumentError, fn ->
        MobileBert.preprocessing(vocab, long_query, "b b b")
      end

    assert error.message =~ "384"
  end

  test "a token the vocabulary cannot map is reported by name" do
    error = assert_raise ArgumentError, fn -> MobileBert.preprocessing(%{}, "hi", "there") end
    assert error.message =~ "vocabulary"
  end

  # Every step of run/3 was matched against :ok or handed straight to Nx, so an
  # interpreter another process held, or a tensor of another shape, was a
  # MatchError or an error from inside Nx.
  test "run/3 names the step the model refused" do
    interpreter =
      TFLiteElixir.Interpreter.new!("test/test_data/mobilenet_v2_1.0_224_inat_bird_quant.tflite")

    not_bert = TFLiteElixir.Interpreter.tensor(interpreter, 0)

    bert = %MobileBert{
      interpreter: interpreter,
      vocab_map: Map.new(Enum.with_index(["[CLS]", "[SEP]", "[UNK]", "hello", "world"])),
      tensors: %{
        input_ids: not_bert,
        input_mask: not_bert,
        segment_ids: not_bert,
        end_logits: not_bert,
        start_logits: not_bert
      }
    }

    error = assert_raise RuntimeError, fn -> MobileBert.run(bert, "hello", "world") end
    assert error.message =~ "input_ids"
  end
end
