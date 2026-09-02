defmodule TFLiteElixir.ImageClassification.Test do
  use ExUnit.Case

  alias TFLiteElixir.ImageClassification

  test "ImageClassification" do
    filename = Path.join([__DIR__, "test_data", "mobilenet_v2_1.0_224_inat_bird_quant.tflite"])
    input_path = Path.join([__DIR__, "test_data", "parrot.jpeg"])

    {:ok, pid} = ImageClassification.start(filename)
    %{class_id: 923, score: _score} = ImageClassification.predict(pid, input_path)

    %{class_id: 923, score: _score} =
      ImageClassification.predict(pid, StbImage.read_file!(input_path))

    %{class_id: 923, score: _score} =
      ImageClassification.predict(pid, StbImage.to_nx(StbImage.read_file!(input_path)))

    assert :ok == ImageClassification.set_label_from_associated_file(pid, "inat_bird_labels.txt")

    %{class_id: 923, label: "Ara macao (Scarlet Macaw)", score: _score} =
      ImageClassification.predict(pid, input_path)

    [
      %{class_id: 923, label: "Ara macao (Scarlet Macaw)", score: _score1},
      %{
        class_id: 837,
        label: "Platycercus elegans (Crimson Rosella)",
        score: _score2
      },
      %{
        class_id: 245,
        label: "Coracias caudatus (Lilac-breasted Roller)",
        score: _score3
      }
    ] = ImageClassification.predict(pid, input_path, top_k: 3)
  end

  # A wrong path is the first mistake anyone makes, and it answered with a
  # FunctionClauseError wrapped in the error tuple from start/2, or took the
  # server down from predict/3 so every later caller got :noproc.
  test "a model or image that cannot be read is named, and the classifier survives it" do
    assert {:error, reason} = ImageClassification.start("/no/such/model.tflite")
    assert is_binary(reason)
    assert reason =~ "/no/such/model.tflite"

    filename = Path.join([__DIR__, "test_data", "mobilenet_v2_1.0_224_inat_bird_quant.tflite"])
    input_path = Path.join([__DIR__, "test_data", "parrot.jpeg"])
    {:ok, pid} = ImageClassification.start(filename)
    assert {:error, reason} = ImageClassification.predict(pid, "/no/such/image.jpeg")
    assert reason =~ "/no/such/image.jpeg"
    assert Process.alive?(pid)
    assert %{class_id: 923} = ImageClassification.predict(pid, input_path)
  end
end
