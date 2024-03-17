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
end
