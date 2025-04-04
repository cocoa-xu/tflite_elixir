defmodule TFLiteElixir.Test.ClassifyImage do
  use ExUnit.Case

  import ExUnit.CaptureIO

  test "Classify Image (CPU)" do
    output =
      capture_io(fn ->
        Mix.Tasks.ClassifyImage.run(
          OptionParser.to_argv(
            model: "test/test_data/mobilenet_v2_1.0_224_inat_bird_quant.tflite",
            input: "test/test_data/parrot.jpeg",
            labels: "test/test_data/inat_bird_labels.txt",
            top: 3,
            threshold: 0.3,
            count: 1,
            mean: 128.0,
            std: 128.0,
            use_tpu: false,
            tpu: ""
          )
        )
      end)

    result =
      String.split(output, "\n")
      |> List.delete_at(0)
      |> List.delete_at(0)
      |> Enum.join("\n")

    assert result =~ "-------RESULTS--------"
    assert result =~ "Ara macao (Scarlet Macaw): 0.7"
    assert result =~ "Platycercus elegans (Crimson Rosella): 0."
    assert result =~ "Coracias caudatus (Lilac-breasted Roller): 0."
  end

  @tag :require_tpu
  test "Classify Image (TPU)" do
    Mix.Tasks.ClassifyImage.run(
      OptionParser.to_argv(
        model: "test/test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
        input: "test/test_data/parrot.jpeg",
        labels: "test/test_data/inat_bird_labels.txt",
        top: 3,
        threshold: 0.3,
        count: 1,
        mean: 128.0,
        std: 128.0,
        use_tpu: true,
        tpu: ""
      )
    )
  end
end
