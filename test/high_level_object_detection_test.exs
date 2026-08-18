defmodule TFLiteElixir.ObjectDetection.Test do
  use ExUnit.Case

  alias TFLiteElixir.ObjectDetection

  @model_path Path.join([__DIR__, "test_data", "ssd_mobilenet_v2_coco_quant_postprocess.tflite"])
  @edgetpu_model_path Path.join([
                        __DIR__,
                        "test_data",
                        "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
                      ])
  @input_path Path.join([__DIR__, "test_data", "cat.jpeg"])
  @labels_path Path.join([__DIR__, "test_data", "coco_labels.txt"])

  test "ObjectDetection" do
    {:ok, pid} = ObjectDetection.start(@model_path)

    [%{class_id: 16, score: score, label: nil, bbox: [3, -1, 294, 240]}] =
      ObjectDetection.predict(pid, @input_path)

    assert_in_delta score, 0.934, 0.05

    [%{class_id: 16, bbox: [3, -1, 294, 240]}] =
      ObjectDetection.predict(pid, StbImage.read_file!(@input_path))

    [%{class_id: 16, bbox: [3, -1, 294, 240]}] =
      ObjectDetection.predict(pid, StbImage.to_nx(StbImage.read_file!(@input_path)))
  end

  test "ObjectDetection with labels" do
    {:ok, pid} = ObjectDetection.start(@model_path, labels: @labels_path)

    [%{class_id: 16, label: "cat"}] = ObjectDetection.predict(pid, @input_path)

    assert :ok == ObjectDetection.set_label(pid, ["a", "b"])
    [%{class_id: 16, label: nil}] = ObjectDetection.predict(pid, @input_path)
  end

  test "ObjectDetection honours the threshold" do
    {:ok, pid} = ObjectDetection.start(@model_path, threshold: 0.99)
    assert [] == ObjectDetection.predict(pid, @input_path)

    # per-call options override the ones given at start
    [%{class_id: 16}] = ObjectDetection.predict(pid, @input_path, threshold: 0.4)
  end

  @tag :require_tpu
  test "ObjectDetection on a TPU" do
    {:ok, pid} = ObjectDetection.start(@edgetpu_model_path, use_tpu: true, tpu: "")

    [%{class_id: 16, score: score}] = ObjectDetection.predict(pid, @input_path)
    assert score > 0.4
  end
end
