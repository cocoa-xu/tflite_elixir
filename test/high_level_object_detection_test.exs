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

    [%{class_id: 16, score: score, label: nil, bbox: [15, -6, 1177, 961]}] =
      ObjectDetection.predict(pid, @input_path)

    assert_in_delta score, 0.934, 0.05

    [%{class_id: 16, bbox: [15, -6, 1177, 961]}] =
      ObjectDetection.predict(pid, StbImage.read_file!(@input_path))

    [%{class_id: 16, bbox: [15, -6, 1177, 961]}] =
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

  # The boxes are in source image pixels. They used to come back in the model
  # input's 300x300 instead, because the letterbox scale was applied twice and
  # cancelled itself out, so every box was a quarter of its true size on this
  # image. Fixed numbers alone would not have caught that going back: these
  # assertions hold only if the box is measured against the image.
  test "ObjectDetection answers in source image coordinates" do
    {:ok, pid} = ObjectDetection.start(@model_path)
    %StbImage{shape: {h, w, _}} = StbImage.read_file!(@input_path)
    [%{bbox: [ymin, xmin, ymax, xmax]}] = ObjectDetection.predict(pid, @input_path)

    # the cat fills the frame, so the box covers most of the image, and most of
    # the image is well outside anything the 300x300 input could express
    assert ymax > div(h, 2)
    assert xmax > div(w, 2)
    assert ymax <= h
    assert xmax <= w
    assert ymin < ymax
    assert xmin < xmax
  end

  @tag :require_tpu
  test "ObjectDetection on a TPU" do
    {:ok, pid} = ObjectDetection.start(@edgetpu_model_path, use_tpu: true, tpu: "")

    [%{class_id: 16, score: score}] = ObjectDetection.predict(pid, @input_path)
    assert score > 0.4
  end

  # cat.jpeg is very nearly square, so it only ever pads rows. These two shapes
  # drive the letterbox down each branch and check the box lands back inside the
  # source image, which is what padding in the wrong place would break.
  test "ObjectDetection letterboxes both ways" do
    {:ok, pid} = ObjectDetection.start(@model_path)
    cat = StbImage.read_file!(@input_path)

    for {h, w} <- [{1200, 400}, {400, 1200}] do
      [%{class_id: class_id, bbox: [ymin, xmin, ymax, xmax]}] =
        ObjectDetection.predict(pid, StbImage.resize(cat, h, w))

      assert 16 == class_id

      # the model is free to predict a little past the edge, so this is not a
      # containment check: it says the box came back on the source image's
      # scale rather than the padded input's, and still covers the animal.
      assert ymin > -0.05 * h and ymax < 1.05 * h
      assert xmin > -0.05 * w and xmax < 1.05 * w
      assert ymax - ymin > 0.5 * h
      assert xmax - xmin > 0.5 * w
    end
  end

  test "ObjectDetection answers the same whichever form the image arrives in" do
    {:ok, pid} = ObjectDetection.start(@model_path)
    cat = StbImage.read_file!(@input_path)

    by_path = ObjectDetection.predict(pid, @input_path)

    assert by_path == ObjectDetection.predict(pid, cat)
    assert by_path == ObjectDetection.predict(pid, StbImage.to_nx(cat))
  end

  # a shape this far from square used to truncate its short side to zero, which
  # StbImage.resize has no clause for, so the call took the server down and
  # every later caller got :noproc.
  test "ObjectDetection survives an extreme aspect ratio" do
    {:ok, pid} = ObjectDetection.start(@model_path)
    cat = StbImage.read_file!(@input_path)

    for {h, w} <- [{1, 1}, {1, 640}, {640, 1}, {2, 999}, {999, 2}, {37, 91}] do
      assert is_list(ObjectDetection.predict(pid, StbImage.resize(cat, h, w)))
      assert Process.alive?(pid)
    end
  end
end
