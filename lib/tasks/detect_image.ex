defmodule Mix.Tasks.DetectImage do
  @moduledoc """
  Image detection mix task: `mix help detect_image`

  Command line arguments:

  - `-m`, `--model`: *Required*. File path of .tflite file.
  - `-i`, `--input`: *Required*. Image to process.
  - `-l`, `--labels`: File path of labels file.
  - `-t`, `--threshold`: Default to `0.4`. Score threshold for detected objects.
  - `-c`, `--count`: Default to `1`. Number of times to run inference.
  - `-j`, `--jobs`: Number of threads for the interpreter (only valid for CPU).
  - `--use-tpu`: Default to false. Add this option to use Coral device.
  - `--tpu`: Default to `""`. Coral device name.

  Code based on [detect_image.py](https://github.com/google-coral/pycoral/blob/master/examples/detect_image.py)
  """

  use Mix.Task

  alias TFLiteElixir.ObjectDetection

  @shortdoc "Object Detection"
  def run(argv) do
    {args, _, _} =
      OptionParser.parse(argv,
        strict: [
          model: :string,
          input: :string,
          labels: :string,
          threshold: :float,
          count: :integer,
          jobs: :integer,
          use_tpu: :boolean,
          tpu: :string
        ],
        aliases: [
          m: :model,
          i: :input,
          l: :labels,
          t: :threshold,
          c: :count,
          j: :jobs
        ]
      )

    default_values = [
      threshold: 0.4,
      count: 1,
      jobs: System.schedulers_online(),
      use_tpu: false,
      tpu: ""
    ]

    args =
      Keyword.merge(args, default_values, fn _k, user, default ->
        if user == nil do
          default
        else
          user
        end
      end)

    {:ok, pid} =
      ObjectDetection.start(args[:model],
        threshold: args[:threshold],
        jobs: args[:jobs],
        use_tpu: args[:use_tpu],
        tpu: args[:tpu],
        labels: args[:labels]
      )

    IO.puts("----INFERENCE TIME----")

    detections =
      Enum.reduce(1..args[:count], nil, fn _, _ ->
        start_time = :os.system_time(:microsecond)
        detections = ObjectDetection.predict(pid, args[:input])
        end_time = :os.system_time(:microsecond)
        inference_time = (end_time - start_time) / 1000.0
        IO.puts("#{Float.round(inference_time, 1)}ms")
        detections
      end)

    Enum.each(detections, fn %{class_id: class_id, score: score, label: label, bbox: bbox} ->
      IO.puts("#{label || class_id}")
      IO.puts("  id   : #{class_id}")
      IO.puts("  score: #{Float.round(score, 3)}")
      IO.puts("  bbox : #{inspect(bbox)}")
    end)
  end
end
