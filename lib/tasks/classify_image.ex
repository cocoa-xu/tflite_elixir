defmodule Mix.Tasks.ClassifyImage do
  @moduledoc """
  Image classification mix task: `mix help classify_image`

  Command line arguments:

  - `-m`, `--model`: *Required*. File path of .tflite file.
  - `-i`, `--input`: *Required*. Image to be classified.
  - `-l`, `--labels`: File path of labels file.
  - `-k`, `--top`: Default to `1`. Max number of classification results.
  - `-t`, `--threshold`: Default to `0.0`. Classification score threshold.
  - `-c`, `--count`: Default to `1`. Number of times to run inference.
  - `-a`, `--mean`: Default to `128.0`. Mean value for input normalization.
  - `-s`, `--std`: Default to `128.0`. STD value for input normalization.
  - `-j`, `--jobs`: Number of threads for the interpreter (only valid for CPU).
  - `--use-tpu`: Default to false. Add this option to use Coral device.
  - `--tpu`: Default to `""`. Coral device name.
    - `""`      -- any TPU device
    - `"usb"`   -- any TPU device on USB bus
    - `"pci"`   -- any TPU device on PCIe bus
    - `":N"`    -- N-th TPU device, e.g. `":0"`
    - `"usb:N"` -- N-th TPU device on USB bus, e.g. `"usb:0"`
    - `"pci:N"` -- N-th TPU device on PCIe bus, e.g. `"pci:0"`

  Code based on [classify_image.py](https://github.com/google-coral/pycoral/blob/master/examples/classify_image.py)
  """

  use Mix.Task

  alias TFLiteElixir.Interpreter
  alias TFLiteElixir.InterpreterBuilder
  alias TFLiteElixir.TFLiteTensor
  alias TFLiteElixir.FlatBufferModel

  @shortdoc "Image Classification"
  def run(argv) do
    {args, _, _} =
      OptionParser.parse(argv,
        strict: [
          model: :string,
          input: :string,
          labels: :string,
          top: :integer,
          threshold: :float,
          count: :integer,
          mean: :float,
          std: :float,
          use_tpu: :boolean,
          tpu: :string,
          jobs: :integer
        ],
        aliases: [
          m: :model,
          i: :input,
          l: :labels,
          k: :top,
          t: :threshold,
          c: :count,
          a: :mean,
          s: :std,
          j: :jobs
        ]
      )

    default_values = [
      top: 1,
      threshold: 0.0,
      count: 1,
      mean: 128.0,
      std: 128.0,
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

    model = load_model(args[:model])
    input_image = load_input(args[:input])
    labels = load_labels(args[:labels])

    tpu_context =
      if args[:use_tpu] do
        TFLiteElixir.Coral.get_edge_tpu_context!(device: args[:tpu])
      else
        nil
      end

    interpreter = make_interpreter(model, args[:jobs], args[:use_tpu], tpu_context)
    :ok = Interpreter.allocate_tensors(interpreter)

    [input_tensor_number | _] = Interpreter.inputs!(interpreter)
    [output_tensor_number | _] = Interpreter.outputs!(interpreter)
    %TFLiteTensor{} = input_tensor = Interpreter.tensor(interpreter, input_tensor_number)

    if input_tensor.type != {:u, 8} do
      raise ArgumentError, "Only support uint8 input type."
    end

    {h, w} =
      case input_tensor.shape do
        {_n, h, w, _c} ->
          {h, w}

        {_n, h, w} ->
          {h, w}

        shape ->
          raise RuntimeError, "not sure the input shape, got #{inspect(shape)}"
      end

    input_image = StbImage.resize(input_image, h, w)

    [scale] = input_tensor.quantization_params.scale
    [zero_point] = input_tensor.quantization_params.zero_point
    mean = args[:mean]
    std = args[:std]

    if abs(scale * std - 1) < 0.00001 and abs(mean - zero_point) < 0.00001 do
      # Input data does not require preprocessing.
      %StbImage{data: input_data} = input_image
      input_data
    else
      # Input data requires preprocessing
      StbImage.to_nx(input_image)
      |> Nx.subtract(mean)
      |> Nx.divide(std * scale)
      |> Nx.add(zero_point)
      |> Nx.clip(0, 255)
      |> Nx.as_type(:u8)
      |> Nx.to_binary()
    end
    |> then(&TFLiteTensor.set_data(input_tensor, &1))

    IO.puts("----INFERENCE TIME----")

    for _ <- 1..args[:count] do
      start_time = :os.system_time(:microsecond)
      Interpreter.invoke!(interpreter)
      end_time = :os.system_time(:microsecond)
      inference_time = (end_time - start_time) / 1000.0
      IO.puts("#{Float.round(inference_time, 1)}ms")
    end

    output_data = Interpreter.output_tensor!(interpreter, 0)
    %TFLiteTensor{} = output_tensor = Interpreter.tensor(interpreter, output_tensor_number)
    scores = get_scores(output_data, output_tensor)
    sorted_indices = Nx.argsort(scores, direction: :desc)
    top_k = Nx.take(sorted_indices, Nx.iota({args[:top]}))
    scores = Nx.to_flat_list(Nx.take(scores, top_k))
    top_k = Nx.to_flat_list(top_k)

    IO.puts("-------RESULTS--------")

    if labels != nil do
      Enum.zip(top_k, scores)
      |> Enum.each(fn {class_id, score} ->
        IO.puts("#{Enum.at(labels, class_id)}: #{Float.round(score, 5)}")
      end)
    else
      Enum.zip(top_k, scores)
      |> Enum.each(fn {class_id, score} ->
        IO.puts("#{class_id}: #{Float.round(score, 5)}")
      end)
    end
  end

  defp load_model(nil) do
    raise ArgumentError, "empty value for argument '--model'"
  end

  defp load_model(model_path) do
    FlatBufferModel.build_from_buffer(File.read!(model_path))
  end

  defp load_input(nil) do
    raise ArgumentError, "empty value for argument '--input'"
  end

  defp load_input(input_path) do
    with {:ok, input_image} <- StbImage.read_file(input_path) do
      input_image
    else
      {:error, error} ->
        raise RuntimeError, error
    end
  end

  defp load_labels(nil), do: nil

  defp load_labels(label_file_path) do
    File.read!(label_file_path)
    |> String.split("\n")
  end

  defp make_interpreter(model, num_jobs, false, _tpu_context) do
    resolver = TFLiteElixir.Ops.Builtin.BuiltinResolver.new!()
    builder = InterpreterBuilder.new!(model, resolver)
    interpreter = Interpreter.new!()
    InterpreterBuilder.set_num_threads!(builder, num_jobs)
    :ok = InterpreterBuilder.build!(builder, interpreter)
    Interpreter.set_num_threads!(interpreter, num_jobs)
    interpreter
  end

  defp make_interpreter(model, _num_jobs, true, tpu_context) do
    TFLiteElixir.Coral.make_edge_tpu_interpreter!(model, tpu_context)
  end

  defp get_scores(output_data, %TFLiteTensor{type: dtype = {:u, _}} = output_tensor) do
    scale = Nx.tensor(output_tensor.quantization_params.scale)
    zero_point = Nx.tensor(output_tensor.quantization_params.zero_point)

    Nx.from_binary(output_data, dtype)
    |> Nx.as_type({:s, 64})
    |> Nx.subtract(zero_point)
    |> Nx.multiply(scale)
  end

  defp get_scores(output_data, %TFLiteTensor{type: dtype = {:s, _}} = output_tensor) do
    [scale] = output_tensor.quantization_params.scale
    [zero_point] = output_tensor.quantization_params.zero_point

    Nx.from_binary(output_data, dtype)
    |> Nx.as_type({:s, 64})
    |> Nx.subtract(zero_point)
    |> Nx.multiply(scale)
  end

  defp get_scores(output_data, %TFLiteTensor{type: dtype}) do
    Nx.from_binary(output_data, dtype)
  end
end
