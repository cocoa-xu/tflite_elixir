defmodule TFLiteElixir.ObjectDetection do
  @moduledoc """
  Experimental object detection module.
  """

  alias TFLiteElixir.Interpreter
  alias TFLiteElixir.InterpreterBuilder
  alias TFLiteElixir.TFLiteTensor
  alias TFLiteElixir.FlatBufferModel

  use GenServer

  @type detection :: %{
          class_id: integer(),
          score: float(),
          label: String.t() | nil,
          bbox: [integer()]
        }

  @spec start(any, any) :: :ignore | {:error, any} | {:ok, pid}
  def start(model, opts \\ []) do
    GenServer.start(__MODULE__, {model, opts})
  end

  @spec predict(pid(), binary() | %StbImage{} | %Nx.Tensor{}, Keyword.t()) :: [detection()]
  def predict(pid, input_path, opts \\ [])

  def predict(pid, input_path, opts) when is_binary(input_path) and is_list(opts) do
    GenServer.call(pid, {:predict, {:image_path, input_path}, opts})
  end

  def predict(pid, stb_image, opts) when is_struct(stb_image, StbImage) and is_list(opts) do
    GenServer.call(pid, {:predict, {:stb_image, stb_image}, opts})
  end

  def predict(pid, image_data, opts) when is_struct(image_data, Nx.Tensor) and is_list(opts) do
    GenServer.call(pid, {:predict, {:nx_tensor, image_data}, opts})
  end

  @spec set_label(pid, String.t() | [String.t()]) :: :ok
  def set_label(pid, label_file) when is_binary(label_file) do
    GenServer.call(pid, {:set_label, label_file})
  end

  def set_label(pid, labels) when is_list(labels) do
    GenServer.call(pid, {:set_label, labels})
  end

  @impl true
  def init({model_path, opts}) do
    default_values = [
      threshold: 0.4,
      jobs: System.schedulers_online(),
      use_tpu: false,
      tpu: "",
      labels: nil
    ]

    args =
      Keyword.merge(opts, default_values, fn _k, user, default ->
        if user == nil do
          default
        else
          user
        end
      end)

    model = load_model(model_path)

    tpu_context =
      if args[:use_tpu] do
        TFLiteElixir.Coral.get_edge_tpu_context!(device: args[:tpu])
      else
        nil
      end

    interpreter = make_interpreter(model, args[:jobs], args[:use_tpu], tpu_context)
    :ok = Interpreter.allocate_tensors(interpreter)

    if Enum.count(Interpreter.outputs!(interpreter)) != 4 do
      raise ArgumentError, "Object detection models should have 4 output tensors"
    end

    {:ok,
     %{
       model_path: model_path,
       interpreter: interpreter,
       opts: args,
       labels: load_labels(args[:labels])
     }}
  end

  @impl true
  def handle_call(
        {:predict, {input_type, input_data}, pred_opts},
        _from,
        state = %{interpreter: interpreter, opts: opts, labels: labels}
      ) do
    opts = Keyword.merge(opts, pred_opts, fn _k, _, p -> p end)

    input_image =
      case input_type do
        :image_path -> load_input(input_data)
        :stb_image -> input_data
        :nx_tensor -> StbImage.from_nx(input_data)
      end

    {:reply, detect(interpreter, input_image, labels, opts[:threshold]), state}
  end

  @impl true
  def handle_call({:set_label, label_file}, _from, state) when is_binary(label_file) do
    {:reply, :ok, %{state | labels: load_labels(label_file)}}
  end

  @impl true
  def handle_call({:set_label, labels}, _from, state) when is_list(labels) do
    {:reply, :ok, %{state | labels: labels}}
  end

  defp detect(interpreter, %StbImage{shape: {h, w, _c}} = input_image, labels, threshold) do
    [input_tensor_number | _] = Interpreter.inputs!(interpreter)
    output_tensor_numbers = Interpreter.outputs!(interpreter)
    %TFLiteTensor{} = input_tensor = Interpreter.tensor(interpreter, input_tensor_number)

    if input_tensor.type != {:u, 8} do
      raise ArgumentError, "Only support uint8 input type."
    end

    {height, width} =
      case input_tensor.shape do
        {_n, height, width, _c} -> {height, width}
        {_n, height, width} -> {height, width}
        shape -> raise RuntimeError, "not sure the input shape, got #{inspect(shape)}"
      end

    # letterbox: keep the aspect ratio and leave the rest of the tensor zeroed
    scale = min(height / h, width / w)
    {resized_h, resized_w} = {trunc(h * scale), trunc(w * scale)}

    resized =
      StbImage.resize(input_image, resized_h, resized_w)
      |> StbImage.to_nx()
      |> Nx.new_axis(0)

    # Nx.put_slice into a zeroed tensor answers the same bytes but walks the
    # whole destination: 5.6 s per image on Nx 0.13's binary backend against
    # 0.4 s for the pad below.
    resized
    |> Nx.pad(0, [{0, 0, 0}, {0, height - resized_h, 0}, {0, width - resized_w, 0}, {0, 0, 0}])
    |> Nx.to_binary()
    # set_data is all-or-nothing and answers {:error, _} when the byte count is
    # wrong, which an RGBA image produces because nothing here asks StbImage for
    # three channels. Dropping that answer meant invoking on a tensor that had
    # not been written and reporting a confident classification of whatever the
    # arena held.
    |> then(&TFLiteTensor.set_data(input_tensor, &1))
    |> case do
      :ok -> :ok
      {:error, reason} -> raise ArgumentError, "cannot write the input tensor: #{reason}"
    end

    Interpreter.invoke!(interpreter)

    {count_id, scores_id, class_ids_id, boxes_id} =
      output_tensor_ids(interpreter, output_tensor_numbers)

    boxes = output_as_nx(interpreter, boxes_id) |> take_first_and_reshape()
    class_ids = output_as_nx(interpreter, class_ids_id) |> take_first_and_reshape()
    scores = output_as_nx(interpreter, scores_id) |> take_first_and_reshape()

    count =
      output_as_nx(interpreter, count_id)
      |> Nx.to_flat_list()
      |> hd()
      |> trunc()

    # the boxes come back normalised against the model input, where the image
    # sits letterboxed in the top left corner. Multiplying by the input side
    # gives a pixel inside that tensor, and dividing by the letterbox scale puts
    # it back in the source image, which is the space the caller works in.
    # sx was taken from the height and sy from the width, which only agreed
    # because every model here has a square input.
    {sy, sx} = {height / scale, width / scale}

    Enum.reduce(0..(count - 1)//1, [], fn index, acc ->
      score = scores |> Nx.take(index) |> Nx.to_flat_list() |> hd()

      if score >= threshold do
        # not multiplied by the scale here: sy and sx already divide by it, so
        # doing both cancelled out and left the box in model input pixels
        [ymin, xmin, ymax, xmax] =
          boxes
          |> Nx.take(index)
          |> Nx.to_flat_list()

        class_id = class_ids |> Nx.take(index) |> Nx.to_flat_list() |> hd() |> trunc()

        [
          %{
            class_id: class_id,
            score: score,
            label: labels && Enum.at(labels, class_id),
            bbox: [trunc(sy * ymin), trunc(sx * xmin), trunc(sy * ymax), trunc(sx * xmax)]
          }
          | acc
        ]
      else
        acc
      end
    end)
    |> Enum.reverse()
  end

  # the four outputs are not always in the same order, so prefer what the model
  # declares and fall back to telling them apart by shape
  defp output_tensor_ids(interpreter, output_tensor_numbers) do
    signature_list = Interpreter.get_signature_defs!(interpreter)

    if signature_list != nil do
      signature_list = Map.values(signature_list)

      if Enum.count(signature_list) > 1 do
        raise ArgumentError, "Only support model with one signature."
      end

      {signature_list[:outputs][:output_0], signature_list[:outputs][:output_1],
       signature_list[:outputs][:output_2], signature_list[:outputs][:output_3]}
    else
      %TFLiteTensor{} =
        output_tensor_3 = Interpreter.tensor(interpreter, Enum.at(output_tensor_numbers, 3))

      if output_tensor_3.shape == {1} do
        {Enum.at(output_tensor_numbers, 3), Enum.at(output_tensor_numbers, 2),
         Enum.at(output_tensor_numbers, 1), Enum.at(output_tensor_numbers, 0)}
      else
        {Enum.at(output_tensor_numbers, 2), Enum.at(output_tensor_numbers, 0),
         Enum.at(output_tensor_numbers, 3), Enum.at(output_tensor_numbers, 1)}
      end
    end
  end

  defp output_as_nx(interpreter, tensor_id) do
    Interpreter.tensor(interpreter, tensor_id)
    |> TFLiteTensor.to_nx(backend: Nx.BinaryBackend)
  end

  defp take_first_and_reshape(tensor) do
    shape = Tuple.delete_at(Nx.shape(tensor), 0)

    Nx.take(tensor, 0)
    |> Nx.reshape(shape)
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
      {:error, error} -> raise RuntimeError, error
    end
  end

  defp load_labels(nil), do: nil
  defp load_labels(labels) when is_list(labels), do: labels

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
end
