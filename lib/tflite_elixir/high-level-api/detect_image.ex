defmodule TFLiteElixir.ObjectDetection do
  @moduledoc """
  Experimental object detection module.
  """

  alias TFLiteElixir.Interpreter
  alias TFLiteElixir.InterpreterBuilder
  alias TFLiteElixir.TFLiteTensor
  alias TFLiteElixir.FlatBufferModel

  use GenServer
  # TFLiteElixir.Interpreter.Server settled on 30s and let the caller say
  # otherwise. These went through GenServer.call's 5s default instead, which is
  # not a budget anyone chose: it is short enough that a model this project
  # ships precompiled binaries for, on a board this project ships them to,
  # cannot finish, and the caller sees an exit rather than a slow answer.
  @default_timeout 30_000

  @type detection :: %{
          class_id: integer(),
          score: float(),
          label: String.t() | nil,
          bbox: [integer()]
        }

  @doc """
  Start a detector for `model`, a path to a `.tflite` file or its contents.

  Options, all optional: `:threshold` (0.4) the score below which a detection is
  dropped, `:labels` (nil) a list of labels or the path to a file holding one per
  line, `:jobs` (`System.schedulers_online/0`) the interpreter's thread count,
  `:use_tpu` (false) and `:tpu` ("") to run on a named Edge TPU.
  """
  @spec start(any, any) :: :ignore | {:error, any} | {:ok, pid}
  def start(model, opts \\ []) do
    GenServer.start(__MODULE__, {model, opts})
  end

  @doc """
  Run the model against an image.

  ## Options

    * `:timeout` - how long to wait for the answer, in milliseconds, or
      `:infinity`. Defaults to `#{@default_timeout}`. Raise it for a large model
      or a slow board; inference is local and bounded, so waiting is the right
      answer more often than giving up.
  """
  @spec predict(pid(), binary() | %StbImage{} | %Nx.Tensor{}, Keyword.t()) :: [detection()]
  def predict(pid, input_path, opts \\ [])

  def predict(pid, input_path, opts) when is_binary(input_path) and is_list(opts) do
    GenServer.call(pid, {:predict, {:image_path, input_path}, opts}, timeout(opts))
  end

  def predict(pid, stb_image, opts) when is_struct(stb_image, StbImage) and is_list(opts) do
    GenServer.call(pid, {:predict, {:stb_image, stb_image}, opts}, timeout(opts))
  end

  def predict(pid, image_data, opts) when is_struct(image_data, Nx.Tensor) and is_list(opts) do
    GenServer.call(pid, {:predict, {:nx_tensor, image_data}, opts}, timeout(opts))
  end

  @doc """
  Give the detector its labels, either as a list or as the path to a file
  holding one label per line.
  """
  @spec set_label(pid, String.t() | [String.t()]) :: :ok
  def set_label(pid, label_file) when is_binary(label_file) do
    GenServer.call(pid, {:set_label, label_file}, @default_timeout)
  end

  def set_label(pid, labels) when is_list(labels) do
    GenServer.call(pid, {:set_label, labels}, @default_timeout)
  end

  defp timeout(opts), do: Keyword.get(opts, :timeout, @default_timeout)

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

    # A path that named no model answered from start/2 as a FunctionClauseError
    # wrapped in the error tuple, with the reason two levels down in a stack
    # frame, and a model whose graph would not prepare as a MatchError.
    case load_model(model_path) do
      {:error, reason} ->
        {:stop, "cannot load model #{model_path}: #{reason}"}

      %FlatBufferModel{} = model ->
        prepare(model, model_path, args)
    end
  end

  # Everything raised while the model is set up, a model with the wrong number
  # of outputs, a delegate that will not attach, a labels file that is not
  # there, used to come back from start/2 as the exception wrapped in the
  # error tuple.
  defp prepare(model, model_path, args) do
    tpu_context =
      if args[:use_tpu] do
        TFLiteElixir.Coral.get_edge_tpu_context!(device: args[:tpu])
      else
        nil
      end

    interpreter = make_interpreter(model, args[:jobs], args[:use_tpu], tpu_context)

    with :ok <- Interpreter.allocate_tensors(interpreter),
         {:ok, outputs} <- Interpreter.outputs(interpreter) do
      if Enum.count(outputs) != 4 do
        raise ArgumentError, "Object detection models should have 4 output tensors"
      end

      {:ok,
       %{
         model_path: model_path,
         interpreter: interpreter,
         opts: args,
         labels: load_labels(args[:labels])
       }}
    else
      {:error, reason} -> {:stop, "cannot prepare #{model_path}: #{reason}"}
    end
  rescue
    error -> {:stop, "cannot prepare #{model_path}: #{Exception.message(error)}"}
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
  rescue
    # anything raised in here used to take the detector down, and every caller
    # queued behind it, for one caller's image or option
    error -> {:reply, {:error, Exception.message(error)}, state}
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

    # an image far enough from square truncates its short side to zero, and
    # StbImage.resize has no clause for a zero side, so the call raised and took
    # the server down with it. One row or column keeps the aspect ratio as close
    # as it can still be represented.
    {resized_h, resized_w} = {max(trunc(h * scale), 1), max(trunc(w * scale), 1)}

    %StbImage{shape: {_, _, channels}, data: resized} =
      StbImage.resize(input_image, resized_h, resized_w)

    letterbox(resized, resized_w, channels, height - resized_h, width - resized_w, width)
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
    case Interpreter.get_signature_defs!(interpreter) do
      nil ->
        output_tensor_ids_by_shape(interpreter, output_tensor_numbers)

      signatures when map_size(signatures) > 1 ->
        raise ArgumentError, "Only support model with one signature."

      signatures ->
        # Map.values/1 was indexed with :outputs as though it were still a map,
        # so this answered four nils whenever a model did declare its outputs.
        # No fixture here carries a signature, so nothing noticed. The names are
        # the model's and arrive as binaries, like every other name this library
        # reads out of one.
        outputs = signatures |> Map.values() |> hd() |> Map.get(:outputs, %{})
        declared = Enum.map(~w(output_0 output_1 output_2 output_3), &Map.get(outputs, &1))

        if Enum.any?(declared, &is_nil/1) do
          output_tensor_ids_by_shape(interpreter, output_tensor_numbers)
        else
          List.to_tuple(declared)
        end
    end
  end

  defp output_tensor_ids_by_shape(interpreter, output_tensor_numbers) do
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

  # Padding is an append, so it is done on the binary. Going through Nx walks
  # every element of the destination instead: 2.1 s for a 640x640 input against
  # 0.1 ms here, for the same bytes.
  defp letterbox(binary, resized_w, channels, pad_rows, pad_cols, width) do
    body =
      if pad_cols == 0 do
        binary
      else
        row = resized_w * channels
        gap = :binary.copy(<<0>>, pad_cols * channels)

        for <<chunk::binary-size(^row) <- binary>>, into: <<>>, do: chunk <> gap
      end

    body <> :binary.copy(<<0>>, pad_rows * width * channels)
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
    case File.read(model_path) do
      {:ok, buffer} -> FlatBufferModel.build_from_buffer(buffer)
      {:error, reason} -> {:error, "cannot read model file: #{:file.format_error(reason)}"}
    end
  end

  defp load_input(nil) do
    raise ArgumentError, "empty value for argument '--input'"
  end

  defp load_input(input_path) do
    case StbImage.read_file(input_path) do
      {:ok, input_image} -> input_image
      {:error, error} -> raise RuntimeError, "cannot read image #{input_path}: #{error}"
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
