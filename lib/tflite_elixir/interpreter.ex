defmodule TFLiteElixir.Interpreter do
  @moduledoc """
  An interpreter for a graph of nodes that input and output from tensors.
  """
  import TFLiteElixir.Errorize

  alias TFLiteElixir.TFLiteTensor
  alias TFLiteElixir.TFLiteQuantizationParams
  alias TFLiteElixir.Interpreter

  @type nif_resource_ok :: {:ok, reference()}
  @type nif_error :: {:error, String.t()}

  @doc """
  New interpreter
  """
  @spec new() :: nif_resource_ok() | nif_error()
  def new() do
    :tflite_beam_interpreter.new()
  end

  deferror(new())

  @doc """
  New interpreter with model filepath
  """
  @spec new(String.t()) :: nif_resource_ok() | nif_error()
  def new(model_path) do
    :tflite_beam_interpreter.new(model_path)
  end

  deferror(new(model_path))

  @doc """
  New interpreter with model buffer
  """
  @spec new_from_buffer(binary()) :: nif_resource_ok() | nif_error()
  def new_from_buffer(model_buffer) do
    :tflite_beam_interpreter.new_from_buffer(model_buffer)
  end

  @doc """
  Provide a list of tensor indexes that are inputs to the model.
  Each index is bound check and this modifies the consistent_ flag of the
  interpreter.
  """
  @spec set_inputs(reference, list(integer())) :: :ok | nif_error()
  def set_inputs(self, inputs) when is_reference(self) and is_list(inputs) do
    :tflite_beam_interpreter.set_inputs(self, inputs)
  end

  @doc """
  Provide a list of tensor indexes that are outputs to the model.
  Each index is bound check and this modifies the consistent_ flag of the
  interpreter.
  """
  @spec set_outputs(reference, list(integer())) :: :ok | nif_error()
  def set_outputs(self, outputs) when is_reference(self) and is_list(outputs) do
    :tflite_beam_interpreter.set_outputs(self, outputs)
  end

  @doc """
  Provide a list of tensor indexes that are variable tensors.
  Each index is bound check and this modifies the consistent_ flag of the
  interpreter.
  """
  @spec set_variables(reference, list(integer())) :: :ok | nif_error()
  def set_variables(self, variables) when is_reference(self) and is_list(variables) do
    :tflite_beam_interpreter.set_variables(self, variables)
  end

  @doc """
  Get the list of input tensors.

  return a list of input tensor id
  """
  @spec inputs(reference()) :: {:ok, [non_neg_integer()]} | nif_error()
  def inputs(self) when is_reference(self) do
    :tflite_beam_interpreter.inputs(self)
  end

  deferror(inputs(self))

  @doc """
  Get the name of the input tensor

  Note that the index here means the index in the result list of `inputs/1`. For example,
  if `inputs/1` returns `[42, 314]`, then `0` should be passed here to get the name of
  tensor `42`
  """
  @spec get_input_name(reference(), non_neg_integer()) :: {:ok, String.t()} | nif_error()
  def get_input_name(self, index) when is_reference(self) and is_integer(index) and index >= 0 do
    :tflite_beam_interpreter.get_input_name(self, index)
  end

  deferror(get_input_name(self, index))

  @doc """
  Get the list of output tensors.

  return a list of output tensor id
  """
  @spec outputs(reference()) :: {:ok, [non_neg_integer()]} | nif_error()
  def outputs(self) when is_reference(self) do
    :tflite_beam_interpreter.outputs(self)
  end

  deferror(outputs(self))

  @doc """
  Get the list of variable tensors.
  """
  @spec variables(reference()) :: {:ok, [non_neg_integer()]} | nif_error()
  def variables(self) when is_reference(self) do
    :tflite_beam_interpreter.variables(self)
  end

  @doc """
  Get the name of the output tensor

  Note that the index here means the index in the result list of `outputs/1`. For example,
  if `outputs/1` returns `[42, 314]`, then `0` should be passed here to get the name of
  tensor `42`
  """
  @spec get_output_name(reference(), non_neg_integer()) :: {:ok, String.t()} | nif_error()
  def get_output_name(self, index)
      when is_reference(self) and is_integer(index) and index >= 0 do
    :tflite_beam_interpreter.get_output_name(self, index)
  end

  deferror(get_output_name(self, index))

  @doc """
  Return the number of tensors in the model.
  """
  @spec tensors_size(reference()) :: non_neg_integer() | nif_error()
  def tensors_size(self) when is_reference(self) do
    :tflite_beam_interpreter.tensors_size(self)
  end

  @doc """
  Return the number of ops in the model.
  """
  @spec nodes_size(reference()) :: non_neg_integer() | nif_error()
  def nodes_size(self) when is_reference(self) do
    :tflite_beam_interpreter.nodes_size(self)
  end

  @doc """
  Return the execution plan of the model.

  Experimental interface, subject to change.
  """
  @spec execution_plan(reference()) :: [non_neg_integer()] | nif_error()
  def execution_plan(self) when is_reference(self) do
    :tflite_beam_interpreter.execution_plan(self)
  end

  @doc """
  Get any tensor in the graph by its id

  Note that the `tensor_index` here means the id of a tensor. For example,
  if `inputs/1` returns `[42, 314]`, then `42` should be passed here to get tensor `42`.
  """
  @spec tensor(reference(), non_neg_integer()) :: %TFLiteTensor{} | nif_error()
  def tensor(self, tensor_index)
      when is_reference(self) and is_integer(tensor_index) and tensor_index >= 0 do
    case :tflite_beam_interpreter.tensor(self, tensor_index) do
      {:tflite_beam_tensor, name, index, shape, shape_signature, type,
       {:tflite_beam_quantization_params, scale, zero_point, quantized_dimension},
       sparsity_params, ref} ->
        %TFLiteTensor{
          name: name,
          index: index,
          shape: List.to_tuple(shape),
          shape_signature: shape_signature,
          type: type,
          quantization_params: %TFLiteQuantizationParams{
            scale: scale,
            zero_point: zero_point,
            quantized_dimension: quantized_dimension
          },
          sparsity_params: sparsity_params,
          reference: ref
        }

      {:error, error} ->
        {:error, error}
    end
  end

  @doc """
  Returns list of all keys of different method signatures defined in the
  model.

  WARNING: Experimental interface, subject to change
  """
  @spec signature_keys(reference) :: [String.t()] | nif_error()
  def signature_keys(self) when is_reference(self) do
    :tflite_beam_interpreter.signature_keys(self)
  end

  @doc """
  Fill data to the specified input tensor

  Note: although we have `typed_input_tensor` available in C++, here what we really passed
  to the NIF is `binary` data, therefore, I'm not pretend that we have type information.
  """
  @spec input_tensor(reference(), non_neg_integer(), binary()) :: :ok | nif_error()
  def input_tensor(self, index, data)
      when is_reference(self) and is_integer(index) and index >= 0 and is_binary(data) do
    :tflite_beam_interpreter.input_tensor(self, index, data)
  end

  deferror(input_tensor(self, index, data))

  @doc """
  Get the data of the output tensor

  Note that the index here means the index in the result list of `outputs/1`. For example,
  if `outputs/1` returns `[42, 314]`, then `0` should be passed here to get the name of
  tensor `42`
  """
  @spec output_tensor(reference(), non_neg_integer()) ::
          {:ok, binary()} | nif_error()
  def output_tensor(self, index) when is_reference(self) and is_integer(index) and index >= 0 do
    :tflite_beam_interpreter.output_tensor(self, index)
  end

  deferror(output_tensor(self, index))

  @doc """
  Allocate memory for tensors in the graph
  """
  @spec allocate_tensors(reference()) :: :ok | nif_error()
  def allocate_tensors(self) when is_reference(self) do
    :tflite_beam_interpreter.allocate_tensors(self)
  end

  deferror(allocate_tensors(self))

  @doc """
  Run forwarding
  """
  @spec invoke(reference()) :: :ok | nif_error()
  def invoke(self) when is_reference(self) do
    :tflite_beam_interpreter.invoke(self)
  end

  deferror(invoke(self))

  @doc """
  Set the number of threads available to the interpreter.

  NOTE: num_threads should be >= 1.

  As TfLite interpreter could internally apply a TfLite delegate by default
  (i.e. XNNPACK), the number of threads that are available to the default
  delegate *should be* set via InterpreterBuilder APIs as follows:

  ```elixir
  interpreter = Interpreter.new!()
  builder = InterpreterBuilder.new!(tflite model, op resolver)
  InterpreterBuilder.set_num_threads(builder, ...)
  assert :ok == InterpreterBuilder.build!(builder, interpreter)
  ```

  `num_threads` follows TfLite: `-1` asks the runtime to choose, `0` means the
  same as `1`, and anything below `-1` is answered with `{:error, reason}`.
  """
  @spec set_num_threads(reference(), integer()) :: :ok | nif_error()
  def set_num_threads(self, num_threads) when is_reference(self) and is_integer(num_threads) do
    :tflite_beam_interpreter.set_num_threads(self, num_threads)
  end

  deferror(set_num_threads(self, num_threads))

  @doc """
  Get SignatureDef map from the Metadata of a TfLite flatbuffer buffer.

  `self`: `TFLiteElixir.Interpreter`

    TFLite model buffer to get the signature_def.

  ##### Returns:

  `{:ok, map()}` of serving names to SignatureDefs, or `{:ok, nil}` for a model
  that carries none.
  """
  @spec get_signature_defs(reference()) :: {:ok, map() | nil} | {:error, String.t()}
  def get_signature_defs(self) do
    :tflite_beam_interpreter.get_signature_defs(self)
  end

  deferror(get_signature_defs(self))

  @doc """
  Get a runner for one of the model's signatures.

  Pass `nil` for the primary subgraph: the first signature that points at it, or a
  placeholder one when the model declares no signatures at all, so this works with
  older exports too.

  The runner keeps this interpreter alive. See `TFLiteElixir.SignatureRunner`.
  """
  @spec get_signature_runner(reference(), String.t() | nil) :: nif_resource_ok() | nif_error()
  def get_signature_runner(self, signature_key)
      when is_reference(self) and (is_binary(signature_key) or is_nil(signature_key)) do
    :tflite_beam_interpreter.get_signature_runner(self, signature_key)
  end

  deferror(get_signature_runner(self, signature_key))

  @doc """
  The inputs of the named signature, as a map of name to tensor index.

  An empty map is returned for a key the model does not declare.
  """
  @spec signature_inputs(reference(), String.t()) :: {:ok, map()} | nif_error()
  def signature_inputs(self, signature_key)
      when is_reference(self) and is_binary(signature_key) do
    :tflite_beam_interpreter.signature_inputs(self, signature_key)
  end

  deferror(signature_inputs(self, signature_key))

  @doc """
  The outputs of the named signature, as a map of name to tensor index.

  An empty map is returned for a key the model does not declare.
  """
  @spec signature_outputs(reference(), String.t()) :: {:ok, map()} | nif_error()
  def signature_outputs(self, signature_key)
      when is_reference(self) and is_binary(signature_key) do
    :tflite_beam_interpreter.signature_outputs(self, signature_key)
  end

  deferror(signature_outputs(self, signature_key))

  @doc """
  The subgraph a signature belongs to, or `-1` for a key the model does not declare.
  """
  @spec get_subgraph_index_from_signature(reference(), String.t()) ::
          {:ok, integer()} | nif_error()
  def get_subgraph_index_from_signature(self, signature_key)
      when is_reference(self) and is_binary(signature_key) do
    :tflite_beam_interpreter.get_subgraph_index_from_signature(self, signature_key)
  end

  deferror(get_subgraph_index_from_signature(self, signature_key))

  @doc """
  Change the dimensionality of a given input tensor.

  Only inputs can be resized, and `allocate_tensors/1` has to be called again
  afterwards.

  `dims` is a list, or the tuple `TFLiteElixir.TFLiteTensor.shape/1` returns.
  """
  @spec resize_input_tensor(reference(), integer(), [integer()] | tuple()) :: :ok | nif_error()
  def resize_input_tensor(self, tensor_index, dims)
      when is_reference(self) and is_integer(tensor_index) and (is_list(dims) or is_tuple(dims)) do
    :tflite_beam_interpreter.resize_input_tensor(self, tensor_index, dims)
  end

  @doc """
  Change the dimensionality of a given input tensor, keeping the rank fixed.

  Unlike `resize_input_tensor/3` this only accepts dimensions the model left unknown,
  so a tensor whose shape is fully fixed cannot be resized.

  `dims` is a list, or the tuple `TFLiteElixir.TFLiteTensor.shape/1` returns.
  """
  @spec resize_input_tensor_strict(reference(), integer(), [integer()] | tuple()) ::
          :ok | nif_error()
  def resize_input_tensor_strict(self, tensor_index, dims)
      when is_reference(self) and is_integer(tensor_index) and (is_list(dims) or is_tuple(dims)) do
    :tflite_beam_interpreter.resize_input_tensor_strict(self, tensor_index, dims)
  end

  @doc """
  Which process this interpreter belongs to, or `:undefined` if it is shared.
  """
  @spec controlling_process(reference()) :: {:ok, pid()} | :undefined | nif_error()
  def controlling_process(self) when is_reference(self) do
    :tflite_beam_interpreter.controlling_process(self)
  end

  @doc """
  Hand this interpreter to `pid`.

  Follows `:gen_tcp.controlling_process/2`: while an interpreter belongs to
  nobody any process may take it, and once it belongs to someone only that
  process may hand it on. Pass `:undefined` to give it back to nobody. A
  controlling process that dies releases it, since an interpreter has no
  equivalent of a socket being closed.

  Two processes whose calls overlap on an unclaimed interpreter get
  `{:error, "interpreter is already in use by another process"}`, and once it is
  claimed every other process gets `{:error, "interpreter belongs to another
  process"}` whether their calls overlap or not.
  """
  @spec controlling_process(reference(), pid() | :undefined) :: :ok | nif_error()
  def controlling_process(self, pid)
      when is_reference(self) and (is_pid(pid) or pid == :undefined) do
    :tflite_beam_interpreter.controlling_process(self, pid)
  end

  deferror(controlling_process(self))
  deferror(controlling_process(self, pid))

  @doc """
  Allow a running `invoke/1` to be cancelled.

  Has to be called before invoking. Without it `cancel/1` is an error.
  """
  @spec enable_cancellation(reference()) :: :ok | nif_error()
  def enable_cancellation(self) when is_reference(self) do
    :tflite_beam_interpreter.enable_cancellation(self)
  end

  @doc """
  Ask an in-flight `invoke/1` to stop.

  Does not block and is safe to call from another process, which is the point: an
  invocation occupies a dirty scheduler and cannot otherwise be interrupted. Later
  invocations are unaffected. Requires `enable_cancellation/1`.
  """
  @spec cancel(reference()) :: :ok | nif_error()
  def cancel(self) when is_reference(self) do
    :tflite_beam_interpreter.cancel(self)
  end

  @doc """
  Release memory that is only needed while invoking.

  Invoking again reallocates it, so this trades time for memory on devices short of
  the latter.
  """
  @spec release_non_persistent_memory(reference()) :: :ok | nif_error()
  def release_non_persistent_memory(self) when is_reference(self) do
    :tflite_beam_interpreter.release_non_persistent_memory(self)
  end

  @doc """
  Reset all variable tensors to zero.
  """
  @spec reset_variable_tensors(reference()) :: :ok | nif_error()
  def reset_variable_tensors(self) when is_reference(self) do
    :tflite_beam_interpreter.reset_variable_tensors(self)
  end

  @doc """
  How many subgraphs the model has.
  """
  @spec subgraphs_size(reference()) :: {:ok, non_neg_integer()} | nif_error()
  def subgraphs_size(self) when is_reference(self) do
    :tflite_beam_interpreter.subgraphs_size(self)
  end

  deferror(subgraphs_size(self))

  @doc """
  Whether float32 operations may be carried out in float16.
  """
  @spec get_allow_fp16_precision_for_fp32(reference()) :: {:ok, boolean()} | nif_error()
  def get_allow_fp16_precision_for_fp32(self) when is_reference(self) do
    :tflite_beam_interpreter.get_allow_fp16_precision_for_fp32(self)
  end

  deferror(get_allow_fp16_precision_for_fp32(self))

  @doc """
  Allow or forbid carrying out float32 operations in float16.

  Only has an effect on backends that can do it, and has to be set before the graph is
  prepared.
  """
  @spec set_allow_fp16_precision_for_fp32(reference(), boolean()) :: :ok | nif_error()
  def set_allow_fp16_precision_for_fp32(self, allow)
      when is_reference(self) and is_boolean(allow) do
    :tflite_beam_interpreter.set_allow_fp16_precision_for_fp32(self, allow)
  end

  @doc """
  Fill input data to corresponding input tensor of the interpreter,
  call `Interpreter.invoke` and return output tensor(s)

  Each input is a binary of the tensor's bytes or an `Nx.Tensor` of its type
  and shape. A model with one input takes it bare; otherwise pass them as a list
  in the order of `inputs/1`, or as a map from tensor name to data.
  """
  @spec predict(
          reference(),
          binary()
          | Nx.Tensor.t()
          | [binary() | Nx.Tensor.t()]
          | %{String.t() => binary() | Nx.Tensor.t()}
        ) :: [Nx.Tensor.t()] | nif_error()
  def predict(interpreter, input) do
    with {:ok, input_tensors} <- Interpreter.inputs(interpreter),
         {:ok, output_tensors} <- Interpreter.outputs(interpreter),
         :ok <- fill_input(interpreter, input_tensors, input),
         # The result of the invoke decides whether the output tensors mean
         # anything. Dropping it meant reading them anyway, so a refused or
         # failed run handed back the answer from the run before, as a perfectly
         # ordinary Nx tensor with no way for the caller to tell.
         :ok <- Interpreter.invoke(interpreter) do
      fetch_outputs(interpreter, output_tensors)
    else
      error -> error
    end
  end

  # Filling answers with :ok, a bare message, or {:error, message} depending on
  # which path produced it. Everything that collects those has to agree on one
  # shape or the join over them breaks on whichever it did not expect.
  defp reason_of({:error, reason}), do: to_string(reason)
  defp reason_of(reason) when is_binary(reason), do: reason
  defp reason_of(other), do: inspect(other)

  # One input is a list of one, so a bare binary or tensor given to a model with
  # several inputs gets the length mismatch by name. An Nx tensor is a map, and
  # used to fall into the clause below it, which then reported every input as
  # missing.
  defp fill_input(interpreter, input_tensors, input)
       when is_binary(input) or is_struct(input, Nx.Tensor) do
    fill_input(interpreter, input_tensors, [input])
  end

  defp fill_input(interpreter, input_tensors, input) when is_list(input) do
    if length(input_tensors) == length(input) do
      input_tensors
      |> Enum.zip(input)
      |> Enum.map(fn {index, data} -> fill_indexed(interpreter, index, data) end)
      |> collect_failures()
    else
      {:error,
       "length mismatch: there are #{length(input_tensors)} input tensors while the input list has #{length(input)} elements"}
    end
  end

  defp fill_input(interpreter, input_tensors, input) when is_map(input) do
    input_tensors
    |> Enum.map(fn index ->
      case Interpreter.tensor(interpreter, index) do
        %TFLiteTensor{name: name} = tensor ->
          case Map.fetch(input, name) do
            {:ok, data} -> fill_tensor(tensor, index, data)
            :error -> {:error, "missing input data for tensor `#{name}`, tensor index: #{index}"}
          end

        {:error, reason} ->
          {:error, reason}
      end
    end)
    |> collect_failures()
  end

  # The Erlang predict/2 has answered these three by name all along. This port
  # carried the clauses that succeed and none that refuse, so anything else
  # raised FunctionClauseError from a private function.
  defp fill_input(_interpreter, _input_tensors, input) do
    {:error,
     "input must be a binary, an Nx tensor, a list of them, or a map of tensor names to them, and this is #{inspect(input)}"}
  end

  defp fill_indexed(interpreter, index, data) do
    case Interpreter.tensor(interpreter, index) do
      %TFLiteTensor{} = tensor -> fill_tensor(tensor, index, data)
      {:error, reason} -> {:error, reason}
    end
  end

  # An Nx tensor carries a type and a shape, so both are held against the
  # tensor's before its bytes go in. A map value skipped both checks, so a tensor
  # of the wrong type but the right byte count was written and run unremarked.
  defp fill_tensor(%TFLiteTensor{} = tensor, index, %Nx.Tensor{} = input) do
    cond do
      tensor.type != Nx.type(input) ->
        {:error,
         "input data type, #{inspect(Nx.type(input))}, does not match the data type of the tensor, #{inspect(tensor.type)}, tensor index: #{index}"}

      tensor.shape != Nx.shape(input) and
          TFLiteTensor.dims(tensor) != [1 | Tuple.to_list(Nx.shape(input))] ->
        {:error,
         "input data shape, #{inspect(Nx.shape(input))}, does not match the shape type of the tensor, #{inspect(tensor.shape)}, tensor index: #{index}"}

      true ->
        TFLiteTensor.set_data(tensor, Nx.to_binary(input))
    end
  end

  defp fill_tensor(%TFLiteTensor{} = tensor, _index, input) when is_binary(input) do
    TFLiteTensor.set_data(tensor, input)
  end

  defp fill_tensor(%TFLiteTensor{}, index, input) do
    {:error,
     "input for tensor index #{index} is #{inspect(input)}, which is neither binary data nor an Nx tensor"}
  end

  defp collect_failures(results) do
    case Enum.reject(results, &(&1 == :ok)) do
      [] -> :ok
      failures -> {:error, failures |> Enum.map(&reason_of/1) |> Enum.join("; ")}
    end
  end

  # One name for both shapes left dialyzer unable to tell which a caller got
  # back, so predict/2's spec had to promise a bare tensor it never returns.
  defp fetch_outputs(interpreter, output_tensors) when is_list(output_tensors) do
    outputs = Enum.map(output_tensors, &fetch_output(interpreter, &1))

    # An output that could not be read is the whole call failing, not an item in
    # the list. A caller matching [out] would otherwise bind an error tuple and
    # carry on with it as though it were a tensor.
    case Enum.filter(outputs, &match?({:error, _}, &1)) do
      [] -> outputs
      failures -> {:error, failures |> Enum.map(&reason_of/1) |> Enum.join("; ")}
    end
  end

  defp fetch_output(interpreter, output_index) when is_integer(output_index) do
    case Interpreter.tensor(interpreter, output_index) do
      %TFLiteTensor{} = tensor ->
        TFLiteTensor.to_nx(tensor)

      error ->
        error
    end
  end
end
