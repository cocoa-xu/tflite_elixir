defmodule TFLiteElixir.SignatureRunner do
  @moduledoc """
  A runner for one of a model's signatures.

  A signature names a subgraph together with its inputs and outputs, so tensors are
  addressed by name instead of by index and the order of a model's outputs no longer
  has to be worked out. Obtain one with `TFLiteElixir.Interpreter.get_signature_runner/2`.

  A runner belongs to the interpreter it came from and keeps that interpreter alive, so
  it stays usable even if nothing else refers to the interpreter any more. Like the
  interpreter it is not safe to use from more than one process at a time.
  """
  import TFLiteElixir.Errorize

  @type nif_error :: {:error, String.t()}

  @doc """
  The key this runner was obtained with.
  """
  @spec signature_key(reference()) :: {:ok, String.t()} | nif_error()
  def signature_key(self) when is_reference(self) do
    :tflite_beam_signature_runner.signature_key(self)
  end

  deferror(signature_key(self))

  @doc """
  How many inputs the signature has.
  """
  @spec input_size(reference()) :: {:ok, non_neg_integer()} | nif_error()
  def input_size(self) when is_reference(self) do
    :tflite_beam_signature_runner.input_size(self)
  end

  deferror(input_size(self))

  @doc """
  How many outputs the signature has.
  """
  @spec output_size(reference()) :: {:ok, non_neg_integer()} | nif_error()
  def output_size(self) when is_reference(self) do
    :tflite_beam_signature_runner.output_size(self)
  end

  deferror(output_size(self))

  @doc """
  The names of the signature's inputs.
  """
  @spec input_names(reference()) :: {:ok, [String.t()]} | nif_error()
  def input_names(self) when is_reference(self) do
    :tflite_beam_signature_runner.input_names(self)
  end

  deferror(input_names(self))

  @doc """
  The names of the signature's outputs.
  """
  @spec output_names(reference()) :: {:ok, [String.t()]} | nif_error()
  def output_names(self) when is_reference(self) do
    :tflite_beam_signature_runner.output_names(self)
  end

  deferror(output_names(self))

  @doc """
  Write data into the named input.

  `allocate_tensors/1` has to have been called first.
  """
  @spec input_tensor(reference(), String.t(), binary()) :: :ok | nif_error()
  def input_tensor(self, input_name, data)
      when is_reference(self) and is_binary(input_name) and is_binary(data) do
    :tflite_beam_signature_runner.input_tensor(self, input_name, data)
  end

  @doc """
  Read the named output.
  """
  @spec output_tensor(reference(), String.t()) :: {:ok, binary()} | nif_error()
  def output_tensor(self, output_name) when is_reference(self) and is_binary(output_name) do
    :tflite_beam_signature_runner.output_tensor(self, output_name)
  end

  deferror(output_tensor(self, output_name))

  @doc """
  Change the dimensions of the named input.

  `allocate_tensors/1` has to be called again afterwards.

  `dims` is a list, or the tuple `TFLiteElixir.TFLiteTensor.shape/1` returns.
  """
  @spec resize_input_tensor(reference(), String.t(), [integer()] | tuple()) :: :ok | nif_error()
  def resize_input_tensor(self, input_name, dims)
      when is_reference(self) and is_binary(input_name) and (is_list(dims) or is_tuple(dims)) do
    :tflite_beam_signature_runner.resize_input_tensor(self, input_name, dims)
  end

  @doc """
  Change the dimensions of the named input, keeping the rank fixed.

  Only dimensions the model left unknown can be changed.

  `dims` is a list, or the tuple `TFLiteElixir.TFLiteTensor.shape/1` returns.
  """
  @spec resize_input_tensor_strict(reference(), String.t(), [integer()] | tuple()) ::
          :ok | nif_error()
  def resize_input_tensor_strict(self, input_name, dims)
      when is_reference(self) and is_binary(input_name) and (is_list(dims) or is_tuple(dims)) do
    :tflite_beam_signature_runner.resize_input_tensor_strict(self, input_name, dims)
  end

  @doc """
  Allocate the tensors of the signature's subgraph.
  """
  @spec allocate_tensors(reference()) :: :ok | nif_error()
  def allocate_tensors(self) when is_reference(self) do
    :tflite_beam_signature_runner.allocate_tensors(self)
  end

  @doc """
  Run the signature.
  """
  @spec invoke(reference()) :: :ok | nif_error()
  def invoke(self) when is_reference(self) do
    :tflite_beam_signature_runner.invoke(self)
  end

  @doc """
  Cancel an in-flight invocation.
  """
  @spec cancel(reference()) :: :ok | nif_error()
  def cancel(self) when is_reference(self) do
    :tflite_beam_signature_runner.cancel(self)
  end

  @doc """
  Feed the signature its inputs, run it and read every output back.

  Inputs and outputs are maps keyed by the names the signature declares, which is what
  makes a signature worth using: neither side depends on the order the model happens to
  list its tensors in.
  """
  @spec predict(reference(), %{String.t() => binary()}) ::
          {:ok, %{String.t() => binary()}} | nif_error()
  def predict(self, inputs) when is_reference(self) and is_map(inputs) do
    with :ok <- allocate_tensors(self),
         :ok <- write_inputs(self, inputs),
         :ok <- invoke(self),
         {:ok, names} <- output_names(self) do
      read_outputs(self, names)
    end
  end

  deferror(predict(self, inputs))

  defp write_inputs(self, inputs) do
    Enum.reduce_while(inputs, :ok, fn {name, data}, :ok ->
      case input_tensor(self, name, data) do
        :ok -> {:cont, :ok}
        error -> {:halt, error}
      end
    end)
  end

  defp read_outputs(self, names) do
    Enum.reduce_while(names, {:ok, %{}}, fn name, {:ok, acc} ->
      case output_tensor(self, name) do
        {:ok, data} -> {:cont, {:ok, Map.put(acc, name, data)}}
        error -> {:halt, error}
      end
    end)
  end
end
