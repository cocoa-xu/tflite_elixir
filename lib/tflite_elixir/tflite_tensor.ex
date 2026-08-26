defmodule TFLiteElixir.TFLiteTensor do
  @moduledoc """
  A typed multi-dimensional array used in Tensorflow Lite.

  A `%TFLiteTensor{}` is a snapshot of the tensor's metadata plus a live handle
  to it. `type/1`, `dims/1`, `shape/1` and `quantization_params/1` answer from
  the snapshot; `to_binary/2`, `to_nx/2` and `set_data/2` go through the handle.
  The two can disagree after `TFLiteElixir.Interpreter.resize_input_tensor/3`,
  which moves the interpreter's tensors and retires every handle taken before
  it: the snapshot still reports the old shape while the handle reports that it
  has been retired. Fetch the tensor again with
  `TFLiteElixir.Interpreter.tensor/2` after a resize, or pass `tensor.reference`
  to ask the interpreter directly.
  """

  alias TFLiteElixir.TFLiteQuantizationParams

  @type nif_resource_ok :: {:ok, reference()}
  @type nif_error :: {:error, String.t()}
  @type tensor_type ::
          :no_type
          | {:f, 32}
          | {:s, 32}
          | {:u, 8}
          | {:s, 64}
          | :string
          | :bool
          | {:s, 16}
          | {:c, 64}
          | {:s, 8}
          | {:f, 16}
          | {:f, 64}
          | {:c, 128}
          | {:u, 64}
          | :resource
          | :variant
          | {:u, 32}

  defstruct [
    :name,
    :index,
    :shape,
    :shape_signature,
    :type,
    :quantization_params,
    :sparsity_params,
    :reference
  ]

  alias __MODULE__, as: T

  @doc """
  Get the data type

  Given a struct this answers the snapshot; pass `tensor.reference` to ask the
  interpreter, which reports a retired handle rather than a stale type.
  """
  @spec type(%T{}) :: tensor_type()
  # Interpreter.tensor/2 answers {:error, _} when the interpreter is held by
  # another process or the handle has been retired, and that answer flows
  # straight into whichever of these the caller piped it to. Without a clause
  # for it the caller lost the reason and got a FunctionClauseError raised from
  # in here instead, which names neither.
  def type({:error, reason}), do: {:error, reason}

  def type(%T{type: type}), do: type

  @spec type(reference()) :: tensor_type() | nif_error()
  def type(self) when is_reference(self) do
    :tflite_beam_tensor.type(self)
  end

  @doc """
  Get the dimensions (C++) API

  Given a struct this answers the snapshot taken when the struct was built, so
  it does not follow a later `TFLiteElixir.Interpreter.resize_input_tensor/3`.
  Sizing a buffer from a stale answer is the way this bites: the dimensions
  still multiply out to the old byte count while `set_data/2` on the same
  struct answers `{:error, _}`. Pass `tensor.reference` to ask the interpreter.
  """
  @spec dims(%T{}) :: [integer()]
  def dims({:error, reason}), do: {:error, reason}

  def dims(%T{shape: shape}), do: Tuple.to_list(shape)

  @spec dims(reference()) :: [integer()] | nif_error()
  def dims(self) do
    :tflite_beam_tensor.dims(self)
  end

  @doc """
  Get the tensor shape

  Given a struct this answers the snapshot; pass `tensor.reference` to ask the
  interpreter, which reports a retired handle rather than a stale shape.
  """
  @spec shape(%T{}) :: tuple()
  def shape({:error, reason}), do: {:error, reason}

  def shape(%T{shape: shape}), do: shape

  @spec shape(reference()) :: tuple() | nif_error()
  def shape(self) do
    :tflite_beam_tensor.shape(self)
  end

  @doc """
  Get the quantization params

  Given a struct this answers the snapshot; pass `tensor.reference` to ask the
  interpreter, which reports a retired handle rather than stale params.
  """
  @spec quantization_params(%T{} | reference()) :: %TFLiteQuantizationParams{} | nif_error()
  def quantization_params({:error, reason}), do: {:error, reason}

  def quantization_params(%T{quantization_params: quantization_params}),
    do: quantization_params

  def quantization_params(self) do
    case :tflite_beam_tensor.quantization_params(self) do
      {:tflite_beam_quantization_params, scale, zero_point, quantized_dimension} ->
        %TFLiteQuantizationParams{
          scale: scale,
          zero_point: zero_point,
          quantized_dimension: quantized_dimension
        }

      {:error, error} ->
        {:error, error}
    end
  end

  @doc """
  Set tensor data
  """
  @spec set_data(%T{} | reference(), binary() | %Nx.Tensor{}) :: :ok | nif_error()
  def set_data({:error, reason}, _data), do: {:error, reason}

  def set_data(%T{reference: reference}, data), do: set_data(reference, data)

  def set_data(self, %Nx.Tensor{} = data) when is_reference(self) do
    :tflite_beam_tensor.set_data(self, Nx.to_binary(data))
  end

  def set_data(self, data) when is_reference(self) and is_binary(data) do
    :tflite_beam_tensor.set_data(self, data)
  end

  @doc """
  Get binary data
  """
  @spec to_binary(%T{} | reference(), non_neg_integer()) :: binary() | {:error, String.t()}
  def to_binary(self, limit \\ 0)

  def to_binary({:error, reason}, _limit), do: {:error, reason}

  def to_binary(%T{reference: reference}, limit) when limit >= 0 do
    to_binary(reference, limit)
  end

  def to_binary(self, limit) when is_reference(self) and limit >= 0 do
    :tflite_beam_tensor.to_binary(self, limit)
  end

  @doc """
  Convert `TFLiteElixir.TFLiteTensor` to `Nx.Tensor`
  """
  @spec to_nx(reference() | %T{}, Keyword.t()) :: %Nx.Tensor{}
  def to_nx(self_struct, opts \\ [])

  def to_nx({:error, reason}, _opts), do: {:error, reason}

  def to_nx(self_struct, opts) when is_struct(self_struct, T) and is_list(opts) do
    with {:ok, type} <- nx_type(type(self_struct)),
         {:ok, shape} <- nx_shape(dims(self_struct)),
         binary when is_binary(binary) <- to_binary(self_struct) do
      binary
      |> to_nx_backend(type, opts[:backend])
      |> Nx.reshape(shape)
    end
  end

  def to_nx(self, opts) when is_reference(self) and is_list(opts) do
    with {:ok, type} <- nx_type(type(self)),
         {:ok, shape} <- nx_shape(dims(self)),
         binary when is_binary(binary) <- to_binary(self) do
      binary
      |> to_nx_backend(type, opts[:backend])
      |> Nx.reshape(shape)
    end
  end

  # Nx cannot represent several of the types TfLite reports, and a bool output is
  # ordinary for a segmentation mask. Handing one of those atoms to
  # Nx.from_binary raises from inside Nx about numerical types, which says
  # nothing about which tensor or why. dims/1 and type/1 were not checked for
  # {:error, _} either, so a retired handle reached Nx as a tuple and raised
  # there instead. The error-tuple clause has to come first: {:error, Reason} is
  # itself a two-element tuple and would otherwise pass for a type.
  defp nx_type({:error, reason}), do: {:error, reason}

  # No clause for the two 8 bit float formats. A tflite_beam built from LiteRT
  # reports them under the names Nx uses, {:f, 8} for E5M2 and {:f8_e4m3fn, 8}
  # for E4M3FN, so the general clause below carries them without translating.
  # Nx gained {:f, 8} in 0.9.0 and {:f8_e4m3fn, 8} in 0.11.0; mix.exs requires
  # 0.11 so that every type this can emit is one Nx can represent. Reading
  # E4M3FN bytes as {:f, 8} would answer a different number rather than fail,
  # so translating between them is never the fallback.
  defp nx_type({kind, bits} = type) when is_atom(kind) and is_integer(bits), do: {:ok, type}

  defp nx_type(other),
    do: {:error, "this tensor's type, #{inspect(other)}, has no Nx equivalent"}

  defp nx_shape({:error, reason}), do: {:error, reason}
  defp nx_shape(dims) when is_list(dims), do: {:ok, List.to_tuple(dims)}
  defp nx_shape(other), do: {:error, "cannot read this tensor's shape: #{inspect(other)}"}

  defp to_nx_backend(binary, type, backend) do
    case backend do
      nil ->
        Nx.from_binary(binary, type)

      module when is_atom(module) ->
        if Code.ensure_loaded?(module) do
          Nx.from_binary(binary, type, backend: module)
        else
          raise "Expecting keyword parameter `backend` to be a module, however, got `#{inspect(module)}`"
        end

      error ->
        raise "Expecting keyword parameter `backend` to be a module, however, got `#{inspect(error)}`"
    end
  end
end
