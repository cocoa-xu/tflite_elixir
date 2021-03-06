defmodule TFLiteElixir.FlatBufferModel do
  import TFLiteElixir.Errorize

  @type nif_error :: {:error, String.t()}

  @behaviour Access
  defstruct [:model]
  alias __MODULE__, as: T

  @doc """
  Build model from a given tflite file

  Note that if the tensorflow-lite library was compiled with `TFLITE_MCU`,
  then this function will always have return type `nif_error()`
  """
  @spec buildFromFile(String.t()) :: %T{} | nif_error()
  def buildFromFile(filename) when is_binary(filename) do
    with {:ok, model} <- TFLiteElixir.Nif.flatBufferModel_buildFromFile(filename) do
      %T{model: model}
    else
      error -> error
    end
  end

  deferror(buildFromFile(filename))

  @doc """
  Build model from caller owned memory buffer

  Note that `buffer` will NOT be copied. Caller has the ensure that
  the buffer lives longer than the returned `reference` of `TFLite.FlatBufferModel`

  Discussion:

    We can copy the data in the NIF, but `FlatBufferModel::BuildFromBuffer` always
    assumes that the buffer is owner by the caller, (in this case, the binding code)

    However, we would have no way to release the copied memory because we couldn't
    identify if the `allocation_` borrows or owns that memory.
  """
  @spec buildFromBuffer(binary()) :: %T{} | nif_error()
  def buildFromBuffer(buffer) when is_binary(buffer) do
    with {:ok, model} <- TFLiteElixir.Nif.flatBufferModel_buildFromBuffer(buffer) do
      %T{model: model}
    else
      error -> error
    end
  end

  deferror(buildFromBuffer(buffer))

  @spec initialized(%T{}) :: bool() | nif_error()
  def initialized(%T{model: self}) when is_reference(self) do
    TFLiteElixir.Nif.flatBufferModel_initialized(self)
  end

  deferror(initialized(self))

  @doc """
  Returns the minimum runtime version from the flatbuffer. This runtime
  version encodes the minimum required interpreter version to run the
  flatbuffer model. If the minimum version can't be determined, an empty
  string will be returned.

  Note that the returned minimum version is a lower-bound but not a strict
  lower-bound; ops in the graph may not have an associated runtime version,
  in which case the actual required runtime might be greater than the
  reported minimum.
  """
  @spec getMinimumRuntime(%T{}) :: String.t() | nif_error()
  def getMinimumRuntime(%T{model: self}) when is_reference(self) do
    TFLiteElixir.Nif.flatBufferModel_getMinimumRuntime(self)
  end

  deferror(getMinimumRuntime(self))

  @doc """
  Return model metadata as a mapping of name & buffer strings.

  See Metadata table in TFLite schema.
  """
  @spec readAllMetadata(%T{}) :: %{String.t() => String.t()} | nif_error()
  def readAllMetadata(%T{model: self}) when is_reference(self) do
    TFLiteElixir.Nif.flatBufferModel_readAllMetadata(self)
  end

  deferror(readAllMetadata(self))

  @doc false
  @impl true
  def fetch(self, :initialized) do
    {:ok, initialized(self)}
  end

  @impl true
  def fetch(self, :minimum_runtime) do
    {:ok, getMinimumRuntime(self)}
  end

  @impl true
  def fetch(self, :metadata) do
    {:ok, readAllMetadata(self)}
  end

  @impl true
  def get_and_update(_self, key, _func) do
    raise RuntimeError, "cannot write to readonly property: #{inspect(key)}"
  end

  @impl true
  def pop(_self, key) do
    raise RuntimeError, "cannot pop readonly property: #{inspect(key)}"
  end

  defimpl Inspect, for: T do
    import Inspect.Algebra

    def inspect(self, opts) do
      concat([
        "#FlatBufferModel<",
        to_doc(
          %{
            "initialized" => T.initialized(self),
            "metadata" => T.readAllMetadata(self),
            "minimum_runtime" => T.getMinimumRuntime(self)
          },
          opts
        ),
        ">"
      ])
    end
  end
end
