defmodule TFLiteElixir.Delegate do
  @moduledoc """
  TfLite delegates: graph accelerators an interpreter builder can be given.
  """
  import TFLiteElixir.Errorize, only: [deferror: 1]

  @type nif_resource_ok :: {:ok, reference()}
  @type nif_error :: {:error, String.t()}

  @doc """
  Which delegate kinds this build of the library can construct.

  This answers "was it compiled in", not "is a device present" -- the two have
  different answers on the same binary. Whether a device is there is discovered
  by trying to create the delegate and getting `{:error, reason}` back.

  Note that an interpreter, and any delegate attached to it, belongs to one
  process at a time.
  """
  @spec available() :: [atom()]
  def available do
    :tflite_beam_delegate.available()
  end

  @doc """
  Create an XNNPACK delegate.

  XNNPACK is compiled into every target except armv6 and armv7l; `available/0`
  is what says so, and this returns `{:error, reason}` where it is not.

  ##### Options
  - `:num_threads`. Size of the delegate's thread pool. Zero or less means no
    thread pool at all, which is XNNPACK's own default and therefore this one.
    Note that this is not the same knob as
    `TFLiteElixir.InterpreterBuilder.set_num_threads/2`, which drives TfLite's
    CPU backend: a delegate created here carries its own pool.
  - `:flags`. A list of atoms, added to XNNPACK's defaults rather than replacing
    them -- TfLite spells turning a default off as its own flag, such as
    `:disable_subgraph_reshaping`. One of `:qs8`, `:qu8`, `:force_fp16`,
    `:dynamic_fully_connected`, `:variable_operators`,
    `:transient_indirection_buffer`, `:enable_latest_operators`,
    `:enable_subgraph_reshaping`, `:slow_consistent_arithmetic`,
    `:disable_subgraph_reshaping` or `:disable_dynamically_quantized_ops`.
  - `:weight_cache_file_path`. Where to keep XNNPACK's cache of packed weights,
    which is read if it exists and written if it does not.
  """
  @spec xnnpack() :: nif_resource_ok() | nif_error()
  def xnnpack do
    xnnpack([])
  end

  deferror(xnnpack())

  @doc """
  Create an XNNPACK delegate. See `xnnpack/0` for the options.
  """
  @spec xnnpack(Keyword.t() | map()) :: nif_resource_ok() | nif_error()
  def xnnpack(opts) do
    :tflite_beam_delegate.xnnpack(as_map(opts))
  end

  deferror(xnnpack(opts))

  @doc """
  Load a delegate from a shared library implementing TfLite's delegate plugin
  interface -- `tflite_plugin_create_delegate` and
  `tflite_plugin_destroy_delegate`.

  That covers Edge TPU, a GPU delegate built elsewhere, and any vendor delegate,
  without this library having to know anything about them.

  The path is resolved to an absolute one before loading, because the loader is
  asked for exactly the file named: a bare `libfoo.so` would otherwise be looked
  up along the system search path, which is not where anyone means.

  Options are handed to the plugin as strings, since that is the whole of the
  plugin ABI -- atoms and integers are converted, and at most 256 pairs fit.
  What the keys mean is the plugin's business.

      {:ok, delegate} =
        TFLiteElixir.Delegate.external("/opt/lib/libvendor_delegate.so",
          device: 0, precision: :fp16)

  The library is never unloaded, which is what TfLite does too.
  """
  @spec external(binary() | list()) :: nif_resource_ok() | nif_error()
  def external(library_path) do
    external(library_path, [])
  end

  deferror(external(library_path))

  @doc """
  Load a delegate from a shared library. See `external/1` for what the options
  mean.
  """
  @spec external(binary() | list(), Keyword.t() | map()) :: nif_resource_ok() | nif_error()
  def external(library_path, opts) do
    :tflite_beam_delegate.external(library_path, as_map(opts))
  end

  deferror(external(library_path, opts))

  defp as_map(opts) when is_map(opts), do: opts
  defp as_map(opts) when is_list(opts), do: Map.new(opts)
end
