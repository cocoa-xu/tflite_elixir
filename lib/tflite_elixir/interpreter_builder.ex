defmodule TFLiteElixir.InterpreterBuilder do
  @moduledoc """
  Build an interpreter capable of interpreting model.
  """
  import TFLiteElixir.Errorize
  alias TFLiteElixir.FlatBufferModel

  @type nif_resource_ok :: {:ok, reference()}
  @type nif_error :: {:error, String.t()}

  @doc """
  New InterpreterBuilder
  """
  @spec new(%FlatBufferModel{}, reference()) :: nif_resource_ok() | nif_error()
  def new(%FlatBufferModel{model: model}, resolver) when is_reference(resolver) do
    :tflite_beam_interpreter_builder.new(model, resolver)
  end

  deferror(new(model, resolver))

  @doc """
  Build the interpreter with the InterpreterBuilder.

  Note: all Interpreters should be built with the InterpreterBuilder,
  which allocates memory for the Interpreter and does various set up
  tasks so that the Interpreter can read the provided model.
  """
  @spec build(reference(), reference()) :: :ok | {:ok, :delegate_declined} | nif_error()
  def build(self, interpreter) do
    :tflite_beam_interpreter_builder.build(self, interpreter)
  end

  deferror(build(self, interpreter))

  @doc """
  Attach a delegate to the builder, with the default decline policy.

  Equivalent to `add_delegate(builder, delegate, [])`.
  """
  @spec add_delegate(reference(), reference()) :: :ok | nif_error()
  def add_delegate(self, delegate) when is_reference(self) and is_reference(delegate) do
    add_delegate(self, delegate, [])
  end

  deferror(add_delegate(self, delegate))

  @doc """
  Attach a delegate to every interpreter this builder goes on to build.

  The delegate is applied in the order delegates were added, and it has to
  outlive every interpreter built from this builder -- which is why there is no
  way to detach or delete one. Holding the reference is not required: the builder
  and each interpreter keep the delegate alive for as long as they need it.

  Attaching any delegate also suppresses the XNNPACK one that `build/2` would
  otherwise add for you.

  ##### Options
  - `:on_decline`. What to do when a delegate reports that it cannot take the
    graph, but leaves the graph runnable -- a static-shape delegate meeting a
    dynamic tensor, say. TfLite discards the whole interpreter in that case.
    - `:error` (the default) -- the decline surfaces as `{:error, reason}` from
      `build/2`.
    - `:fallback` -- `build/2` builds again without the delegates that were added
      with this policy, and answers `{:ok, :delegate_declined}`. Only a decline
      is retried; every other failure still fails.

  Note that an interpreter, and any delegate attached to it, belongs to one
  process at a time. Nothing here is serialised for you.
  """
  @spec add_delegate(reference(), reference(), Keyword.t() | map()) :: :ok | nif_error()
  def add_delegate(self, delegate, opts) when is_reference(self) and is_reference(delegate) do
    :tflite_beam_interpreter_builder.add_delegate(self, delegate, as_map(opts))
  end

  deferror(add_delegate(self, delegate, opts))

  @doc """
  Sets the number of CPU threads to use for the interpreter.
  Returns `true` on success, `{:error, reason}` on error.
  """
  @spec set_num_threads(reference(), integer()) :: :ok | nif_error()
  def set_num_threads(self, num_threads) when is_integer(num_threads) and num_threads >= 1 do
    :tflite_beam_interpreter_builder.set_num_threads(self, num_threads)
  end

  deferror(set_num_threads(self, num_threads))

  defp as_map(opts) when is_map(opts), do: opts
  defp as_map(opts) when is_list(opts), do: Map.new(opts)
end
