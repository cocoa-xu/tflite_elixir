defmodule TFLiteElixir do
  @moduledoc """
  This module contains some helper functions from the `tflite`
  namespace in TensorFlow Lite's codebase.
  """

  alias TFLiteElixir.TFLiteTensor

  @doc """
  The version of the TfLite sources this was built from, for example `"2.2.0"`.

  Since the runtime moved to LiteRT's `tflite` subtree this is LiteRT's version,
  not TensorFlow's. The two are separate version lines and the numbers are not
  comparable: LiteRT's 2.2.0 is newer than TensorFlow's 2.21.0, not older. For
  the TensorFlow release the build borrowed from, see `tensorflow_version/0`.

  **This is the number a delegate plugin has to match.** A plugin loaded through
  `TFLiteElixir.Delegate.external/1` must have been built from the same release;
  upstream offers no binary stable delegate interface, so a mismatch is undefined
  behaviour rather than an error.
  """
  @spec tflite_version() :: String.t()
  def tflite_version, do: :tflite_beam.tflite_version()

  @doc """
  The TensorFlow release this build pulled in, for example `"2.21.0-rc0"`.

  TensorFlow is not where the runtime comes from any more: LiteRT reaches into it
  for `compiler/mlir/lite`, TSL and XLA. Worth having when something reads wrong,
  not for matching a plugin against.
  """
  @spec tensorflow_version() :: String.t()
  def tensorflow_version, do: :tflite_beam.tensorflow_version()

  @doc """
  Which source tree the loaded shared object was built from. Answers `:litert`.

  There is no other answer: the C++ behind it names a type only LiteRT's schema
  defines, so a binary built from anything else does not compile, and a release
  from before the move has no such function at all. Worth asking in a test rather
  than trusting the build, because a stale precompiled artifact looks exactly
  like a fresh one from the outside.
  """
  @spec source_tree() :: :litert
  def source_tree, do: :tflite_beam.source_tree()

  @doc """
  Prints a dump of what tensors and what nodes are in the interpreter.

  Note that this function directly prints to stdout
  """
  @spec print_interpreter_state(reference()) :: nil
  def print_interpreter_state(interpreter) do
    :tflite_beam_nif.tflite_print_interpreter_state(interpreter)
    nil
  end

  @doc """
  Resets a variable tensor to the default value.
  """
  @spec reset_variable_tensor(%TFLiteTensor{} | reference()) :: :ok | {:error, String.t()}
  def reset_variable_tensor(%TFLiteTensor{reference: reference}) do
    # was: passing the struct itself, which the NIF resolves as a resource and
    # never can, so this returned {:error, "cannot access ..."} every single time
    reset_variable_tensor(reference)
  end

  def reset_variable_tensor(reference) when is_reference(reference) do
    :tflite_beam_nif.tflite_reset_variable_tensor(reference)
  end
end
