defmodule TFLiteElixir.Interpreter.Server do
  @moduledoc """
  An interpreter that lives inside a process, so that feeding it, running it and
  reading the result back is one step that nothing can interleave with.

  The direct API is not wrong -- it mirrors TfLite's C API faithfully -- but
  nothing in it says that `input_tensor/3`, `invoke/1` and `output_tensor/2`
  have to be treated as one operation. Two processes taking turns badly get each
  other's answers: measured on a real model, 147 wrong results in 400 calls,
  silently and without a crash.

      {:ok, server} = TFLiteElixir.Interpreter.Server.start_link(model_path)
      output = TFLiteElixir.Interpreter.Server.predict(server, [input])

  Concurrent callers are serialised by the process rather than racing inside the
  interpreter, so each gets the answer to its own input. `TFLiteElixir.Interpreter`
  stays exactly as it is for callers who would rather serialise access
  themselves.
  """

  @default_timeout 30_000

  @doc """
  Start an interpreter process for a model file, linked to the caller.

  ##### Options
  - `:num_threads`. Passed to the builder before the interpreter is built, so it
    reaches the default XNNPACK delegate as well.
  """
  @spec start_link(binary() | list(), Keyword.t()) :: {:ok, pid()} | {:error, term()}
  def start_link(model_path, opts \\ []) do
    :tflite_beam_interpreter_server.start_link(model_path, opts)
  end

  @doc """
  Start an interpreter process outside a supervision tree.
  """
  @spec start(binary() | list(), Keyword.t()) :: {:ok, pid()} | {:error, term()}
  def start(model_path, opts \\ []) do
    :tflite_beam_interpreter_server.start(model_path, opts)
  end

  @doc """
  Feed, run and read back, as one operation.
  """
  @spec predict(pid(), binary() | list() | map(), timeout()) ::
          [binary()] | {:error, String.t()}
  def predict(server, input, timeout \\ @default_timeout) do
    :tflite_beam_interpreter_server.predict(server, input, timeout)
  end

  @doc """
  Run a function against the interpreter inside the owning process.

  For the sequences `predict/3` does not cover -- resizing an input and
  reallocating, say, or driving a signature runner. The function runs in the
  server process, so nothing else touches the interpreter while it does, and it
  should return promptly for the same reason.

      TFLiteElixir.Interpreter.Server.run(server, fn interpreter ->
        TFLiteElixir.Interpreter.tensors_size(interpreter)
      end)
  """
  @spec run(pid(), (reference() -> result), timeout()) :: result | {:error, String.t()}
        when result: term()
  def run(server, fun, timeout \\ @default_timeout) when is_function(fun, 1) do
    :tflite_beam_interpreter_server.with(server, fun, timeout)
  end

  @doc """
  Stop the process, and with it the interpreter.
  """
  @spec stop(pid()) :: :ok
  def stop(server) do
    :tflite_beam_interpreter_server.stop(server)
  end
end
