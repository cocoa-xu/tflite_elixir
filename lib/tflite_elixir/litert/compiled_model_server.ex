defmodule TFLiteElixir.LiteRT.CompiledModel.Server do
  @moduledoc """
  A compiled model that lives inside a process, so several processes can share
  one model without taking turns badly.

  `TFLiteElixir.LiteRT.CompiledModel` refuses a second concurrent caller, which
  is safe but leaves the caller to arrange the turns. This does the arranging:
  calls are serialised by the process, and the model is claimed by it, so a
  reference that escaped cannot be used behind its back.

      {:ok, env} = TFLiteElixir.LiteRT.CompiledModel.environment()
      {:ok, server} = TFLiteElixir.LiteRT.CompiledModel.Server.start_link(env, path)
      {:ok, outputs} = TFLiteElixir.LiteRT.CompiledModel.Server.run(server, inputs)

  A model still runs one inference at a time, because LiteRT does; what the
  process adds is that waiting is explicit and bounded rather than a race.

  ## The queue is bounded

  A caller that submits faster than the model runs would otherwise grow the
  mailbox until the node dies. Past `:max_queue` pending calls the server
  answers `{:error, "the model's queue is full"}` instead, which is a back
  pressure signal a caller can act on. The default is 64.
  """

  import TFLiteElixir.Errorize

  alias TFLiteElixir.LiteRT.CompiledModel

  @type opts :: [
          {:max_queue, non_neg_integer()} | {atom(), term()}
        ]

  @erl :tflite_beam_litert_compiled_model_server
  @default_timeout 30_000

  @doc "Start a compiled model process linked to the caller."
  @spec start_link(reference(), String.t()) :: {:ok, pid()} | {:error, term()}
  def start_link(env, model_path), do: @erl.start_link(env, model_path)

  @doc """
  Start a compiled model process linked to the caller.

  Takes everything `TFLiteElixir.LiteRT.CompiledModel.new/3` takes, plus:

  - `:max_queue`. How many calls may be waiting before further ones are refused.
    Defaults to 64.
  """
  @spec start_link(reference(), String.t(), opts()) :: {:ok, pid()} | {:error, term()}
  def start_link(env, model_path, opts) when is_list(opts) do
    @erl.start_link(env, model_path, Map.new(opts))
  end

  def start_link(env, model_path, opts) when is_map(opts) do
    @erl.start_link(env, model_path, opts)
  end

  @doc "Start one outside a supervision tree."
  @spec start(reference(), String.t()) :: {:ok, pid()} | {:error, term()}
  def start(env, model_path), do: @erl.start(env, model_path)

  @doc "Start one outside a supervision tree, with options."
  @spec start(reference(), String.t(), opts()) :: {:ok, pid()} | {:error, term()}
  def start(env, model_path, opts) when is_list(opts),
    do: @erl.start(env, model_path, Map.new(opts))

  def start(env, model_path, opts) when is_map(opts), do: @erl.start(env, model_path, opts)

  @doc "Run the model over a list of input binaries."
  @spec run(pid(), [binary()]) :: {:ok, [binary()]} | {:error, String.t()}
  def run(server, inputs), do: @erl.run(server, inputs)

  @doc "Run the model, waiting at most `timeout`."
  @spec run(pid(), [binary()], timeout()) :: {:ok, [binary()]} | {:error, String.t()}
  def run(server, inputs, timeout), do: @erl.run(server, inputs, timeout)

  deferror(run(server, inputs))
  deferror(run(server, inputs, timeout))

  @doc """
  Run a function against the compiled model inside the owning process.

  The escape hatch for anything this module does not forward. The reference is
  only usable for the duration of the call, because the server owns the model
  and takes it back afterwards. A function that raises costs the call and not
  the model.
  """
  @spec with(pid(), (reference() -> result)) :: result | {:error, String.t()} when result: term()
  def with(server, fun), do: @erl.with(server, fun)

  @doc "As `with/2`, waiting at most `timeout`."
  @spec with(pid(), (reference() -> result), timeout()) :: result | {:error, String.t()}
        when result: term()
  def with(server, fun, timeout), do: @erl.with(server, fun, timeout)

  @doc "Whether the accelerator took the whole graph."
  @spec fully_accelerated(pid()) :: {:ok, boolean()} | {:error, String.t()}
  def fully_accelerated(server), do: @erl.fully_accelerated(server)

  @doc "As `fully_accelerated/1`, answering `false` rather than an error."
  @spec fully_accelerated?(pid()) :: boolean()
  def fully_accelerated?(server) do
    case @erl.fully_accelerated(server) do
      {:ok, fully} -> fully
      _ -> false
    end
  end

  deferror(fully_accelerated(server))

  @doc "The byte size of each input and output tensor, as `{inputs, outputs}`."
  @spec io_sizes(pid()) ::
          {:ok, {[non_neg_integer()], [non_neg_integer()]}} | {:error, String.t()}
  def io_sizes(server), do: @erl.io_sizes(server)

  deferror(io_sizes(server))

  @doc "Run and collect whatever counters the accelerator reports."
  @spec run_with_metrics(pid(), [binary()]) ::
          {:ok, {[binary()], [{binary(), CompiledModel.metric_value()}]}} | {:error, String.t()}
  def run_with_metrics(server, inputs), do: @erl.run_with_metrics(server, inputs)

  @doc "As `run_with_metrics/2`, at a given detail level."
  @spec run_with_metrics(pid(), [binary()], non_neg_integer()) ::
          {:ok, {[binary()], [{binary(), CompiledModel.metric_value()}]}} | {:error, String.t()}
  def run_with_metrics(server, inputs, detail_level) do
    @erl.run_with_metrics(server, inputs, detail_level)
  end

  @doc "As `run_with_metrics/3`, waiting at most `timeout`."
  @spec run_with_metrics(pid(), [binary()], non_neg_integer(), timeout()) ::
          {:ok, {[binary()], [{binary(), CompiledModel.metric_value()}]}} | {:error, String.t()}
  def run_with_metrics(server, inputs, detail_level, timeout) do
    @erl.run_with_metrics(server, inputs, detail_level, timeout)
  end

  deferror(run_with_metrics(server, inputs))
  deferror(run_with_metrics(server, inputs, detail_level))
  deferror(run_with_metrics(server, inputs, detail_level, timeout))

  @doc """
  Every profiling event recorded so far.

  The profile belongs to the model, not to a call, so this covers every run any
  process has made since the last reset.
  """
  @spec profile(pid()) :: {:ok, [CompiledModel.event()]} | {:error, String.t()}
  def profile(server), do: @erl.profile(server)

  @doc "The most recent `limit` profiling events, or all of them when zero."
  @spec profile(pid(), non_neg_integer()) ::
          {:ok, [CompiledModel.event()]} | {:error, String.t()}
  def profile(server, limit), do: @erl.profile(server, limit)

  deferror(profile(server))
  deferror(profile(server, limit))

  @doc "How many profiling events are waiting, without reading them."
  @spec pending_events(pid()) :: {:ok, non_neg_integer()} | {:error, String.t()}
  def pending_events(server), do: @erl.pending_events(server)

  deferror(pending_events(server))

  @doc "Per-operator totals over every run since the last reset, slowest first."
  @spec summarise_profile(pid()) ::
          {:ok, [CompiledModel.summary_entry()]} | {:error, String.t()}
  def summarise_profile(server), do: @erl.summarise_profile(server)

  deferror(summarise_profile(server))

  @doc "Forget the events recorded so far and keep recording."
  @spec reset_profile(pid()) :: :ok | {:error, String.t()}
  def reset_profile(server), do: @erl.reset_profile(server)

  deferror(reset_profile(server))

  @doc "Stop the process, and with it the compiled model."
  @spec stop(pid()) :: :ok
  def stop(server), do: @erl.stop(server)

  @doc false
  def default_timeout, do: @default_timeout
end
