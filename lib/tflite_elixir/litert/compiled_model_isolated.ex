defmodule TFLiteElixir.LiteRT.CompiledModel.Isolated do
  @moduledoc """
  A compiled model on a node of its own, so that a crash inside the runtime
  costs one node instead of yours.

  Everything under this binding is native code, and native code that segfaults
  takes the whole VM with it regardless of how carefully the Elixir above it is
  written. Where that is unacceptable, this puts the model on a separate BEAM
  node and forwards calls to it:

      {:ok, model} = TFLiteElixir.LiteRT.CompiledModel.Isolated.start_link(
                       model_path: path, accelerators: [:cpu, :gpu])
      {:ok, outputs} = TFLiteElixir.LiteRT.CompiledModel.Isolated.run(model, inputs)

  The far side is an ordinary `TFLiteElixir.LiteRT.CompiledModel.Server`, so the
  calls are the same ones; only the wire is different. When the node dies the
  calls answer `{:error, "the isolated model is no longer there"}` rather than
  hanging or taking the caller down.

  ## What it costs

  Every input and every output crosses the distribution link, which for image
  sized tensors is not free, and the node takes a moment to start. Reach for it
  when a crash must not be fatal, not by default.

  Distribution is started if it is not already up, because a library should not
  insist the caller arranged that in advance.
  """

  import TFLiteElixir.Errorize

  alias TFLiteElixir.LiteRT.CompiledModel

  @type opts :: [
          {:model_path, String.t()}
          | {:runtime_library_dir, String.t()}
          | {:accelerators, [CompiledModel.accelerator()]}
          | {:precision, CompiledModel.precision()}
          | {:profile, boolean()}
          | {:signature, non_neg_integer() | String.t()}
          | {:max_model_bytes, non_neg_integer()}
          | {:max_queue, non_neg_integer()}
          | {:peer_args, [String.t()]}
        ]

  @erl :tflite_beam_litert_compiled_model_isolated

  @doc """
  Start a model on a node of its own, linked to the caller.

  `:model_path` is required. The rest are what
  `TFLiteElixir.LiteRT.CompiledModel.new/3` and
  `TFLiteElixir.LiteRT.CompiledModel.Server.start_link/3` take, plus:

  - `:peer_args`. Extra arguments for the node itself, e.g. `["+sbwt", "none"]`.
  """
  @spec start_link(opts()) :: {:ok, pid()} | {:error, term()}
  def start_link(opts) when is_list(opts), do: @erl.start_link(Map.new(opts))
  def start_link(opts) when is_map(opts), do: @erl.start_link(opts)

  @doc "As `start_link/1`, with options for the gen_server itself."
  @spec start_link(opts(), list()) :: {:ok, pid()} | {:error, term()}
  def start_link(opts, gen_opts) when is_list(opts) do
    @erl.start_link(Map.new(opts), gen_opts)
  end

  def start_link(opts, gen_opts) when is_map(opts), do: @erl.start_link(opts, gen_opts)

  @doc "Start one outside a supervision tree."
  @spec start(opts()) :: {:ok, pid()} | {:error, term()}
  def start(opts) when is_list(opts), do: @erl.start(Map.new(opts))
  def start(opts) when is_map(opts), do: @erl.start(opts)

  @doc "As `start/1`, with options for the gen_server itself."
  @spec start(opts(), list()) :: {:ok, pid()} | {:error, term()}
  def start(opts, gen_opts) when is_list(opts), do: @erl.start(Map.new(opts), gen_opts)
  def start(opts, gen_opts) when is_map(opts), do: @erl.start(opts, gen_opts)

  @doc "Run the model over a list of input binaries."
  @spec run(pid(), [binary()]) :: {:ok, [binary()]} | {:error, String.t()}
  def run(model, inputs), do: @erl.run(model, inputs)

  @doc "Run the model, waiting at most `timeout`."
  @spec run(pid(), [binary()], timeout()) :: {:ok, [binary()]} | {:error, String.t()}
  def run(model, inputs, timeout), do: @erl.run(model, inputs, timeout)

  deferror(run(model, inputs))
  deferror(run(model, inputs, timeout))

  @doc "The byte size of each input and output tensor, as `{inputs, outputs}`."
  @spec io_sizes(pid()) ::
          {:ok, {[non_neg_integer()], [non_neg_integer()]}} | {:error, String.t()}
  def io_sizes(model), do: @erl.io_sizes(model)

  deferror(io_sizes(model))

  @doc "Whether the accelerator took the whole graph."
  @spec fully_accelerated(pid()) :: {:ok, boolean()} | {:error, String.t()}
  def fully_accelerated(model), do: @erl.fully_accelerated(model)

  @doc "As `fully_accelerated/1`, answering `false` rather than an error."
  @spec fully_accelerated?(pid()) :: boolean()
  def fully_accelerated?(model) do
    case @erl.fully_accelerated(model) do
      {:ok, fully} -> fully
      _ -> false
    end
  end

  deferror(fully_accelerated(model))

  @doc "Every profiling event recorded so far."
  @spec profile(pid()) :: {:ok, [CompiledModel.event()]} | {:error, String.t()}
  def profile(model), do: @erl.profile(model)

  @doc "The most recent `limit` profiling events, or all of them when zero."
  @spec profile(pid(), non_neg_integer()) ::
          {:ok, [CompiledModel.event()]} | {:error, String.t()}
  def profile(model, limit), do: @erl.profile(model, limit)

  deferror(profile(model))
  deferror(profile(model, limit))

  @doc "How many profiling events are waiting, without reading them."
  @spec pending_events(pid()) :: {:ok, non_neg_integer()} | {:error, String.t()}
  def pending_events(model), do: @erl.pending_events(model)

  deferror(pending_events(model))

  @doc "Per-operator totals over every run since the last reset, slowest first."
  @spec summarise_profile(pid()) ::
          {:ok, [CompiledModel.summary_entry()]} | {:error, String.t()}
  def summarise_profile(model), do: @erl.summarise_profile(model)

  deferror(summarise_profile(model))

  @doc "Forget the events recorded so far and keep recording."
  @spec reset_profile(pid()) :: :ok | {:error, String.t()}
  def reset_profile(model), do: @erl.reset_profile(model)

  deferror(reset_profile(model))

  @doc "The node the model is running on."
  @spec node_of(pid()) :: {:ok, node()} | {:error, String.t()}
  def node_of(model), do: @erl.node_of(model)

  deferror(node_of(model))

  @doc "Stop the model and the node it is on."
  @spec stop(pid()) :: :ok
  def stop(model), do: @erl.stop(model)
end
