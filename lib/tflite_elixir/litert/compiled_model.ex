defmodule TFLiteElixir.LiteRT.CompiledModel do
  @moduledoc """
  A model compiled through LiteRT, which is where the accelerators and the
  per-operator profile live.

  This is a different path into the same runtime as `TFLiteElixir.Interpreter`,
  not a replacement for it. What it adds is a choice of accelerator that is
  asked for by name and answered honestly, and a profiler:

      {:ok, env} = TFLiteElixir.LiteRT.CompiledModel.environment()
      {:ok, model} = TFLiteElixir.LiteRT.CompiledModel.new(env, path,
                       accelerators: [:cpu, :gpu], profile: true)
      {:ok, outputs} = TFLiteElixir.LiteRT.CompiledModel.run(model, inputs)
      {:ok, slowest} = TFLiteElixir.LiteRT.CompiledModel.summarise_profile(model)

  `fully_accelerated?/1` says whether the accelerator took the whole graph or
  only part of it, which is the difference between a speedup and a slowdown and
  is not otherwise visible.

  ## One caller at a time

  LiteRT does not promise its compiled model API is safe to use from several
  threads, and the profile buffer under it says outright that it is not. So a
  second concurrent caller is refused here rather than allowed to corrupt
  anything, and `{:error, "compiled model is in use by another caller"}` is a
  normal answer rather than a fault. `TFLiteElixir.LiteRT.CompiledModel.Server`
  is the way to share one model between processes.

  ## Availability

  The LiteRT API is a build option and is off by default, so every function here
  can answer `{:error, "tflite_beam was compiled without the LiteRT API"}` on an
  ordinary build. `platform_support/0` says what this build can reach.
  """

  import TFLiteElixir.Errorize, only: [deferror: 1]

  @typedoc "What to run on, in order of preference."
  @type accelerator :: :cpu | :gpu | :npu

  @typedoc """
  Compute precision. `:default` leaves it to the accelerator, which for Metal
  means fp32; `:fp16` trades accuracy for speed.
  """
  @type precision :: :default | :fp16 | :fp32 | :fp16_with_fp32_accum

  @type opts :: [
          accelerators: [accelerator()],
          precision: precision(),
          profile: boolean(),
          signature: non_neg_integer() | String.t(),
          max_model_bytes: non_neg_integer()
        ]

  @typedoc "A profiling event. An integer type or source is one this build has no name for."
  @type event :: %{
          tag: binary(),
          us: non_neg_integer(),
          type: atom() | integer(),
          source: atom() | integer()
        }

  @type summary_entry :: %{
          tag: binary(),
          kind: :operator | :delegate_operator | :delegate_profiled,
          count: pos_integer(),
          us: non_neg_integer()
        }

  @type metric_value :: integer() | float() | boolean() | binary() | :unsupported

  @erl :tflite_beam_litert_compiled_model

  @doc """
  A LiteRT environment, which the accelerator plugins are loaded into.

  One is enough for any number of models and it has to outlive them.
  """
  @spec environment() :: {:ok, reference()} | {:error, String.t()}
  def environment, do: @erl.environment()

  deferror(environment())

  @doc """
  An environment that looks in `runtime_library_dir` for accelerator plugins.

  Without a directory the plugins are searched for relative to nothing, which is
  the usual reason a GPU accelerator silently does not load.
  """
  @spec environment(String.t()) :: {:ok, reference()} | {:error, String.t()}
  def environment(runtime_library_dir), do: @erl.environment(runtime_library_dir)

  deferror(environment(runtime_library_dir))

  @doc """
  The names of a model file's signatures, without compiling it.

  A model with no named signature reports the one default signature LiteRT gives
  it, so the list is never empty.
  """
  @spec signatures(reference(), String.t()) :: {:ok, [binary()]} | {:error, String.t()}
  def signatures(env, model_path), do: @erl.signatures(env, model_path)

  deferror(signatures(env, model_path))

  @doc "Compile a model with the default options: CPU, no profiling."
  @spec new(reference(), String.t()) :: {:ok, reference()} | {:error, String.t()}
  def new(env, model_path), do: @erl.new(env, model_path)

  @doc """
  Compile a model.

  ##### Options
  - `:accelerators`. What to run on, in order of preference, e.g. `[:cpu, :gpu]`.
    Naming an accelerator asks for it; whether it was used is
    `fully_accelerated?/1`. Defaults to `[:cpu]`.
  - `:precision`. `:default`, `:fp16`, `:fp32` or `:fp16_with_fp32_accum`.
  - `:profile`. Record per-operator timings, readable with `profile/1` and
    `summarise_profile/1`. Off by default because it is not free.
  - `:signature`. Which signature to compile for, by index or by name.
  - `:max_model_bytes`. Refuse a model file larger than this.
  """
  @spec new(reference(), String.t(), opts()) :: {:ok, reference()} | {:error, String.t()}
  def new(env, model_path, opts) when is_list(opts) do
    @erl.new(env, model_path, Map.new(opts))
  end

  def new(env, model_path, opts) when is_map(opts) do
    @erl.new(env, model_path, opts)
  end

  deferror(new(env, model_path))
  deferror(new(env, model_path, opts))

  @doc """
  Run the model over a list of input binaries, one per input tensor.

  The sizes each input has to be are `io_sizes/1`, and a wrong size is refused
  rather than read past.
  """
  @spec run(reference(), [binary()]) :: {:ok, [binary()]} | {:error, String.t()}
  def run(model, inputs), do: @erl.run(model, inputs)

  deferror(run(model, inputs))

  @doc "The byte size of each input and output tensor, as `{inputs, outputs}`."
  @spec io_sizes(reference()) ::
          {:ok, {[non_neg_integer()], [non_neg_integer()]}} | {:error, String.t()}
  def io_sizes(model), do: @erl.io_sizes(model)

  deferror(io_sizes(model))

  @doc """
  Whether the accelerator took the whole graph.

  A partly accelerated model pays for every crossing between the accelerator and
  the CPU, and is often slower than the CPU alone, so a `false` here is worth
  acting on rather than ignoring.
  """
  @spec fully_accelerated?(reference()) :: boolean()
  def fully_accelerated?(model) do
    case @erl.fully_accelerated(model) do
      {:ok, fully} -> fully
      _ -> false
    end
  end

  @doc "As `fully_accelerated?/1`, but reports why it could not be answered."
  @spec fully_accelerated(reference()) :: {:ok, boolean()} | {:error, String.t()}
  def fully_accelerated(model), do: @erl.fully_accelerated(model)

  deferror(fully_accelerated(model))

  @doc "Every profiling event recorded so far, oldest first."
  @spec profile(reference()) :: {:ok, [event()]} | {:error, String.t()}
  def profile(model), do: @erl.profile(model)

  @doc """
  The most recent `limit` profiling events, or all of them when `limit` is zero.

  `limit` bounds the events returned, not the reading: LiteRT will not hand over
  part of a backlog, so every call copies whatever `pending_events/1` reports.
  That is about 109 MiB for a full buffer, which is nothing on a workstation and
  fatal on a board with 256 MB.
  """
  @spec profile(reference(), non_neg_integer()) :: {:ok, [event()]} | {:error, String.t()}
  def profile(model, limit), do: @erl.profile(model, limit)

  deferror(profile(model))
  deferror(profile(model, limit))

  @doc """
  How many profiling events are waiting, without reading them.

  Zero for a model compiled without `profile: true`. This is what sizes the copy
  `profile/2` has to make, so it is the number to look at before calling it on a
  memory-constrained target.
  """
  @spec pending_events(reference()) :: {:ok, non_neg_integer()} | {:error, String.t()}
  def pending_events(model), do: @erl.pending_events(model)

  deferror(pending_events(model))

  @doc """
  Per-operator totals over every run since the last reset, slowest first.

  Only operator events are folded in. The enclosing `Invoke`, tensor allocation
  and LiteRT's own buffer handling are events too, and adding them together
  would count the operators twice, so they stay in `profile/1` alone.
  """
  @spec summarise_profile(reference()) :: {:ok, [summary_entry()]} | {:error, String.t()}
  def summarise_profile(model), do: @erl.summarise_profile(model)

  deferror(summarise_profile(model))

  @doc "Forget the events recorded so far and keep recording."
  @spec reset_profile(reference()) :: :ok | {:error, String.t()}
  def reset_profile(model), do: @erl.reset_profile(model)

  deferror(reset_profile(model))

  @doc "Run the model and collect whatever counters the accelerator reports."
  @spec run_with_metrics(reference(), [binary()]) ::
          {:ok, {[binary()], [{binary(), metric_value()}]}} | {:error, String.t()}
  def run_with_metrics(model, inputs), do: @erl.run_with_metrics(model, inputs)

  @doc """
  Run with metrics collection bracketing the inference.

  Usually the counters come back empty. Filling them in is the accelerator's
  job, through two entries of its definition that are allowed to be null, so an
  empty list means nobody offered anything rather than that something went
  wrong. Use `profile/1` for timings; this is for counters a backend chooses to
  expose.
  """
  @spec run_with_metrics(reference(), [binary()], non_neg_integer()) ::
          {:ok, {[binary()], [{binary(), metric_value()}]}} | {:error, String.t()}
  def run_with_metrics(model, inputs, detail_level) do
    @erl.run_with_metrics(model, inputs, detail_level)
  end

  deferror(run_with_metrics(model, inputs))
  deferror(run_with_metrics(model, inputs, detail_level))

  @doc """
  Which process the model belongs to, or `:undefined` if it is unclaimed.

  An unclaimed model is open to every process. Claiming one is `controlling_process/2`.
  """
  @spec controlling_process(reference()) :: {:ok, pid()} | :undefined | {:error, String.t()}
  def controlling_process(model), do: @erl.controlling_process(model)

  @doc """
  Hand the model to a process, after which no other process may use it.

  The claim is dropped when that process dies, so a crash does not strand the
  model.
  """
  @spec controlling_process(reference(), pid()) :: :ok | {:error, String.t()}
  def controlling_process(model, pid), do: @erl.controlling_process(model, pid)

  deferror(controlling_process(model))
  deferror(controlling_process(model, pid))

  @doc """
  Whether this build has the LiteRT API at all.

  It is a build option and it is off by default, so on an ordinary build every
  other function here answers `{:error, "the LiteRT API was not compiled into
  this build..."}`. Asking this first is cheaper than finding out from a call
  that was meant to do something.
  """
  @spec available?() :: boolean()
  def available? do
    # Also false against a tflite_beam too old to have the LiteRT modules at
    # all, which is a different way of not having them and not a reason to
    # raise at the one call whose job is to answer this.
    Code.ensure_loaded?(@erl) and function_exported?(@erl, :available, 0) and @erl.available()
  end

  @doc """
  Which buffer kinds this platform can reach, e.g. `%{metal: true, opencl: false}`.

  Answers `{:error, reason}` on a build without the LiteRT API; `available?/0`
  is the question to ask first.
  """
  @spec platform_support() :: {:ok, %{atom() => boolean()}} | {:error, String.t()}
  def platform_support, do: @erl.platform_support()
end
