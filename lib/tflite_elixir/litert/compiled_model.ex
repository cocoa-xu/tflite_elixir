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

  `fully_accelerated?/1` includes CPU delegates such as XNNPACK. Use
  `non_cpu_fully_accelerated?/1` to ask whether GPU or NPU executes every operation.

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

  alias TFLiteElixir.LiteRT.Compatibility

  @typedoc "The accelerators allowed to execute the model; list order does not set priority."
  @type accelerator :: :cpu | :gpu | :npu

  @typedoc """
  Compute precision. `:default` leaves it to the accelerator; request `:fp32`
  explicitly when accuracy matters. `:fp16` trades accuracy for speed.
  """
  @type precision :: :default | :fp16 | :fp32 | :fp16_with_fp32_accum

  @type opts ::
          [
            accelerators: [accelerator()],
            precision: precision(),
            profile: boolean(),
            signature: non_neg_integer() | String.t(),
            max_model_bytes: non_neg_integer(),
            backend_options: backend_options()
          ]
          | map()

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

  @type backend_options :: %{
          optional(:cpu | :gpu | :arm | :intel_openvino | :google_tensor | :mediatek | :qualcomm) =>
            map()
        }
  @type detail_level :: 0..2_147_483_647

  @type metrics_options ::
          %{
            optional(:detail_level) => detail_level(),
            optional(:host_timing) => boolean(),
            optional(:extra_metrics) => [{binary(), integer() | float() | boolean() | binary()}]
          }
          | Keyword.t()

  @erl :tflite_beam_litert_compiled_model

  @doc """
  A LiteRT environment, which the accelerator plugins are loaded into.

  One can serve any number of models. Each compiled model retains its environment.
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
  - `:accelerators`. Which accelerators may run the model, e.g. `[:cpu, :gpu]`.
    List order does not set priority.
    `non_cpu_fully_accelerated?/1` reports whether GPU or NPU executes the
    whole graph. Defaults to `[:cpu]`.
  - `:precision`. `:default`, `:fp16`, `:fp32` or `:fp16_with_fp32_accum`.
  - `:profile`. Record per-operator timings, readable with `profile/1` and
    `summarise_profile/1`. Off by default because it is not free.
  - `:signature`. Which signature to compile for, by index or by name.
  - `:max_model_bytes`. Refuse a model file larger than this.
  - `:backend_options`. A map of backend names to typed settings. Requires an
    upstream version with `backend_options/1`; older versions return an error
    rather than ignoring the requested settings.
  """
  @spec new(reference(), String.t(), opts()) :: {:ok, reference()} | {:error, String.t()}
  def new(env, model_path, opts) when is_list(opts) do
    new(env, model_path, Map.new(opts))
  end

  def new(env, model_path, opts) when is_map(opts) do
    Compatibility.with_backend_options(opts, &@erl.new(env, model_path, &1))
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
  Whether every operation was delegated, including to CPU delegates such as
  XNNPACK. Use `non_cpu_fully_accelerated/1` to distinguish GPU or NPU execution
  from CPU execution.
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

  @doc "Whether every operation executes on GPU or NPU, with no CPU execution."
  @spec non_cpu_fully_accelerated(reference()) :: {:ok, boolean()} | {:error, String.t()}
  def non_cpu_fully_accelerated(model) do
    Compatibility.call(@erl, :non_cpu_fully_accelerated, [model])
  end

  @doc "As `non_cpu_fully_accelerated/1`, returning false if the query fails."
  @spec non_cpu_fully_accelerated?(reference()) :: boolean()
  def non_cpu_fully_accelerated?(model) do
    case non_cpu_fully_accelerated(model) do
      {:ok, fully} -> fully
      _ -> false
    end
  end

  deferror(non_cpu_fully_accelerated(model))

  @doc """
  Validate typed backend settings and return their `{identifier, TOML}` payloads.

  Takes a map such as `%{cpu: %{num_threads: 4}}`. Validation does not require
  vendor hardware. A setting takes effect only when the loaded backend supports
  it. This requires a newer upstream API; older dependencies return an error.
  """
  @spec backend_options(backend_options()) :: {:ok, [{binary(), binary()}]} | {:error, String.t()}
  def backend_options(options), do: Compatibility.call(@erl, :backend_options, [options])

  deferror(backend_options(options))

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

  The third argument accepts the existing integer detail level or a keyword
  list/map with `:detail_level`, `:host_timing` and `:extra_metrics`.
  `host_timing: true` adds `"tflite_beam_run_us"`, measured across input transfer,
  synchronous inference and output transfer, excluding queue waiting and metrics
  collection. `extra_metrics` contains `{binary_name, value}` tuples; values may
  be signed 64-bit integers, floats, booleans or binaries without NUL bytes.
  These are caller-provided measurements, not callbacks into native code.

  Metrics options require a newer upstream API; older dependencies return an
  error. Integer detail levels retain their existing behavior.

  Usually the counters come back empty. Filling them in is the accelerator's
  job, through two entries of its definition that are allowed to be null, so an
  empty list means nobody offered anything rather than that something went
  wrong. Use `profile/1` for timings; this is for counters a backend chooses to
  expose.
  """
  @spec run_with_metrics(reference(), [binary()], detail_level() | metrics_options()) ::
          {:ok, {[binary()], [{binary(), metric_value()}]}} | {:error, String.t()}
  def run_with_metrics(model, inputs, detail_level) do
    Compatibility.metrics(@erl, [model, inputs, detail_level])
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
