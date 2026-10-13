defmodule TFLiteElixir.LiteRT.API23Test do
  use ExUnit.Case

  @moduletag :litert
  @moduletag :litert_23

  alias TFLiteElixir.LiteRT.CompiledModel
  alias TFLiteElixir.LiteRT.CompiledModel.Isolated
  alias TFLiteElixir.LiteRT.CompiledModel.Server
  alias TFLiteElixir.LiteRT.Model

  setup do
    {:ok, env} = CompiledModel.environment()
    path = Path.join([__DIR__, "test_data", "mobilenet_v2_1.0_224_inat_bird_quant.tflite"])
    input = File.read!(Path.join([__DIR__, "test_data", "parrot.bin"]))
    {:ok, env: env, path: path, input: input}
  end

  test "backend options and keyword metrics preserve CPU inference", context do
    options = %{cpu: %{num_threads: 2, kernel_mode: :reference}}
    assert [{"xnnpack", payload}] = CompiledModel.backend_options!(options)
    assert payload =~ "num_threads"
    model = CompiledModel.new!(context.env, context.path, backend_options: options)
    assert false == CompiledModel.fully_accelerated!(model)
    assert false == CompiledModel.non_cpu_fully_accelerated!(model)
    assert false == CompiledModel.non_cpu_fully_accelerated?(model)
    outputs = CompiledModel.run!(model, [context.input])

    {^outputs, metrics} =
      CompiledModel.run_with_metrics!(model, [context.input],
        host_timing: true,
        extra_metrics: [{"batch", 1}]
      )

    assert {"batch", 1} in metrics
    assert {"tflite_beam_run_us", duration} = List.keyfind(metrics, "tflite_beam_run_us", 0)
    assert is_integer(duration) and duration >= 0
  end

  test "server adapts keyword metrics and preserves errors without losing its model", context do
    {:ok, server} =
      Server.start(context.env, context.path,
        backend_options: %{cpu: %{num_threads: 2, kernel_mode: :reference}}
      )

    on_exit(fn -> if Process.alive?(server), do: Server.stop(server) end)
    assert false == Server.fully_accelerated!(server)
    assert false == Server.non_cpu_fully_accelerated!(server)
    outputs = Server.run!(server, [context.input])

    assert {^outputs, metrics} =
             Server.run_with_metrics!(server, [context.input], [host_timing: true], 30_000)

    assert List.keymember?(metrics, "tflite_beam_run_us", 0)
    assert {:error, _} = Server.run_with_metrics(server, [context.input], %{unknown: true})
    assert ^outputs = Server.run!(server, [context.input])
  end

  test "nonCPU predicate answers false for a stopped server" do
    gone = spawn(fn -> :ok end)
    monitor = Process.monitor(gone)
    assert_receive {:DOWN, ^monitor, :process, ^gone, _}
    assert false == Server.non_cpu_fully_accelerated?(gone)
  end

  @tag timeout: 120_000
  test "isolated forwarding exposes nonCPU query and keyword metrics", context do
    {:ok, model} =
      Isolated.start(
        model_path: context.path,
        backend_options: %{cpu: %{num_threads: 2, kernel_mode: :reference}}
      )

    on_exit(fn -> if Process.alive?(model), do: Isolated.stop(model) end)
    assert false == Isolated.fully_accelerated!(model)
    assert false == Isolated.non_cpu_fully_accelerated!(model)
    outputs = Isolated.run!(model, [context.input])

    assert {^outputs, metrics} =
             Isolated.run_with_metrics!(model, [context.input], [host_timing: true], 30_000)

    assert List.keymember?(metrics, "tflite_beam_run_us", 0)
    {:ok, node} = Isolated.node_of(model)
    :erpc.cast(node, :erlang, :halt, [0])
    assert {:error, _} = await_node_down(model, 100)
    assert false == Isolated.non_cpu_fully_accelerated?(model)
  end

  defp await_node_down(model, 0), do: Isolated.non_cpu_fully_accelerated(model)

  defp await_node_down(model, tries) do
    case Isolated.non_cpu_fully_accelerated(model) do
      {:ok, _} ->
        Process.sleep(20)
        await_node_down(model, tries - 1)

      result ->
        result
    end
  end

  test "experimental graph wrappers preserve sources and expose bang variants", context do
    binary = File.read!(context.path)
    source = {:buffer, binary}
    graphs = Model.inspect!(context.env, source)
    assert is_list(graphs) and graphs != []
    rewritten = Model.rewrite!(context.env, source, [])
    assert is_binary(rewritten)
    assert graphs == Model.inspect!(context.env, {:buffer, rewritten})
    assert binary == File.read!(context.path)
    assert_raise RuntimeError, fn -> Model.inspect!(context.env, {:buffer, <<>>}) end
  end
end
