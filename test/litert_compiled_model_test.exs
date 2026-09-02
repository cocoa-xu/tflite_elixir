defmodule TFLiteElixir.LiteRT.CompiledModel.Test do
  use ExUnit.Case

  # The LiteRT API is off in the default build, so these are excluded unless it
  # is there; see test_helper.exs. Skipping says which, rather than leaving a
  # wall of failures that all mean "not compiled in".
  @moduletag :litert

  alias TFLiteElixir.LiteRT.CompiledModel
  alias TFLiteElixir.LiteRT.CompiledModel.Isolated
  alias TFLiteElixir.LiteRT.CompiledModel.Server

  @model "mobilenet_v2_1.0_224_inat_bird_quant.tflite"

  setup_all do
    {:ok, env} = CompiledModel.environment()
    {:ok, env: env, path: Path.join([__DIR__, "test_data", @model])}
  end

  # nodedown does not arrive the instant the node is halted, so the refusal is
  # waited for rather than assumed
  defp wait_for_refusal(_model, _inputs, 0), do: flunk("the model never noticed its node died")

  defp wait_for_refusal(model, inputs, tries) do
    case Isolated.run(model, inputs) do
      {:error, "the isolated model's node went down"} ->
        :ok

      _ ->
        Process.sleep(100)
        wait_for_refusal(model, inputs, tries - 1)
    end
  end

  defp inputs_for(model) do
    {:ok, {ins, _}} = CompiledModel.io_sizes(model)
    Enum.map(ins, &:binary.copy(<<0>>, &1))
  end

  describe "direct" do
    test "compiles, runs and reports its own shape", %{env: env, path: path} do
      assert {:ok, [_ | _]} = CompiledModel.signatures(env, path)

      model = CompiledModel.new!(env, path, accelerators: [:cpu])
      assert {:ok, {[150_528], [965]}} = CompiledModel.io_sizes(model)

      assert {:ok, [out]} = CompiledModel.run(model, inputs_for(model))
      assert byte_size(out) == 965

      assert is_boolean(CompiledModel.fully_accelerated?(model))
    end

    test "refuses an input of the wrong size rather than reading past it", %{
      env: env,
      path: path
    } do
      model = CompiledModel.new!(env, path, accelerators: [:cpu])
      assert {:error, message} = CompiledModel.run(model, [<<0, 1, 2>>])
      assert is_binary(message)
    end

    test "a keyword list and a map mean the same thing", %{env: env, path: path} do
      assert {:ok, _} = CompiledModel.new(env, path, accelerators: [:cpu])
      assert {:ok, _} = CompiledModel.new(env, path, %{accelerators: [:cpu]})
    end

    test "a bang variant raises the message rather than returning it", %{env: env} do
      assert_raise RuntimeError, fn -> CompiledModel.new!(env, "/no/such/model.tflite") end
    end

    test "profiling is off unless asked for", %{env: env, path: path} do
      plain = CompiledModel.new!(env, path, accelerators: [:cpu])
      {:ok, _} = CompiledModel.run(plain, inputs_for(plain))
      assert {:ok, []} = CompiledModel.profile(plain)
      assert {:ok, 0} = CompiledModel.pending_events(plain)
      assert {:error, _} = CompiledModel.reset_profile(plain)
    end

    test "a profiled model names its slowest operators", %{env: env, path: path} do
      model = CompiledModel.new!(env, path, accelerators: [:cpu], profile: true)
      {:ok, _} = CompiledModel.run(model, inputs_for(model))

      {:ok, pending} = CompiledModel.pending_events(model)
      assert pending > 0

      {:ok, events} = CompiledModel.profile(model)
      assert length(events) > 0
      # named, not LiteRT's raw enumeration numbers
      assert Enum.all?(events, &(is_atom(&1.type) and is_atom(&1.source)))

      {:ok, [slowest | _] = summary} = CompiledModel.summarise_profile(model)
      assert %{tag: tag, kind: kind, count: count, us: _} = slowest
      assert is_binary(tag) and count > 0
      assert kind in [:operator, :delegate_operator, :delegate_profiled]

      totals = Enum.map(summary, & &1.us)
      assert totals == Enum.sort(totals, :desc)

      :ok = CompiledModel.reset_profile(model)
      assert {:ok, 0} = CompiledModel.pending_events(model)
      # recording has to survive the reset, or the model silently stops profiling
      {:ok, _} = CompiledModel.run(model, inputs_for(model))
      {:ok, after_reset} = CompiledModel.pending_events(model)
      assert after_reset > 0
    end

    test "a claimed model refuses another process", %{env: env, path: path} do
      model = CompiledModel.new!(env, path, accelerators: [:cpu])
      assert :undefined == CompiledModel.controlling_process(model)
      # the inputs are worked out before the claim, because after it even
      # io_sizes/1 belongs to the owner
      inputs = inputs_for(model)

      assert :ok == CompiledModel.controlling_process(model, self())
      assert {:ok, pid} = CompiledModel.controlling_process(model)
      assert pid == self()

      task = Task.async(fn -> CompiledModel.run(model, inputs) end)
      assert {:error, message} = Task.await(task)
      assert message =~ "another process"

      # and the owner is still free to use it
      assert {:ok, [_]} = CompiledModel.run(model, inputs)
    end
  end

  describe "server" do
    test "shares one model between processes", %{env: env, path: path} do
      {:ok, server} = Server.start_link(env, path, accelerators: [:cpu])
      {:ok, {ins, _}} = Server.io_sizes(server)
      inputs = Enum.map(ins, &:binary.copy(<<0>>, &1))

      results =
        1..4
        |> Enum.map(fn _ -> Task.async(fn -> Server.run(server, inputs) end) end)
        |> Enum.map(&Task.await(&1, 30_000))

      assert Enum.all?(results, &match?({:ok, [_]}, &1))
      Server.stop(server)
    end

    test "with/2 hands out the model and takes it back", %{env: env, path: path} do
      {:ok, server} = Server.start_link(env, path, accelerators: [:cpu])
      assert {:ok, {_, _}} = Server.with(server, &CompiledModel.io_sizes/1)
      Server.stop(server)
    end

    # Documented as answering false rather than an error, and a server that is
    # gone answered with an exit instead.
    test "fully_accelerated?/1 answers false for a server that is gone" do
      gone = spawn(fn -> :ok end)
      ref = Process.monitor(gone)
      assert_receive {:DOWN, ^ref, :process, ^gone, _}
      assert false == Server.fully_accelerated?(gone)
    end

    test "a raising callback costs the call, not the model", %{env: env, path: path} do
      {:ok, server} = Server.start_link(env, path, accelerators: [:cpu])
      assert {:error, _} = Server.with(server, fn _ -> raise "boom" end)
      # the server is still there and still works
      assert {:ok, {_, _}} = Server.io_sizes(server)
      Server.stop(server)
    end

    test "the profile belongs to the model, not to a call", %{env: env, path: path} do
      {:ok, server} = Server.start_link(env, path, accelerators: [:cpu], profile: true)
      {:ok, {ins, _}} = Server.io_sizes(server)
      inputs = Enum.map(ins, &:binary.copy(<<0>>, &1))

      {:ok, _} = Server.run(server, inputs)
      {:ok, _} = Server.run(server, inputs)
      {:ok, summary} = Server.summarise_profile(server)
      assert length(summary) > 0
      # two runs, so an operator that runs once per inference is counted twice
      assert Enum.all?(summary, &(&1.count > 0))
      Server.stop(server)
    end
  end

  # Every one of these is a one line forward, and a one line forward is exactly
  # what a typo survives: a wrong atom in the dispatch reaches the far side and
  # comes back as an error nobody looks at. Calling each once is the whole test.
  describe "forwarding" do
    test "the server forwards every call it claims to", %{env: env, path: path} do
      {:ok, server} = Server.start(env, path, accelerators: [:cpu], profile: true)
      {:ok, {ins, _}} = Server.io_sizes(server)
      inputs = Enum.map(ins, &:binary.copy(<<0>>, &1))

      assert {:ok, _} = Server.run(server, inputs)
      assert {:ok, bool} = Server.fully_accelerated(server)
      assert is_boolean(bool)
      assert is_boolean(Server.fully_accelerated?(server))
      assert {:ok, {out, metrics}} = Server.run_with_metrics(server, inputs)
      assert is_list(out) and is_list(metrics)
      assert {:ok, events} = Server.profile(server)
      assert is_list(events)
      assert {:ok, few} = Server.profile(server, 2)
      assert length(few) <= 2
      assert {:ok, n} = Server.pending_events(server)
      assert is_integer(n)
      assert {:ok, _} = Server.summarise_profile(server)
      assert :ok == Server.reset_profile(server)
      assert {:ok, 0} == Server.pending_events(server)

      Server.stop(server)
    end

    @tag timeout: 120_000
    test "the isolated layer forwards every call it claims to", %{path: path} do
      Process.flag(:trap_exit, true)

      case Isolated.start(model_path: path, accelerators: [:cpu], profile: true) do
        {:error, reason} ->
          IO.puts("skipping isolated forwarding: #{inspect(reason)}")
          assert true

        {:ok, model} ->
          {:ok, {ins, _}} = Isolated.io_sizes(model)
          inputs = Enum.map(ins, &:binary.copy(<<0>>, &1))

          assert {:ok, _} = Isolated.run(model, inputs)
          assert {:ok, bool} = Isolated.fully_accelerated(model)
          assert is_boolean(bool)
          assert is_boolean(Isolated.fully_accelerated?(model))
          assert {:ok, {out, metrics}} = Isolated.run_with_metrics(model, inputs)
          assert is_list(out) and is_list(metrics)
          assert {:ok, events} = Isolated.profile(model)
          assert is_list(events)
          assert {:ok, few} = Isolated.profile(model, 2)
          assert length(few) <= 2
          assert {:ok, n} = Isolated.pending_events(model)
          assert is_integer(n)
          assert {:ok, _} = Isolated.summarise_profile(model)
          assert :ok == Isolated.reset_profile(model)
          # with/2 sends the callback to the owning node and applies it there,
          # which needs the module it belongs to to exist over there. A capture
          # of a compiled function does; a fun written inline in this test does
          # not, because the compiler kept its module in memory and there is no
          # file to send. Both are checked, because the second one used to come
          # back as a bare undef and now says which module and why.
          assert {:ok, {^ins, _}} = Isolated.with(model, &CompiledModel.io_sizes/1)

          # A fun written inline here belongs to a module the compiler kept in
          # memory, so the peer has nothing to load. The refusal names the module
          # and says what does cross, rather than passing on a bare undef.
          assert {:error, why} = Isolated.with(model, fn m -> CompiledModel.io_sizes(m) end)
          assert why =~ "no compiled file to send"

          Isolated.stop(model)
      end
    end

    test "the direct layer answers about itself", %{env: env, path: path} do
      assert TFLiteElixir.LiteRT.CompiledModel.available?()
      assert {:ok, support} = CompiledModel.platform_support()
      assert is_map(support)
      assert Enum.all?(Map.values(support), &is_boolean/1)

      model = CompiledModel.new!(env, path, accelerators: [:cpu])
      {:ok, {ins, _}} = CompiledModel.io_sizes(model)
      inputs = Enum.map(ins, &:binary.copy(<<0>>, &1))
      assert {:ok, bool} = CompiledModel.fully_accelerated(model)
      assert is_boolean(bool)
      assert {:ok, {out, metrics}} = CompiledModel.run_with_metrics(model, inputs)
      assert is_list(out) and is_list(metrics)
    end
  end

  describe "isolated" do
    @tag timeout: 120_000
    test "runs on its own node and survives that node dying", %{path: path} do
      # Trapping exits is what makes the link observable rather than fatal. A
      # machine that cannot start distribution makes init return {:stop, reason},
      # and start_link then both returns the error and sends an exit signal;
      # without this the case dies on the signal before it can decide to skip,
      # which is how a CI runner with no distribution reported a failure.
      Process.flag(:trap_exit, true)

      case Isolated.start_link(model_path: path, accelerators: [:cpu]) do
        {:error, reason} ->
          # not a defect in what is being tested: no distribution, no isolation
          IO.puts("skipping isolated case: #{inspect(reason)}")
          assert true

        {:ok, model} ->
          run_isolated_case(model)
      end
    end

    defp run_isolated_case(model) do
      {:ok, node} = Isolated.node_of(model)
      assert node != Node.self()

      {:ok, {ins, _}} = Isolated.io_sizes(model)
      inputs = Enum.map(ins, &:binary.copy(<<0>>, &1))
      assert {:ok, [out]} = Isolated.run(model, inputs)
      assert byte_size(out) == 965

      # The point of the whole module is that the node dying is survivable, so
      # kill it and require an answer rather than a hang or a crash here. A
      # test that only starts a node and runs on it would pass without ever
      # exercising the thing it is named after.
      # an MFA, not a closure: the peer does not have this module's code, and
      # a cast because the node dies before it could answer a call
      :erpc.cast(node, :erlang, :halt, [0])
      wait_for_refusal(model, inputs, 100)

      # linked, so the caller being alive is only true if the isolating process
      # did not exit; both are the point of the module
      assert Process.alive?(self())
      assert Process.alive?(model)
      assert {:error, "the isolated model's node went down"} = Isolated.run(model, inputs)
      # and it still knows which node it was, which is what an error report wants
      assert {:ok, ^node} = Isolated.node_of(model)

      Isolated.stop(model)
    end
  end
end
