defmodule TFLiteElixir.LiteRT.CompatibilityTest do
  use ExUnit.Case, async: true

  alias TFLiteElixir.LiteRT.Compatibility
  alias TFLiteElixir.LiteRT.CompiledModel
  alias TFLiteElixir.LiteRT.CompiledModel.Isolated
  alias TFLiteElixir.LiteRT.CompiledModel.Server
  alias TFLiteElixir.LiteRT.Model

  test "missing upstream exports return an error" do
    assert {:error, message} = Compatibility.call(:tflite_beam_litert_model, :missing, [])
    assert message =~ "does not support missing/0"
    assert {:ok, :called} = Compatibility.with_backend_options(%{}, fn _ -> {:ok, :called} end)
  end

  test "the released dependency refuses newly requested capabilities without raising" do
    unless Compatibility.supported?(:tflite_beam_litert_compiled_model, :backend_options, 1) do
      env = make_ref()
      options = %{backend_options: %{cpu: %{num_threads: 2}}}
      assert {:error, _} = CompiledModel.new(env, "/unused", options)
      assert {:error, _} = Server.start(env, "/unused", options)
      assert {:error, _} = Server.start_link(env, "/unused", options)
      isolated_options = Map.put(options, :model_path, "/unused")
      assert {:error, _} = Isolated.start(isolated_options)
      assert {:error, _} = Isolated.start(isolated_options, [])
      assert {:error, _} = Isolated.start_link(isolated_options)
      assert {:error, _} = Isolated.start_link(isolated_options, [])
      assert {:error, _} = CompiledModel.backend_options(options.backend_options)
      assert_raise RuntimeError, fn -> CompiledModel.backend_options!(options.backend_options) end

      for module <- [CompiledModel, Server, Isolated] do
        assert {:error, _} = module.non_cpu_fully_accelerated(env)
        assert false == module.non_cpu_fully_accelerated?(env)
        assert {:error, _} = module.run_with_metrics(env, [], host_timing: true)
      end

      assert {:error, _} = Model.inspect(env, {:buffer, <<>>})
      assert {:error, _} = Model.rewrite(env, {:buffer, <<>>}, [])
    end
  end
end
