defmodule TFLiteElixir.MixProject do
  use Mix.Project

  @app :tflite_elixir
  @version "1.0.0-rc4"
  @github_url "https://github.com/cocoa-xu/tflite_elixir"

  def project do
    [
      app: @app,
      version: @version,
      dialyzer: dialyzer(),
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      source_url: @github_url,
      description: description(),
      package: package()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Set TFLITE_BEAM_PATH to develop against a checkout of tflite_beam rather
  # than the published version. The two repositories move together often enough
  # that doing this by editing mix.exs invites committing the edit.
  defp tflite_beam_dep do
    case System.get_env("TFLITE_BEAM_PATH") do
      nil -> {:tflite_beam, "1.0.0-rc5"}
      path -> {:tflite_beam, path: path, override: true}
    end
  end

  defp deps do
    [
      tflite_beam_dep(),
      {:nx, "~> 0.11"},
      {:stb_image, "~> 0.6"},
      {:ex_doc, "~> 0.27", only: :docs, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
    ]
  end

  # The deferror/1 macro writes the same three-way case for every raising
  # variant, so the branch for a charlist reason is dead wherever the wrapped
  # function only ever returns a binary one. That is a property of the macro, not
  # of the function it wrapped, and 58 of them buried everything else.
  # extra_return and missing_return are the pair that does catch something here:
  # a @spec that is narrower than what the function can actually return.
  defp dialyzer do
    [
      flags: [:extra_return, :missing_return, :error_handling, :unknown],
      plt_add_apps: [:mix],
      ignore_warnings: ".dialyzer_ignore.exs",
      list_unused_filters: true
    ]
  end

  defp description() do
    "TensorFlow Lite Elixir binding with optional TPU support."
  end

  defp package() do
    [
      name: to_string(@app),
      files: ~w(
        lib
        .formatter.exs
        mix.exs
        README*
        LICENSE*
      ),
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @github_url}
    ]
  end
end
