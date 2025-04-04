defmodule TFLiteElixir.MixProject do
  use Mix.Project
  require Logger

  @app :tflite_elixir
  @version "0.3.9"
  @github_url "https://github.com/cocoa-xu/tflite_elixir"

  def project do
    [
      app: @app,
      version: @version,
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

  defp deps do
    [
      {:tflite_beam, "~> 0.3.9"},
      {:nx, "~> 0.5"},
      {:stb_image, "~> 0.6"},
      {:ex_doc, "~> 0.27", only: :docs, runtime: false}
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
