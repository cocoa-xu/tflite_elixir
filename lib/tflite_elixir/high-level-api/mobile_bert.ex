defmodule TFLiteElixir.MobileBert do
  defstruct [:tensors, :interpreter, :vocab_map]
  alias __MODULE__, as: T
  alias TFLiteElixir.FlatBufferModel
  alias TFLiteElixir.Interpreter
  alias TFLiteElixir.TFLiteTensor
  alias TFLiteElixir.Tokenizer.FullTokenizer

  @max_seq_len 384
  @max_ans_len 32
  @output_offset 1
  @predict_answer_count 5

  @doc """
  Load a MobileBERT question-answering model from `model_file`.

  The vocabulary comes from the model's own `vocab.txt`, so a model without one
  is refused by name rather than failing somewhere inside the tokenizer.

      {:ok, bert} = TFLiteElixir.MobileBert.init("mobilebert.tflite")
      answers = TFLiteElixir.MobileBert.run(bert, query, content)

  """
  @spec init(String.t()) :: {:ok, %T{}}
  def init(model_file) do
    model_buffer = File.read!(model_file)
    # get_associated_file/2 answers a map, a string or {:error, _}, and only the
    # string is a vocabulary. Handing the other two to String.split raised out of
    # String, naming neither this model nor the file it is missing.
    vocab =
      case FlatBufferModel.get_associated_file(model_buffer, "vocab.txt") do
        text when is_binary(text) ->
          text

        {:error, reason} ->
          raise ArgumentError,
                "#{model_file} carries no vocab.txt, which MobileBert needs: #{reason}"

        other ->
          raise ArgumentError,
                "#{model_file} answered #{inspect(other)} for vocab.txt, expected its contents"
      end

    vocabs = String.split(vocab, "\n")
    vocab_map = Map.new(Enum.with_index(vocabs))
    {:ok, interpreter} = Interpreter.new_from_buffer(model_buffer)

    # This was a with/1 over tagged tuples whose else clauses dialyzer could not
    # agree with itself about: OTP 26 called two of them unreachable and OTP 28
    # did not. Each check now says what it wants and raises where it fails, which
    # both versions can follow.
    {input_ids_idx, input_mask_idx, segment_ids_idx} = input_indices(interpreter)
    {end_logits_idx, start_logits_idx} = output_indices(interpreter)

    input_ids_tensor = input_tensor_of_expected_shape(interpreter, input_ids_idx)
    input_mask_tensor = input_tensor_of_expected_shape(interpreter, input_mask_idx)
    segment_ids_tensor = input_tensor_of_expected_shape(interpreter, segment_ids_idx)
    end_logits_tensor = output_tensor_of_expected_shape(interpreter, end_logits_idx)
    start_logits_tensor = output_tensor_of_expected_shape(interpreter, start_logits_idx)

    {:ok,
     %T{
       tensors: %{
         :input_ids => input_ids_tensor,
         :input_mask => input_mask_tensor,
         :segment_ids => segment_ids_tensor,
         :end_logits => end_logits_tensor,
         :start_logits => start_logits_tensor
       },
       interpreter: interpreter,
       vocab_map: vocab_map
     }}
  end

  defp input_indices(interpreter) do
    case Interpreter.inputs(interpreter) do
      {:ok, [input_ids, input_mask, segment_ids]} ->
        {input_ids, input_mask, segment_ids}

      _ ->
        raise RuntimeError, "Unexpected model: Number of input tensors"
    end
  end

  defp output_indices(interpreter) do
    case Interpreter.outputs(interpreter) do
      {:ok, [end_logits, start_logits]} ->
        {end_logits, start_logits}

      _ ->
        raise RuntimeError, "Unexpected model: Number of Output Tensors"
    end
  end

  defp input_tensor_of_expected_shape(interpreter, index) do
    case Interpreter.tensor(interpreter, index) do
      tensor = %TFLiteTensor{shape: {1, 384}} ->
        tensor

      _ ->
        raise RuntimeError, "Unexpected model: Expect input tensor shape to be {1, 384}"
    end
  end

  defp output_tensor_of_expected_shape(interpreter, index) do
    case Interpreter.tensor(interpreter, index) do
      tensor = %TFLiteTensor{shape: {1, 384}} ->
        tensor

      %TFLiteTensor{} = tensor ->
        raise RuntimeError,
              "Unexpected model: Expect output tensor (#{tensor.name}) shape to be {1, 384}, " <>
                "got #{inspect(tensor.shape)}"

      {:error, reason} ->
        raise RuntimeError,
              "Unexpected model: output tensor #{index} could not be read: #{reason}"
    end
  end

  @doc """
  Answer `query` from `content`.

  Returns up to five `{score, excerpt}` tuples, best first, where each excerpt is
  a span of `content` and the scores are a softmax over the spans considered.
  """
  def run(self = %T{}, query, content) when is_binary(query) and is_binary(content) do
    {features, content_data} = preprocessing(self.vocab_map, query, content)

    :ok = TFLiteTensor.set_data(self.tensors.input_ids, Nx.tensor(features.input_ids, type: :s32))

    :ok =
      TFLiteTensor.set_data(self.tensors.input_mask, Nx.tensor(features.input_mask, type: :s32))

    :ok =
      TFLiteTensor.set_data(self.tensors.segment_ids, Nx.tensor(features.segment_ids, type: :s32))

    :ok = Interpreter.invoke(self.interpreter)

    end_logits = Nx.squeeze(TFLiteTensor.to_nx(self.tensors.end_logits))
    start_logits = Nx.squeeze(TFLiteTensor.to_nx(self.tensors.start_logits))

    postprocessing(start_logits, end_logits, content_data)
  end

  @doc false
  # Public only so the padding arithmetic can be tested without a model: it needs
  # a vocabulary and two strings, nothing else.
  def preprocessing(vocab_map, query, content) do
    query_tokens = FullTokenizer.tokenize(query, true, vocab_map)
    content_words = String.split(content)

    content_tokens = Enum.map(content_words, &FullTokenizer.tokenize(&1, true, vocab_map))

    content_token_idx_to_word_idx_mapping =
      for {token, i} <- Enum.with_index(content_tokens), reduce: [] do
        acc ->
          [List.duplicate(i, Enum.count(token)) | acc]
      end
      |> Enum.reverse()
      |> List.flatten()

    content_tokens = List.flatten(content_tokens)

    # -3 accounts for [CLS], [SEP] and [SEP].
    max_content_len = @max_seq_len - Enum.count(query_tokens) - 3
    content_tokens = Enum.take(content_tokens, max_content_len)

    # Start of generating the `InputFeatures`.
    tokens = ["[CLS]" | query_tokens]
    segment_ids = List.duplicate(0, Enum.count(query_tokens) + 1)

    tokens = tokens ++ ["[SEP]"] ++ content_tokens
    segment_ids = segment_ids ++ [0] ++ List.duplicate(1, Enum.count(content_tokens))

    tokens_count = Enum.count(query_tokens) + 2

    token_idx_to_word_idx_mapping =
      for {_doc_token, i} <- Enum.with_index(content_tokens), reduce: [] do
        acc ->
          [{i + tokens_count, Enum.at(content_token_idx_to_word_idx_mapping, i)} | acc]
      end
      |> Map.new()

    tokens = tokens ++ ["[SEP]"]
    segment_ids = segment_ids ++ [1]

    {:ok, input_ids} = FullTokenizer.convert_to_id(tokens, vocab_map)
    input_mask = List.duplicate(1, Enum.count(input_ids))

    # The other way round. content_tokens is already cut to
    # @max_seq_len - length(query) - 3 above, and the three brackets put back
    # exactly those three, so input_ids can never be longer than @max_seq_len.
    # Subtracting in the old order therefore gave zero or less every time, the
    # branch below never ran, and the tensors went to the model short of their
    # {1, 384}. run/3 matches :ok on set_data, so that was a MatchError for any
    # input that did not happen to fill the sequence exactly.
    n_padding = @max_seq_len - Enum.count(input_ids)

    {input_ids, input_mask, segment_ids} =
      if n_padding > 0 do
        padding = List.duplicate(0, n_padding)

        {
          input_ids ++ padding,
          input_mask ++ padding,
          segment_ids ++ padding
        }
      else
        {input_ids, input_mask, segment_ids}
      end

    {
      %{
        :input_ids => input_ids,
        :input_mask => input_mask,
        :segment_ids => segment_ids
      },
      %{
        :content_words => content_words,
        :token_idx_to_word_idx_mapping => token_idx_to_word_idx_mapping,
        :original_content => content
      }
    }
  end

  defp postprocessing(start_logits, end_logits, content_data) do
    start_indexes = candidate_answer_indexes(start_logits)
    end_indexes = candidate_answer_indexes(end_logits)

    word_range =
      for start <- start_indexes, end_idx <- end_indexes, reduce: [] do
        acc ->
          if start <= end_idx do
            if end_idx - start + 1 < @max_ans_len do
              start_index = content_data.token_idx_to_word_idx_mapping[start + @output_offset]
              end_index = content_data.token_idx_to_word_idx_mapping[end_idx + @output_offset]

              if start_index < end_index do
                [
                  {start_index, end_index,
                   Nx.to_number(Nx.add(start_logits[start], end_logits[end_idx]))}
                  | acc
                ]
              else
                acc
              end
            else
              acc
            end
          else
            acc
          end
      end
      |> Enum.reject(&is_nil/1)
      |> Enum.sort(fn {_, _, a_logit}, {_, _, b_logit} ->
        a_logit > b_logit
      end)
      |> Enum.take(@predict_answer_count)

    answers =
      softmaxed(word_range)
      |> Enum.map(fn {score, {start_idx, end_idx, _}} ->
        {score, excerpt_words(content_data, start_idx, end_idx)}
      end)
      |> Enum.reject(&is_nil/1)

    answers
  end

  defp excerpt_words(content_data, start_idx, end_idx) do
    pattern =
      Enum.slice(content_data.content_words, start_idx..(end_idx - 1))
      |> Enum.map(&Regex.escape/1)
      |> Enum.join("\\s+")

    with {:ok, reg} <- Regex.compile(pattern),
         exceprt = Regex.run(reg, content_data.original_content),
         [first_match | _] <- exceprt do
      first_match
    else
      _ ->
        nil
    end
  end

  defp softmaxed([]), do: []

  defp softmaxed(word_range) do
    max_logit = elem(Enum.at(word_range, 0), 2)

    numerators =
      Enum.map(word_range, fn {_, _, l} ->
        :math.exp(l - max_logit)
      end)

    sum = Enum.sum(numerators)
    Enum.map(Enum.zip(numerators, word_range), fn {s, word_r} -> {s / sum, word_r} end)
  end

  defp candidate_answer_indexes(logits) do
    Nx.to_flat_list(logits[[0..(@max_seq_len - 1)]])
    |> Enum.with_index()
    |> Enum.sort(fn {a, _}, {b, _} -> a > b end)
    |> Enum.take(@predict_answer_count)
    |> Enum.map(fn {_, offset} -> offset end)
  end
end
