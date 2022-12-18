import torch as t
import torch.nn.functional as F

def greedy_search(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, )

    Return: the most likely token (as an integer)
    '''
    return int(logits.argmax(dim=-1).squeeze())


def sample_basic(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    distribution = t.distributions.categorical.Categorical(logits=logits)
    return int(distribution.sample())


def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    '''
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    assert temperature > 0
    logits = logits / temperature
    return logits


def apply_freq_penalty(input: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    '''
    input: shape (seq, )
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    vocab_size = logits.shape[0]
    counts = t.bincount(input=input, minlength=vocab_size)
    
    return logits - counts * freq_penalty


def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    '''
    top_logits = t.topk(logits, k=top_k)
    sample_idx = t.distributions.categorical.Categorical(logits=top_logits.values).sample()
    return int(top_logits.indices[sample_idx])


def sample_top_p(logits: t.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    sorted_logits, logit_indices = logits.sort(descending=True)
    cumulative = sorted_logits.softmax(-1).cumsum(dim=-1)

    select_count = max(t.searchsorted(cumulative, top_p, right=False).item()+1, min_tokens_to_keep)
    select_indices = logit_indices[:select_count]
    select_logits = logits[select_indices]

    sample_idx = t.distributions.categorical.Categorical(logits=select_logits).sample()

    return int(select_indices[sample_idx])


def apply_sampling_methods(
    input: t.Tensor, logits: t.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0
) -> int:
    '''
    Return the next token, sampled from the model's probability distribution with modifiers.
x
    input: shape (seq,)
    '''
    assert input.ndim == 1, "input should be a 1D sequence of token ids"
    assert temperature >= 0, "Temperature should be non-negative"
    assert 0 <= top_p <= 1.0, "Top-p must be a probability"
    assert 0 <= top_k, "Top-k must be non-negative"
    assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

    if temperature == 0:
        return greedy_search(logits)
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)
    if freq_penalty != 0.0:
        logits = apply_freq_penalty(input, logits, freq_penalty)
    if top_k > 0:
        return sample_top_k(logits, top_k)
    if top_p > 0:
        return sample_top_p(logits, top_p)
    return sample_basic(logits)

def sample_tokens(
    model,
    tokenizer,
    initial_text: str,
    max_tokens_generated: int = 30,
    **kwargs
) -> str:
    '''
    Sample tokens until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    '''
    model.eval()
    input: list = tokenizer.encode(initial_text)
    generated = []
    device = next(model.parameters()).device
    for _ in range(max_tokens_generated):
        new_input = t.tensor(input + generated, dtype=t.int64, device=device)
        new_input_truncated = new_input[-min(tokenizer.model_max_length, new_input.shape[0]):].unsqueeze(0)
        output = model(new_input_truncated)
        all_logits = output if isinstance(output, t.Tensor) else output.logits
        logits = all_logits[0, -1]
        new_token = apply_sampling_methods(new_input, logits, **kwargs)
        assert isinstance(new_token, int)
        generated.append(output)
        if new_token == getattr(tokenizer, "eos_token_id", None):
            break
    return tokenizer.decode(input + generated)


def sample_tokens_no_detokenization(
    model,
    input,
    max_tokens_generated: int = 30,
    max_seq_len: int = 5,
    **kwargs
) -> str:
    '''
    Sample tokens until the specified token limit is reached.

    Return: the prompt and continuation concatenated
    '''
    model.eval()
    generated = []
    device = next(model.parameters()).device
    for _ in range(max_tokens_generated):
        new_input = t.tensor(input + generated, dtype=t.int64, device=device)
        new_input_truncated = new_input[-min(max_seq_len, new_input.shape[0]):].unsqueeze(0)
        output = model(new_input_truncated)
        all_logits = output if isinstance(output, t.Tensor) else output.logits
        logits = all_logits[0, -1]
        new_token = apply_sampling_methods(new_input, logits, **kwargs)
        assert isinstance(new_token, int)
        generated.append(new_token)

    return input + generated


def sample_numbers(
    model,
    input: str,
    max_tokens_generated: int = 30,
    max_seq_len: int = 5,
    **kwargs
) -> str:
    '''
    Sample tokens until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    '''
    model.eval()
    generated = []
    device = next(model.parameters()).device
    for _ in range(max_tokens_generated):
        new_input = t.tensor(input + generated, dtype=t.int64, device=device)
        # Ensure that the model is not fed more than 5 tokens at a time
        new_input_truncated = new_input[-min(max_seq_len, new_input.shape[0]):].unsqueeze(0)

        output = model(new_input_truncated)
        #print(output.shape)
        output = output[0, -1]
        output = output.to(t.int64).item()
        #all_logits = output if isinstance(output, t.Tensor) else output.logits
        #
        #new_token = apply_sampling_methods(new_input, logits, **kwargs)
        #assert isinstance(new_token, int)
        generated.append(output)
        #if new_token > 1_000_000:
        #    break
    return input + generated