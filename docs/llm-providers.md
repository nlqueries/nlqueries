# LLM providers

NLQueries uses a language model to classify each question, write the SQL, correct it when the database refuses it, and phrase the answer. This page covers the providers it has been run against: **Anthropic** (the default), **OpenAI**, **Amazon Bedrock** and **DeepSeek**. Every setting named here is an environment variable, or a line in `./.env`; the full list is in [configuration.md](configuration.md).

---

## How a provider is chosen

Two clients do the work. `anthropic` calls the Anthropic API through Anthropic's SDK. `litellm` calls everything else through [LiteLLM](https://docs.litellm.ai/), where a model is written `provider/model-id` (`openai/gpt-4o`, `deepseek/deepseek-chat`, `bedrock/us.anthropic.claude-sonnet-4-6`).

`LLM_PROVIDER` names the client: `anthropic`, `litellm`, or `bedrock`, which is a name for `litellm`. Any other value fails with `Unknown LLM provider` when the first LLM call is made. Unset, it is detected in this order:

1. an `LLM_MODEL` starting `bedrock/` selects Bedrock;
2. `ANTHROPIC_API_KEY` selects `anthropic`;
3. `OPENAI_API_KEY` selects `litellm`, with OpenAI models as the defaults;
4. otherwise `anthropic`.

There are two model tiers. `LLM_MODEL` writes the SQL and the answers. `LLM_MODEL_FAST` handles the short calls: intent classification and follow-up resolution, which carry the question and, for a follow-up, the conversation. Set both when you change provider. A fast tier left on another provider's model sends those calls there.

A provider and a model that disagree are refused before anything is sent: an Anthropic provider with `LLM_MODEL=deepseek/deepseek-chat`, for example, would otherwise transmit the prompt and schema to api.anthropic.com before the model was reported missing. [Provider and model must agree](configuration.md#provider-and-model-must-agree) has the rules and the two cases that are not checked.

After changing provider, run:

```bash
nlqueries health
```

It makes one real call to the configured model and reports the provider, the model and the round-trip time, or the error.

---

## Anthropic

The default. One key is enough:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

| Setting | Default |
|---|---|
| `LLM_MODEL` | `claude-sonnet-5-5` |
| `LLM_MODEL_FAST` | `claude-haiku-5-5` |

`anthropic/claude-sonnet-5-5` is accepted too and treated as `claude-sonnet-5-5`.

**Effort.** Claude 5 and later think adaptively, and `LLM_EFFORT` caps how much: `low` (the default), `medium`, `high`, `xhigh` or `max`, sent as `output_config.effort`. Blank sends none, and the model uses its own default (`high` for Sonnet 5.5, `medium` for Haiku 5.5). Earlier Claude models are sent no effort. Claude 5 and later reject a `temperature`, so none is sent to them.

**A gateway.** `ANTHROPIC_BASE_URL` points the client at a proxy in front of the API. While it is set, the provider-and-model check is skipped, since the request no longer goes to api.anthropic.com. A plain `http://` URL to a host that is not loopback logs a warning, once per endpoint.

**Prompt caching.** The SQL prompt's stable part (the instructions and the agent's schema, the same for every question to that agent) is sent with an ephemeral `cache_control` marker, so repeat questions read it from Anthropic's prompt cache. Nothing needs setting.

---

## OpenAI

OpenAI is reached through LiteLLM. With only an OpenAI key set, that is detected:

```bash
OPENAI_API_KEY=sk-...
```

| Setting | Default with only `OPENAI_API_KEY` |
|---|---|
| `LLM_PROVIDER` | `litellm` |
| `LLM_MODEL` | `openai/gpt-4o` |
| `LLM_MODEL_FAST` | `openai/gpt-4o-mini` |

To pick another model, set `LLM_MODEL=openai/<model>`, and `LLM_MODEL_FAST` with it.

**With both keys set, Anthropic wins.** To use OpenAI then, set `LLM_PROVIDER=litellm`; the OpenAI defaults above apply again. `LLM_PROVIDER=openai` is not a provider name and fails.

**Reasoning models.** No effort is sent to OpenAI models: they use their own default. A reasoning model bills its private reasoning from the same allowance as the answer, and spends it first, so raise `LLM_MAX_OUTPUT_TOKENS` (below) if answers come back short or empty. `nlqueries health` uses a 16-token probe, the smallest that OpenAI's Responses API accepts, and reports a reasoning model that spends it all before writing anything as reachable, with a note.

---

## Amazon Bedrock

Bedrock is reached through LiteLLM and authenticates through boto3, so there is no API key. Set the model and the region:

```bash
LLM_MODEL=bedrock/us.anthropic.claude-sonnet-4-6
LLM_MODEL_FAST=bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0
AWS_REGION=us-east-1
```

Credentials come from the usual AWS chain: `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in the environment, a profile, or the role of the EC2 instance, ECS task or EKS pod. Unset, `LLM_MODEL_FAST` follows `LLM_MODEL`, which works but costs the main model's price for every short call. A `LLM_MODEL_FAST` that is not a `bedrock/` id is refused, so a leftover Anthropic id cannot send the question outside your AWS account.

[Amazon Bedrock](configuration.md#amazon-bedrock) covers the rest, and each item fails in its own way:

- how the region is chosen (LiteLLM falls back to `us-west-2` silently);
- enabling model access;
- the `us.`/`eu.` cross-region inference profiles that newer Claude models need;
- the IAM permissions: `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream`.

**Effort.** Not sent on Bedrock: each model keeps its own default. `LLM_EFFORT` applies only on routes that reach the Anthropic API.

**Prompt caching.** For Claude on Bedrock, where LiteLLM lists the model as supporting prompt caching, the stable part of the SQL prompt is marked as on Anthropic, and LiteLLM sends it as Bedrock's `cachePoint`. Other Bedrock models are sent the prompt without a marker.

---

## DeepSeek

DeepSeek is reached through LiteLLM, which reads `DEEPSEEK_API_KEY`. Name the provider and both models:

```bash
DEEPSEEK_API_KEY=sk-...
LLM_PROVIDER=litellm
LLM_MODEL=deepseek/deepseek-chat
LLM_MODEL_FAST=deepseek/deepseek-chat
```

`LLM_PROVIDER=litellm` is required. Without it, and with no Anthropic or OpenAI key set, the provider is detected as `anthropic`, and the first call is refused because a `deepseek/` model was given to it. `LLM_MODEL_FAST` is required too. Its default is `claude-haiku-5-5`, which LiteLLM sends to Anthropic.

**Thinking.** DeepSeek thinks before it answers by default. `LLM_THINKING=off` turns that off (`thinking: {"type": "disabled"}`). With it on, `LLM_EFFORT` is sent as DeepSeek's `reasoning_effort`, which has three levels:

| `LLM_EFFORT` | `reasoning_effort` sent |
|---|---|
| `low` (default), `medium` | `low` |
| `high` | `high` |
| `xhigh`, `max` | `max` |

Both go in the request's `extra_body`, because LiteLLM rewrites a top-level `reasoning_effort` for DeepSeek and drops the level. With `LLM_THINKING=off` no effort is sent.

Thinking is billed from `LLM_MAX_OUTPUT_TOKENS`, and spent first. See [Output budget](#output-budget).

**The CLI's credential checks recognise `DEEPSEEK_API_KEY`**, as they do `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and a Bedrock configuration. So with only a DeepSeek key:

- `nlqueries health` and the MCP `health` tool report the credential;
- `process-history` annotates;
- `export-kb --describe-columns` writes the descriptions.

---

## Settings that apply to every provider

### Output budget

`LLM_MAX_OUTPUT_TOKENS` (default `4096`) is the number of tokens an answer may generate. The other budgets derive from it:

- a SQL correction gets half, at least 512;
- classification and follow-up resolution get an eighth, at least 200.

It is a ceiling, not a charge: tokens are billed as generated. A reasoning model spends its reasoning from the same allowance before it writes anything, so a budget sized for the answer alone returns a short answer, or nothing. On most calls, producing nothing raises `OutputBudgetExhausted`, naming the model and the budget. Follow-up resolution and knowledge-base column descriptions log it as a warning instead and carry on. Raise this setting for a model that reasons at length. The details are in [configuration.md](configuration.md).

### Timeout

`LLM_TIMEOUT_SECONDS` (default `180`) is how long a call may go **without producing data** before it fails with `LLMTimeout`. It is a read deadline, not a cap on the whole call: a stream that keeps sending can run past it.

---

## Token usage

Core does not price calls. A Python application embedding NLQueries can collect the token counts of every LLM call a block of work makes, with `nlqueries.llm.use_usage_sink`:

```python
from nlqueries.llm import UsageRecord, use_usage_sink

records: list[UsageRecord] = []
with use_usage_sink(records):
    ...  # run_query(...) and anything else that calls the model
```

Each `UsageRecord` carries the model, `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `reasoning_tokens` (already counted in `output_tokens`), and `estimated`. That last flag is true when the provider reported no usage and the counts are a rough character-based estimate. Streamed answers on OpenAI-compatible routes, DeepSeek and OpenAI among them, ask the provider for their usage, so they are reported rather than estimated.

---

## Other models

LiteLLM reaches many more providers than the four above, and `LLM_PROVIDER=litellm` with a `provider/model-id` passes the model through. They are not covered here because NLQueries has not been run against them. In particular, no effort is sent to them, and their prompts carry no cache markers.

[What leaves this machine](configuration.md#what-leaves-this-machine) lists what every LLM call carries to the provider, whichever one it is.
