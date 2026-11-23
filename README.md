# data-analyst

A conversational data analyst built with the [Mistral Agents SDK](https://docs.mistral.ai/studio/agents/getting-started/quickstart).
Give it a CSV file from your Google Drive and ask questions in plain language;
it answers with real numbers computed through a fixed set of tools.

- **File access**: through the `google_drive_mcp` Studio connector, which acts
  as the user who runs the session (the connector must be connected in
  [Studio](https://console.mistral.ai/build/connectors)).
- **Fixed tools** (no shell, no sandbox, no code interpreter — the only way to
  touch the data):
  - `load_csv` — parses CSV content read from Drive and caches it for the
    session.
  - `describe_columns` — per-column dtype, nulls, uniques and stats.
  - `filter_rows` — fixed condition ops (`eq`, `gt`, `contains`, `in`, ...).
  - `group_by_aggregate` — group-by with fixed aggregations
    (`count`, `sum`, `mean`, ...), optional filters, sorting.
- **Memory**: the agent runs as a durable session; the dataset is cached per
  session, so follow-ups reuse the file without re-sharing it.

Set `MISTRAL_API_KEY` in `.env`, with a key from Studio's [API keys](https://console.mistral.ai/api-keys)
page. `.env` is git-ignored: never commit it. If you export `MISTRAL_API_KEY`
in your shell instead, it takes precedence over `.env`. Then:

```bash
uv run --env-file .env agents info     # show what the CLI loads
uv run --env-file .env agents chat     # chat with the agent in this process
uv run --env-file .env agents serve    # run it as a durable worker
uv run --env-file .env agents deploy   # deploy it as a managed agent
```

`agents deploy` builds the commit you pushed: the project must be in a GitHub
repository, and the GitHub Connector must be enabled on your Mistral account
in Studio. Because the agent uses a connector that acts as the session user,
the deployment is hardened (the CLI does this by default).

After deploying, chat with the deployed agent with `uv run --env-file .env
agents chat data-analyst`, from the `Try the agent` link `agents deploy`
prints (opens a chat in Vibe Code Web), or from any program with
`agents.Agent.load_from(agent_name="data-analyst")`.
