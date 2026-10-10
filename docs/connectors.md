# Connectors (planned)

> Not implemented yet. This page defines the intended contract so other Amy
> WorkspAIce apps can be built against it.

A **connector** is another app that the AI Platform can use as a set of tools,
for example "read this spreadsheet range" or "create a task".

## Contract

Each Amy WorkspAIce app (and any third-party service) that wants to be usable by
the AI exposes an **MCP server** ([Model Context Protocol](https://modelcontextprotocol.io))
over Streamable HTTP, for example at `https://<app>/mcp`.

- **Tools** are namespaced by app: `tasks.create_issue`, `docs.search`, `sheets.read_range`.
- **Auth:** the app issues an API key (later: OIDC client credentials). The user
  pastes the URL and key into the AI Platform's connector settings.
- **Permissions:** the user enables or disables individual tools per connector.
  Write tools require confirmation by default.

## Flow

1. The user registers a connector: URL + key.
2. The platform fetches the tool list and stores it.
3. In a chat, the platform offers enabled tools to the model (OpenAI tool-calling format).
4. When the model calls a tool, the platform invokes it over MCP and returns the
   result to the model, streaming progress to the UI.

## The other direction

Apps that want AI features don't need a connector. They call this platform's
`/v1/chat/completions` with an API key like any OpenAI client.
