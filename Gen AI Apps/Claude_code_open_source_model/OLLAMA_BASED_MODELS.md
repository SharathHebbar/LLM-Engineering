# Run Claude Code using Local LLMs via Ollama

## Claude Code Installation
#### macOS, Linux, WSL:

```sh
curl -fsSL https://claude.ai/install.sh | bash
```

#### Windows PowerShell:

```sh
irm https://claude.ai/install.ps1 | iex
```

#### Windows CMD
```sh
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

## Connect Claude
Configure environment variables to use Ollama:

```sh
export ANTHROPIC_AUTH_TOKEN=ollama
export ANTHROPIC_BASE_URL=http://localhost:11434
```

## Run Claude Code with an Ollama model:

```sh
ollama launch claude --model gemma4
```