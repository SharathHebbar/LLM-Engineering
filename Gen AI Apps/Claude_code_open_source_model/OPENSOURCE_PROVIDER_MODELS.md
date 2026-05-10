# Open Source Claude Code: Load Open Source Models to Claude Code

## Install Claude Code

```sh
npm install -g @anthropic-ai/claude-code
```
Verify Installation

```sh
claude --version
```

## Install LiteLLM
Create a Python environment and install LiteLLM.

#### Create Project Environment

```sh
uv init opensource_claudecode_demo
```

```sh
cd opensource_claudecode_demo
```

#### Create Virtual Environment

```sh
uv venv
```

#### Activate the environment:

Install LiteLLM

```sh
uv add litellm "litellm[proxy]"
```

## Get Your NVIDIA API Key


Go to: Nvidia Build Platform
Steps:
Sign up (free, no credit card required)
Open Profile
Navigate to API Keys
Generate a new key
Copy the key

Your key will look like:
nvapi-xxxxxxxxxxxxxxxx


## Create Your Project Folder

```sh
mkdir opensource_claudecode_demo
cd opensource_claudecode_demo
```

## Create config.yaml

#### Create a file named config.yaml Add the following configuration:


config.yaml
```yaml
model_list:
  - model_name: claude-4-sonnet
    litellm_params:
      model: nvidia_nim/minimaxai/minimax-m2.7
      api_base: https://integrate.api.nvidia.com/v1
      api_key: nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

  - model_name: claude-4-opus
    litellm_params:
      model: nvidia_nim/stepfun-ai/step-3.5-flash
      api_base: https://integrate.api.nvidia.com/v1
      api_key: nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

  - model_name: claude-cohere-command-a
    litellm_params:
      model: cohere/command-a
      api_key: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

  - model_name: claude-google-gemini-3-flash-preview
    litellm_params:
      model: gemini/gemini-3-flash-preview
      api_key: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  
  - model_name: claude-groq-gpt-oss-120b
    litellm_params:
      model: groq/gpt-oss-120b
      api_base: https://api.groq.com/openai/v1
      api_key: gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

litellm_settings:
  drop_params: true
```

## Export Your NVIDIA API Key

```sh
export NVIDIA_API_KEY="nvapi-xxxxxxxx"
```

Or paste it in config.yaml in api_key 

## Create below folder

```sh
mkdir .claude
```

Create settings.json inside .claude folder and paste

settings.json

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:4000",
    "ANTHROPIC_AUTH_TOKEN": "any-key-works"
  }
}
```

or

```sh
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=any-key-works
```

## Start LiteLLM Proxy

Run:

```sh
litellm --config config.yaml
```

By default it starts at:

```sh
http://localhost:4000
```
## Start Claude code

```sh
claude
```
## Additional Information
- Use claude as a prefix to the models or else the claude code would not detect it as a model
- Also export the major models as HAIKU, SONNET and OPUS for more reliability as it has less chances of breaking the system

```sh
export ANTHROPIC_DEFAULT_HAIKU_MODEL="groq/model"
export ANTHROPIC_DEFAULT_SONNET_MODEL="groq/model"
export ANTHROPIC_DEFAULT_OPUS_MODEL="groq/model"
```
- Tried with major providers 
- - Nvidia build - Worked
- - Groq - Didnt work
- - Cerebras - Didnt work
- - HF - Didnt work
- - Gemini - Didnt work
- - Cohere - Didnt work
