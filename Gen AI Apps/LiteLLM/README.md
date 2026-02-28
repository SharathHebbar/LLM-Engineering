# Steps to try this

1. Create a `config.yaml` and add in all the necessary models
2. Create a `.env` file for storing API Keys
3. Load in the API KEY to the environment 
    ```bash
    export GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxx
    ```
4. Run the LiteLLM proxy
    ```bash
    litellm --config config.yaml --port 4000 --debug
    ```
5. Call the model using the script