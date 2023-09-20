import openai


class LLM:
    def __init__(self, system_prompt, temperature=.7):
        self.system_prompt = system_prompt
        self.temperature = temperature

    def run(self, message):
        pass


class OpenAI(LLM):
    def __init__(self, system_prompt, api_key, temperature=.7):
        super().__init__(system_prompt, temperature)
        openai.api_key = api_key

    def run(self, message):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}
            ]
        )
        return response['choices'][0]['message']['content']
