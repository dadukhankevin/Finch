import openai


class LLM:
    def __init__(self, system_prompt, temperature=.7):
        self.system_prompt = system_prompt
        self.temperature = temperature

    def run(self, message):
        pass


class OpenAI(LLM):
    def __init__(self, api_key, temperature=.7, system_prompt='You are an evolutionary algorithm'):
        super().__init__(system_prompt, temperature)
        openai.api_key = api_key

    def run(self, message):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message}
            ]
        )
        return response['choices'][0]['message']['content']
