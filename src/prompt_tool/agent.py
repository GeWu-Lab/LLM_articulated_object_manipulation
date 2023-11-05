import openai
openai.api_key = "xx"
from prompt_tool.prompt import Prompt

class Agent:
    def __init__(self,  parser, model = "gpt-4"):
        self.model = model
        self.parser = parser

    def request(self, prompt):
        response = openai.ChatCompletion.create(
            model = self.model,
            messages = prompt.get_content(),
            temperature = 0.0005,
        )
        return self.parse(response) 

    def parse(self,response):
        ERROR, points = self.parser.parse(response) 
        if not ERROR:
            print("The points are: ", points)
        else:
            print(ERROR)
        return ERROR, points
    
    def feedback(self):
        pass

if __name__ == "__main__":
    content = [
        {"role": "system", "content": "you are a helpful assistant, but you should answer the question as brief as possible"},
        {"role": "user", "content": "Could you tell me how to open the door?"}
    ]
    prompt = Prompt(content=content)
    agent = Agent()
    print(agent.prompt.get_content())
    answer = agent.request(prompt)
    prompt.add_response(answer)
    

    prompt.add_request({"role": "user", "content": "What is the most important step to open the door?"})
    agent.request(prompt)
    