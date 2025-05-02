import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, RunContextWrapper, handoff
from openai import AsyncOpenAI, api_key
from agents import enable_verbose_stdout_logging

load_dotenv()

def on_handooff(agent : Agent , ctx : RunContextWrapper[None]):
    agent_name  = agent.name
    print("----------------------------------")
    print(f"Handing off to Agent {agent_name}")
    print("----------------------------------")

async def setup_config():
    external_client = AsyncOpenAI(
        api_key=os.getenv('GOOGLE_API_KEY'),
        base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
    )

    model = OpenAIChatCompletionsModel(
        model='gemini-2.0-flash-exp',
        openai_client=external_client

    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    web_dev = Agent(
        name="Web Dev Agent",
        instructions=" You help user building a front-end using JS,HTML and CSS and some libraries that are now more popular",
        handoff_description="Must be handed off when dealing with front-end stuff",
        model=model,
    )

    devops_agent = Agent(
        name="devops_agent",
        instructions="You are a helpful agent that assist user's in deploying applications and managing the devops stuff",
        model=model
    )

    ai_engineer = Agent(
        name="ai_engineer",
        instructions="You are an ai engineer that helps user build AI applications and enterprise AI first solutions",
        model=model,

    )

    backend_agent = Agent(
        name="backend_agent",
        handoff_description="Must be handed off when dealing with backend-devops and ai related stuff",
        instructions="You are a helpful agent that helps user build backend in python and use your tools if the user ask's for devops or ai related stuff",

        tools=[
            devops_agent.as_tool(
                tool_name="devops_agent",
                tool_description="Help's user in deploying and managing containers and work with docker and rancher or K8's related stuff",

            ),

            ai_engineer.as_tool(
                tool_name="ai_engineer",
                tool_description="Help's user Intergrate AI into there existing application or build from scratch",
            ),
        ],
        model=model,
    )

    triage_agent = Agent(
        name="triage_agent",
        instructions="Your job is to pass the right query asked by the user to the right agent",
        model=model,
        handoffs=[
            handoff(web_dev, on_handoff= lambda ctx: on_handooff(web_dev,ctx=ctx)),
            handoff(backend_agent, on_handoff= lambda ctx: on_handooff(backend_agent,ctx=ctx)),
        ]
    )

    while True:
        user_input = input(
            "Do you want to build your product's frontend or backend and specify if doing devops or ai related stuff. (Type 'quit' to exit): ")
        if user_input.strip().lower() == 'quit':
            print("Exiting agent. Goodbye!")
            break
        result = await Runner.run(triage_agent, input=user_input)
        print(result.final_output)


if __name__ == '__main__':
    asyncio.run(setup_config())
    enable_verbose_stdout_logging()
