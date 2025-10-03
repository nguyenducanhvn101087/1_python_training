# This version uses OpenAI which can incurse costs. Please be aware of your usage.
from langchain_core.messages import HumanMessage  # highe level framework to build AI applications
from langchain_openai import ChatOpenAI # to use open AI within langchain and langgraph
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent # complex framework to build AI agents
from dotenv import load_dotenv # load environment variables in .env file from within python script

# Diff AI chatbot vs AI agent 
# AI chatbot is a simple conversational agent that can respond to user inputs, while 
# AI agent is a more complex system that can perform tasks, make decisions, and interact with other systems i.e. can access tools and APIs

# In this example make a calculator
load_dotenv() # load environment variables from .env file

@tool # decorator to makes it available for AI agents. Otherwise agent won't know about it
def calculator(a: float, b: float) -> str: # Takes two arguments, a and b, both expected to be floats. Returns a string.
    """Useful for performing basic arithmetic calculations with numbers.""" # Doc string: tell agent what this tool does
    # This is important for agents that automatically decide which tool to call.

    print("Adding ...")
    return f"The sum of {a} and {b} is {a + b}"

def main(): # initialize chatbot/agent
    # initialize chat model with gpt-4o and temperature 0 for deterministic responses (higher temp => more randomness)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0) 

    tools = [calculator] # tools or external services that agent can utilize

    # create_react_agent = prebuilt agent taking some model and tools and automatically handles how to use tools and utilize model
    agent_executor = create_react_agent(model, tools)  

    # Some static welcome message
    print("Welcome to the AI Agent Calculator! Type 'exit' to quit.")
    print("You can ask me to perform calculations for you or chat with me.")

    while True: # infinite loop to keep chatbot running
        user_input = input("\nYou: ").strip() # get user input and stripping whitespaces

        if user_input.lower() == 'exit': # if user types exit, break the loop and end program
            print("Goodbye!")
            break
        
        # Assistant response
        print("\nAssistant: ", end = "") # by default end ="\n" => override

        # stream response from LLM using agent_executor.stream based on a human input. The response is yielded in chunks
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ): # loop through each chunk of response
            if "agent" in chunk and "messages" in chunk["agent"]: # If chunk contains agent messages
                for messsage in chunk["agent"]["messages"]: # loop through each message
                    print(messsage["content"], end="") # stream the output giving the impression the agent is typing rather than outputting it all at once

        print()

if __name__ == "__main__":
    main()# run the main function only if this script is run directly vs e.g. called from another file (convention)


