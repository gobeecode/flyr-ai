import json

import gradio as gr
import ollama

ticket_prices = {'coimbatore': '$100', 'chennai': '$600', 'bangalore': '$1200'}


def get_ticket_price(city):
    print(f'Tool called for {city}')
    city = city.lower()
    return ticket_prices.get(city, 'unknown')


def handle_tool_call(tool_data):
    city = tool_data.get("city")
    price = get_ticket_price(city)
    response = {
        "role": "tool",
        "content": json.dumps({"city": city, "price": price})
    }
    return response, city


price_function = {
    'name': 'get_ticket_price',
    'description': 'Gets the price of a ticket. Call this function whenever you need to know the price of a ticket. For example, when the customer asks what is the price of a hat.',
    'parameters': {
        'type': 'object',
        'properties': {

            'city': {
                'type': 'string',
                'description': 'Name of the city to travel.'
            }
        },
        'required': ['city'],
        'additionalProperties': False
    }
}

tools = [{'type': 'function', 'function': price_function}]

def chat(message: str, history: list):
    model = 'llama3.2'
    system_prompt = (
        "You are a helpful assistant for an airline called Flyr AI. You should respond in a short and crispy way. "
        "If you do not know anything, say so. Do not add any additional unknown information.")
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
    messages.append({"role": "user", "content": message})
    print("History: ", history)
    print("Messages: ", messages)
    response = ollama.chat(model=model, messages=messages, tools=tools)
    message = response["message"]
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        tool_name = tool_call["function"]["name"]
        tool_args = tool_call["function"]["arguments"]
        print(f"Tool requested: {tool_name} with args {tool_args}")
        response, ticket = handle_tool_call(tool_args)
        messages.append(message)
        messages.append(response)
    response = ollama.chat(model=model, messages=messages, stream=True)
    all_content = ''
    for chunk in response:
        content = chunk.get("message", {}).get("content", "")
        all_content += content
        yield all_content


def main():
    view = gr.ChatInterface(fn=chat)
    view.launch()


if __name__ == '__main__':
    main()
