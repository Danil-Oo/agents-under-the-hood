import inspect
import re  # noqa: F401

from dotenv import load_dotenv

load_dotenv()
import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7B"

# ----- Tools -----


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Ищет стоимость продукта в каталоге"""
    print(f"  >> Выполняю get_product_price(product:{product})")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Применяет скидку к цене в зависимости от уровня скидки и возвращает итоговую цену
    Возможные уровни: bronze, silver, gold."""
    print(
        f"  >> Выполняю apply_discount(price={price}, discount_tier={discount_tier})"
    )
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


# Мы удалили json схемы с описание и вместо этого создадим функцию для извлечения
# описаний функций. Теперь вся информация о функциях будет находиться в самом
# промпте в виде обычного текста.
def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # Из-за того, что все функции обернуты в декоратор @traceable, нам нужно
        # получить из каждой функции атрибут __wrapped__
        original_function = getattr(
            tool_function, "__wrapped__", tool_function
        )
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")

    return "\n".join(descriptions)


tools_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
СТРОГИЕ ПРАВИЛА - ты должен придерживаться этих правил:
1. Никогда не угадывай и не предполагай стоимость любого продукта. Ты должен 
сначала вызвать функцию get_product_price чтобы получить настоящую цену.
2. Вызывай функцию apply_discount только после того как ты получил цену от 
get_product_price. Пересчитай с учетом скидки иммено то число, которое ты 
получил из get_product_price, не возвращай просто вымышленное число.
3. Никогда не рассчитывай скидку самостоятельно, используя математику, всегда 
используй apply_discount функцию.
4. Если пользователь не указывает уровень скидки, спроси у него какой уровень 
скидки он имеет, не предугадывай уровень скидки.
5. Пользователь будет задавать тебе вопросы на русском, а аргументы для функции 
тебе нужно подать на английском языке, Вот тебе небольшой словарь для 
аргументов: 
ноутбук - laptop
наушники - headphones
клавиатура - keyboard
золотая скидка - gold
серебряная скидка - silver
бронзовая скидка - bronze.

Ответь на следующие вопросы настолько хорошо, насколько сможешь. У тебя есть 
доступ к следующим инструментам:

{tools_descriptions}

Используй следующий формат:

Question: входной вопрос, на который нужно ответить
Thought: ты всегда должен обдумывать, что делать
Action: действие, которое нужно предпринять, должно быть одним из [{tool_names}]
Action Input: входные данные для действия
Observation: результат действия
... (этот цикл Thought/Action/Action Input/Observation может повторяться N раз)
Thought: теперь я знаю окончательный ответ
Final Answer: окончательный ответ на исходный вопрос

Начинай!

Question: {{question}}
Thought:"""


# ----- Helper: traced Ollama call -----
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


# ----- Agent Loop -----
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):

    print(f"Question: {question}")
    print("=" * 60)

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        full_prompt = prompt + scratchpad
        # Стоп-токен не позволяет LLM генерировать собственное наблюдение —
        # вместо этого мы вводим реальный результат инструмента
        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.message.content
        print(f"LLM Output: {output}")

        print("  [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("=" * 60)
            print(f"  [Final Answer Found] {final_answer}")
            return final_answer

        # Парсим вызов функции с помощью regex из сырого тектса, это "хрупкая"
        # реализация, потому что все работает только в том случае, если LLM
        # вернет ответ именно в том формате, какой мы от нее ожидаем, если она
        # что-то перепуает или начнет галлюцинировать, то все сломается
        print(
            "  [Parsing] Looking for Action and Action Input in LLM output..."
        )

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()
        print(
            f" [Выбран инструмент] {tool_name} с аргументами: {tool_input_raw}"
        )

        #  Split comma-separated args; strip key= prefix if LLM outputs key=value format
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")

        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))
        print(f"[Результат от инструмента] {observation}")

        # Отправляем в историю (scratchpad) результат применения инструмента,
        # история - это по сути большая строка, которая постепенно наполняется
        # новой информацией от агента. В данном случае, эта строка заменяет
        # messages.append
        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ОШИБКА: достигнуто максимальное число итераций, ответ не получен.")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent(
        "Какая будет цена ноутбука после применения золотой скидки?"
    )
