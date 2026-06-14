from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool  # noqa: I001
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage  # noqa: F401
from langchain.chat_models import init_chat_model  # noqa: F401
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7B"

# ----- Tools -----


@tool
def get_product_price(product: str) -> float:
    """Ищет стоимость продукта в каталоге"""
    print(f"  >> Выполняю get_product_price(product:{product})")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Применяет скидку к цене в зависимости от уровня скидки и возвращает итоговую цену
    Возможные уровни: bronze, silver, gold."""
    print(
        f"  >> Выполняю apply_discount(price={price}, discount_tier={discount_tier})"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# ----- Agent Loop -----
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    # llm = init_chat_model("openrouter:deepseek/deepseek-v3.2", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "Ты - полезный агент для шоппинга. "
                "У тебя есть доступ к функциям каталога и скидкам.\n\n "
                "СТРОГИЕ ПРАВИЛА - ты должен придерживаться этих правил\n"
                "1. Никогда не угадывай и не предполагай стоимость "
                "любого продукта. "
                "Ты должен сначала вызвать функцию get_product_price "
                "чтобы получить настоящую цену.\n"
                "2. Вызывай функцию apply_discount только после того "
                "как ты получил цену от get_product_price. Пересчитай "
                "с учетом скидки иммено то число, которое ты получил из "
                "get_product_price, не возвращай просто вымышленное "
                "число.\n"
                "3. Никогда не рассчитывай скидку самостоятельно, "
                "используя математику, всегда используй apply_discount "
                "функцию.\n"
                "4. Если пользователь не указывает уровень скидки, "
                "спроси у него какой уровень скидки он имеет, "
                "не предугадывай уровень скидки.\n"
                "5. Пользователь будет задавать тебе вопросы на русском, а "
                "аргументы для функции тебе нужно подать на английском языке, "
                "Вот тебе небольшой словарь для аргументов: ноутбук - laptop\n"
                "наушники - headphones\n"
                "клавиатура - keyboard\n"
                "золотая скидка - gold\n"
                "серебряная скидка - silver\n"
                "бронзовая скидка - bronze.\n"
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        # Если мы не получили никаких вызовов, значит модель готова вернуть ответ
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Обработка первого вызова функции (get_product_price)
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f" [Выбран инструмент] {tool_name} с аргументами: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found in tools_dict")
        observation = tool_to_use.invoke(tool_args)
        print(f"[Результат от инструмента] {observation}")
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ОШИБКА: достигнуто максимальное число итераций, ответ не получен.")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent(
        "Какая будет цена ноутбука после применения золотой скидки?"
    )
