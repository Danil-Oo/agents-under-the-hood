# Agents Under the Hood

Три реализации ReAct-цикла агента с нарастающей сложностью — от высокоуровневого использования LangChain до полностью ручной реализации на чистом промпте.

## Зачем это нужно

Понять, как работают AI-агенты «под капотом»: как LLM вызывает инструменты, как парсит ответы и как организуется цикл мышления → действия → наблюдения.

## Подходы

### 1. LangChain Tool Calling (`1_agent_loop_langchain_tool_calling.py`)

Максимально простой вариант с использованием LangChain:
- Декоратор `@tool` для описания инструментов
- `llm.bind_tools()` для автоматической привязки
- Автоматический парсинг вызовов функций

```bash
python 1_agent_loop_langchain_tool_calling.py
```

### 2. Raw Ollama API (`2_agent_loop_langchain_tool_calling.py`)

Средний уровень — без абстракций LangChain для инструментов:
- JSON-схемы инструментов описываются вручную
- Вызов через `ollama.chat()` напрямую
- Ручная обработка `tool_calls` из ответа

```bash
python 2_agent_loop_langchain_tool_calling.py
```

### 3. Raw ReAct Prompt (`3_raw_react_prompt.py`)

Низкоуровневая реализация — весь агентный цикл через промпт:
- Формат Thought → Action → Action Input → Observation
- Парсинг ответа LLM через regex
- Scratchpad вместо истории сообщений
- Стоп-токен для остановки генерации перед Observation

```bash
python 3_raw_react_prompt.py
```

## Стек технологий

- **Python 3.10+**
- **Ollama** (qwen3:1.7B) — локальная LLM
- **LangChain** — для первого варианта
- **LangSmith** — трассировка вызовов

## Установка

```bash
# Установить зависимости
uv sync

# Убедиться, что Ollama запущен и модель скачана
ollama pull qwen3:1.7B
```

## Пример работы

Вопрос: *«Какая будет цена ноутбука после применения золотой скидки?»*

Агент:
1. Вызывает `get_product_price("laptop")` → 1299.99
2. Вызывает `apply_discount(1299.99, "gold")` → 1000.99
3. Возвращает итоговую цену
