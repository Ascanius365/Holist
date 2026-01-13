import multiprocessing
import time
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from simple_chalk import chalk
from langchain_openai import ChatOpenAI
import asyncio
import json
import os
from pathlib import Path

from Loader import write_log
from RAG import generate_rag_query, start_rag
from Amygdala import Amygdala


# Tools
class ActionType(str, Enum):
    EAT = "eat"
    DIG = "dig"
    CRAFT = "craft"
    CHAT = "chat"
    DEPOSIT = "putInChest"
    WITHDRAW = "takeFromChest"
    VIEW = "viewChest"
    SMELT = "smeltItem"
    CLEAR = "clearFurnace"
    Go = "goToCoordinates"
    NONE = "none"


class MinecraftAction(BaseModel):
    """The structured format for bot decision-making."""
    action: ActionType = Field(description="Kind of tool.")
    item: Optional[str] = Field(None, description="The item (e.g. 'berries') or none.")
    count: Optional[int] = Field(default=1, description="How many items...")

    x: Optional[float] = Field(None, description="Only required for goToCoordinates.")
    y: Optional[float] = Field(None, description="Only required for goToCoordinates.")
    z: Optional[float] = Field(None, description="Only required for goToCoordinates.")

    reasoning: str = Field(
        description="Description of your reasoning, if you use a tool. "
                    "Also write here a specific overarching goal that should be achieved with the tool use. "
                    "If you use chat, write e.g. 'Hello, what do we want to do?', "
                    "'What tasks do you want to complete?' etc."
    )


class SimpleMemory:
    """A memory system with AI-based summarization."""

    def __init__(self, bot_name, max_messages: int = 5, summarizer_llm=None):
        self.messages: List[Dict[str, str]] = []
        self.max_messages = max_messages
        self.summarizer_llm = summarizer_llm
        self.summary = ""
        self.bot_name = bot_name
        self.persistence_file = f"Memory/{bot_name}_memory.json"
        self.sessions_file = "premem_module/dataset/sessions.jsonl"

        # During initialization, attempt to load old memory.
        self.load_from_disk()


    def load_from_disk(self):
        """Loads memory from disc."""
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.summary = data.get("summary", "")
                    self.messages = data.get("messages", [])
                print(f"💾 Loaded memory for {self.bot_name}.")
            except Exception as e:
                print(f"⚠️ Error while loading: {e}")


    def save_to_disk(self):
        """Saves the memory."""
        data = {"summary": self.summary, "messages": self.messages}
        with open(self.persistence_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


    def add_message(self, input_text: str, output_text: str, observation):
        """Adds an input-output pair to memory."""
        self.messages.append({
            "input": input_text,
            "output": output_text,
            "timestamp": time.time()
        })

        if len(self.messages) >= self.max_messages:
            self._summarize_old_messages(observation)

        self.save_to_disk()


    def _summarize_old_messages(self, observation):
        """Summarizes the oldest news stories using AI."""
        if not self.summarizer_llm:
            print(f"❌ No Summarizer LLM available, delete oldest message")
            self.messages.pop(0)
            return

        try:
            print(f"🔄 Start summarizing... (Messages: {len(self.messages)})")
            old_msgs = self.messages[:5]
            self.messages = self.messages[5:]

            messages_text = "\n".join([
                f"- Observation: {msg['input']}\n  Action: {msg['output']}"
                for msg in old_msgs
            ])

            summary_prompt = (
                "You are a minecraft bot that has been talking and playing minecraft by using commands. "
                "Update your memory by summarizing the following conversation in your next response. "
                "Write the location at the end of your summary. "
                f"NEW INTERACTIONS:\n{messages_text}\n\n"
                "Summary: "
            )

            print(f"📡 Call Summarizer LLM ...")
            response = self.summarizer_llm.invoke(summary_prompt)
            response = response.content if hasattr(response, 'content') else str(response)

            # Schritt B: Suche gezielt nach dem Wort "Summary:"
            if "Summary:" in response:
                # Wir nehmen alles, was NACH "Summary:" kommt
                response = response.split("Summary:")[-1].strip()

            else:
                # Schritt A: Entferne die <think> Blöcke (Gedankengänge)
                import re
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

                # 1. Alles hinter **Reasoning:** abschneiden
                # split() teilt den Text in zwei Teile, [0] nimmt nur den Text davor
                if "**Reasoning:**" in response:
                    response = response.split("**Reasoning:**")[0].strip()
                elif "Reasoning:" in response:  # Falls die Sterne mal fehlen
                    response = response.split("Reasoning:")[0].strip()
                else:
                    response = response.strip()

            # Limit local memory to the NEW summary
            self.summary = response.strip()

            print(f"✅ Summarized memory: {response[:80]}...")
            print(f"📊 After summarizing: {len(self.messages)} current messages + Summary there")

            # Write to sessions.jsonl for PREMem (append mode)
            session_entry = {
                "session_id": f"session_{int(time.time())}",
                "session_date": observation.get("time", ""),
                "text": self.summary
            }

            with open(self.sessions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(session_entry, ensure_ascii=False) + "\n")

            print(f"✅ New summary saved in sessions.jsonl and local memory updated.")

        except Exception as e:
            print(f"❌ Error while summarizing: {type(e).__name__}: {e}")
            print(f"⚠️ Fallback: Delete oldest message instead of summarizing.")
            if self.messages:
                self.messages.pop(0)


    def get_formatted_history(self, log_path) -> str:
        """Creates a readable structure for the system prompt."""
        formatted_parts = []
        final_summary = []

        # 1. Long term memory (summary)
        if self.summary:
            formatted_parts.append("### LONG-TERM MEMORY (Summary of past events)")
            formatted_parts.append(self.summary)
            formatted_parts.append("-" * 40)

            final_summary.append("### LONG-TERM MEMORY (Summary of past events)")
            final_summary.append(self.summary)
            final_summary.append("-" * 40)

        # 2. Short term memory (Recent Interactions)
        if self.messages:
            formatted_parts.append("### SHORT-TERM MEMORY (Most recent interactions)")
            for msg in self.messages:
                # Combination of timestamp, perception and reaction
                entry = (
                    f"Time: {msg['timestamp']}\n"
                    f"Action Result: {msg['output']}\n"
                    f"Observation: {msg['input']}\n"
                )
                formatted_parts.append(entry)

        # If memory is empty
        if not formatted_parts:
            return "No previous interactions recorded."

        # Join together to form a clean block
        final_history = "\n".join(formatted_parts)
        final_summary2 = "\n".join(final_summary)

        write_log(log_path, final_history)

        # Display the debug log nicely in the console
        print(chalk.blue("\n--- CONTEXT SENT TO AI ---"))
        print(final_history)
        print(chalk.blue("--------------------------\n"))

        return final_summary2


# --- PRIVATE QUEUES & PROZESS-VAR ---
_request_queue = None
_response_queue = None

amy = Amygdala()


# --- WORKER PROZESS ---
def agent_worker_process(req_q, res_q, log_path, bot_name):
    """Hintergrundprozess für die API-Kommunikation mit Gedächtnis."""
    print("✅ API Agent gestartet. Warte auf Observations...")

    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    success = load_dotenv(dotenv_path=env_path)

    if not success:
        print(f"❌ Kritisch: .env konnte nicht geladen werden unter {env_path}")

    # The LLM for summarization (Summary)
    summarizer_llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct:free",
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=128,
    )

    # The Memory-Objekt
    memory = SimpleMemory(bot_name=bot_name, max_messages=5, summarizer_llm=summarizer_llm)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    json_schema = MinecraftAction.model_json_schema()

    # DEFINITION OF THE SYSTEM PROMPT
    system_prompt = (
        "You are a bot inside of Minecraft using the Node.js mineflayer library. "
        "You can communicate and interact with the game by using all available commands. "
        f"Your username is '{bot_name}' "
        "Always respond in a concise 1-2 sentence format, followed by a command to execute your action. "
        "Examples for this are provided below. NEVER try using a command that doesn't exist! "
        "If you receive a message from 'System', treat it as an automated event and respond as if you had the thought yourself. "
        "Use it as an opportunity to appear more lively or take initiative."
        f"Output ONLY valid JSON matching this schema: {json_schema}. "
        
        "\n### EXAMPLE:\n"
        "{\n"
        '  "action": "dig",\n'
        '  "item": "oak_log",\n'
        '  "count": 10,\n'
        '  "reasoning": "I need more oak logs to build a shelter. I will gather 10 more to increase my inventory."\n'
        "}\n"

        "{\n"
        '  "action": "smeltItem",\n'
        '  "item": "raw_copper",\n'
        '  "count": 10,\n'
        '  "reasoning": "I see a furnace and I have raw copper and coal in my inventory. I should smelt it to get copper ingots."\n'
        "}\n"

        "{\n"
        '  "action": "goToCoordinates",\n'
        '  "x": 125.5,\n'
        '  "y": 64,\n'
        '  "z": -300.2,\n'
        '  "reasoning": "I am heading back to the coordinates where I previously found a large coal vein to continue mining."\n'
        "}\n"
    )

    while True:
        try:
            observation = req_q.get()
            if observation is None: break

            # Loading history
            history_text = memory.get_formatted_history(log_path)

            # Generate Query
            rag_query = generate_rag_query(observation, memory.messages, history_text)
            print("rag_query: " + rag_query)

            # Call RAG
            rag_context = start_rag(rag_query, observation)

            # Call Amygdala
            warning = amy.inject_to_prompt()

            # EXTEND SYSTEM PROMPT WITH HISTORY
            system_prompt_with_history = system_prompt + "\n\n" + rag_context + "\n\n" + history_text

            messages = [{"role": "system", "content": system_prompt_with_history}]

            # ADD ALL MEMORY ENTRIES AS INDIVIDUAL MESSAGES (for better display)
            for msg_entry in memory.messages:
                messages.append({
                    "role": "system",
                    "content": f"Past observation: {msg_entry['input']}"
                })
                messages.append({
                    "role": "user",
                    "content": f"Past action: {msg_entry['output']}"
                })

            # CURRENT OBSERVATION
            messages.append({
                "role": "user",
                "content": json.dumps(observation, ensure_ascii=False)
            })

            # Append warnings from Amygdala
            if warning:
                messages.append({
                    "role": "user",
                    "content": f"{warning}"
                })

            print(chalk.green(json.dumps(messages, indent=2, ensure_ascii=False)))

            # In Agent.py vor dem client.chat.completions.create Call:
            print(f"DEBUG: Anzahl der Nachrichten: {len(messages)}")
            for i, m in enumerate(messages):
                print(f"DEBUG: Message {i} ({m['role']}): {m['content'][:100]}...")

            if len(messages) < 2 or len(messages[1]['content']) < 10:
                print("❌ FEHLER: Der Prompt ist zu kurz! Abbruch vor API-Call.")
                return None

            response = client.chat.completions.create(
                model="qwen/qwen3-235b-a22b-2507",
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )

            print(response)

            raw_text = response.choices[0].message.content.strip()
            raw_json_string = extract_single_json(raw_text)

            print(f"DEBUG Cleaned JSON string: [{raw_json_string}]")

            if not raw_json_string:
                raise ValueError("Could not extract a valid, isolated JSON object from the LLM response.")

            action_data = ""
            if raw_json_string:
                # 3. Validierung der Antwort
                action_data = MinecraftAction.model_validate_json(raw_json_string)

            # Accessing the first element of the list
            final_feedback = ""
            if observation:
                tool_feedback = observation.get("Tool feedback", "")
                if isinstance(tool_feedback, str):
                    final_feedback = tool_feedback.strip()
                else:
                    final_feedback = json.dumps(tool_feedback, ensure_ascii=False)
                print(f"💬 Found chat message: {bool(final_feedback)} | Length: "
                      f"{len(final_feedback) if final_feedback else 0}")
                if final_feedback:
                    print(f"💬 Chat: {final_feedback[:100]}...")
            final_feedback += observation.get("Current position", "")

            try:
                action_data = MinecraftAction.model_validate_json(raw_json_string)
                data = action_data.model_dump()

                print(f"✅ Determined tool: {action_data.action}")
                print(f"📝 Reasoning: {action_data.reasoning[:80]}...")
                print(f"🔢 Count: {action_data.count}")

                amy.analyze_situation(data, observation)

                # Memory aktualisieren - nur Chat-Message speichern
                print(f"⏳ Before add_message: {len(memory.messages)} entrys")
                memory.add_message(
                    input_text=final_feedback,
                    output_text=action_data.reasoning,
                    observation=observation,
                )
                print(
                    f"🧠 Memory updated. Current entries: {len(memory.messages)} | Summary available: {bool(memory.summary)}")

                # Daten in die Antwort-Queue legen
                res_q.put(data)
                print("📤 Queue entry confirmed.")

            except Exception as val_e:
                print(f"⚠️ Validation failed: {val_e}")
                res_q.put({"action": "none", "item": None, "reasoning": "Validation failed."})

        except Exception as api_e:
            print(f"❌ API failure: {api_e}")
            res_q.put({"action": "none", "item": None, "reasoning": f"API error: {str(api_e)}"})


def extract_single_json(raw_text: str) -> Optional[str]:
    """Extracts the first valid JSON string from a string containing noise."""

    # Remove initial and final Markdown code blocks
    if raw_text.startswith("```"):
        try:
            raw_text = raw_text.split("```")[1].replace("json", "").strip()
        except IndexError:
            pass

    # Removal of invisible characters
    raw_text = raw_text.replace('\xa0', ' ').replace('\u200b', '').strip()

    # Find first '{'
    start = -1
    try:
        start = raw_text.index('{')
    except ValueError:
        return None  # No JSON found

    # Parenthesis counting logic to find the end of the first complete JSON object
    balance = 0
    end = -1
    for i in range(start, len(raw_text)):
        char = raw_text[i]
        if char == '{':
            balance += 1
        elif char == '}':
            balance -= 1

        if balance == 0:
            end = i + 1
            break

    if end != -1:
        # Extract only the first complete JSON object
        json_string = raw_text[start:end].strip()

        # Final sanity check: Is it a valid JSON file?
        try:
            json.loads(json_string)
            return json_string
        except json.JSONDecodeError:
            pass

    return None  # Could not isolate valid JSON


# --- PUBLIC INTERFACE FOR MAIN.PY ---

def start_agent_process(req_q, res_q, bot_name):
    """Create a new process and return it."""

    log_path = f"logs/{bot_name}_ai.log"

    process = multiprocessing.Process(
        target=agent_worker_process,
        args=(req_q, res_q, log_path, bot_name)
    )
    process.start()
    print(f"🚀 New LLM worker launched (PID: {process.pid})")
    return process


async def fetch_agent_response(res_q):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, res_q.get)


def run_agent_async(req_q, obs):
    """Send new observation."""
    req_q.put(obs)


def stop_agent_worker(req_q):
    """It ends the process cleanly."""
    global _agent_process
    if _agent_process and _agent_process.is_alive():
        req_q.put(None)
        _agent_process.join()


def init_queues():
    """Creates and returns the queues to be used in Main.py."""
    return multiprocessing.Queue(), multiprocessing.Queue()


__all__ = ['run_agent_async', 'fetch_agent_response', 'start_agent_process', 'stop_agent_worker', 'init_queues']