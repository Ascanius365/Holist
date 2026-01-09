from javascript import require, On, off
from simple_chalk import chalk
from Agent import run_agent_async, fetch_agent_response, start_agent_process, init_queues
from Tasks.observation import observe
from Tasks.needs import eat
from Tasks.mine import dig
from Tasks.craft import craft
from Tasks.inventory import Chest, smeltItem, clearFurnace
from Tasks.movement import goToBase
import os
from RAG import update_long_term_memory
import asyncio


# Import the javascript libraries
mineflayer = require("mineflayer")
mineflayer_pathfinder = require("mineflayer-pathfinder")
BuilderModule = require('mineflayer-builder')
vec3 = require("vec3")

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Global bot parameters
server_host = "localhost"
server_port = 3000
reconnect = True
version = "1.21.1"

import threading

# Global lock to ensure that only one update is running at a time.
ltm_update_lock = threading.Lock()
ltm_update_in_progress = False


async def periodic_ltm_update():
    """Calls update_long_term_memory(), but prevents overlaps."""
    global ltm_update_in_progress

    while True:
        try:
            await asyncio.sleep(30)

            # Check if an update is already running.
            if ltm_update_in_progress:
                print("⏳ [LTM] Update is still in progress, skip this cycle....")
                continue

            # Check if sessions.jsonl has any new entries.
            sessions_file = "premem_module/dataset/sessions.jsonl"
            if not os.path.exists(sessions_file):
                print("⚠️ [LTM] sessions.jsonl not found")
                continue

            # Start the update in the background thread
            print("🔄 [LTM] Update started...")
            thread = threading.Thread(target=_run_ltm_update_safe, daemon=True)
            thread.start()

            await asyncio.sleep(30)

        except Exception as e:
            print(f"❌ Error in periodic_ltm_update: {e}")


def _run_ltm_update_safe():
    """Wrapper for update_long_term_memory() with lock."""
    global ltm_update_in_progress

    with ltm_update_lock:
        try:
            ltm_update_in_progress = True
            print("⏱️ [LTM] Update started (Lock acquired)")

            update_long_term_memory()

            print("✅ [LTM] Update completed successfully")
        except Exception as e:
            print(f"❌ [LTM] Update failed: {type(e).__name__}: {str(e)[:100]}")
        finally:
            ltm_update_in_progress = False
            print("🔓 [LTM] Lock released")


class MCBot:
    def __init__(self, bot_name):
        self.bot_name = bot_name
        # Individual queues and processes for this bot
        self.request_queue, self.response_queue = init_queues()
        self.agent_process = start_agent_process(self.request_queue, self.response_queue, bot_name)

        self.mineflayer_pathfinder = mineflayer_pathfinder

        self.bot_args = {
            "username": bot_name,
            "host": server_host,
            "port": server_port,
            "version": version,
        }
        self.agent_is_busy = False
        self.chat_history_buffer = []
        self.chat_history = [(f"   - Chat history: ")]
        self.event_history = []
        self.start_bot()

    def start_bot(self):
        self.bot = mineflayer.createBot(self.bot_args)
        self.bot.loadPlugin(self.mineflayer_pathfinder.pathfinder)

        # Pathfinder settings
        self.bot.pathfinder.thinkTimeout = 2000  # Max 5 Sek. nachdenken
        self.bot.pathfinder.tickTimeout = 100  # Zeit pro Tick

        self.mcData = require('minecraft-data')(self.bot.version)

        self.start_events()


    # Tags bot username before console messages
    def log(self, message):
        print(f"[{self.bot.username}] {message}")


    async def main_agent_loop(self):

        while True:
            try:
                # Gather Observations
                observation = observe(self.bot, self.chat_history_buffer, self.chat_history, self.event_history)

                # Send to YOUR OWN queue
                run_agent_async(self.request_queue, observation)
                self.chat_history_buffer.clear()

                response = await fetch_agent_response(self.response_queue)

                if response and "action" in response:

                    if len(self.event_history) > 0:
                        self.event_history.pop(0)

                    action = response["action"]
                    item = response.get("item")
                    reason = response.get("reasoning")
                    count = response.get("count")

                    if action == "chat":
                        self.bot.chat(reason)

                    elif action == "eat":
                        msg = await eat(self.bot, self.mcData, item)
                        self.chat_history_buffer.append(msg)

                    elif action == "putInChest":
                        msg = await self.my_chest.depositOneToChest(self.mineflayer_pathfinder, item, count)
                        self.chat_history_buffer.append(msg)

                    elif action == "takeFromChest":
                        msg = await self.my_chest.withdrawOneFromChest(self.mineflayer_pathfinder, item, count)
                        self.chat_history_buffer.append(msg)

                    elif action == "viewChest":
                        msg = await self.my_chest.viewChest(mineflayer_pathfinder)
                        self.chat_history_buffer.append(msg)

                    elif action == "dig":
                        msg = await dig(self.bot, self.mcData, self.mineflayer_pathfinder, item)
                        self.chat_history_buffer.append(msg)

                    elif action == "craft":
                        msg = await craft(self.bot, self.mcData, self.mineflayer_pathfinder, item, count)
                        self.chat_history_buffer.append(msg)

                    elif action == "smeltItem":
                        msg = await smeltItem(self.bot, self.mcData, self.mineflayer_pathfinder, item, count)
                        self.chat_history_buffer.append(msg)

                    elif action == "clearFurnace":
                        msg = await clearFurnace(self.bot, self.mcData, self.mineflayer_pathfinder, item, count)
                        self.chat_history_buffer.append(msg)

                    elif action == "goToBase":
                        msg = await goToBase(self.bot, self.mineflayer_pathfinder)
                        self.chat_history_buffer.append(msg)

            except Exception as e:
                print(f"Fehler im Loop: {e}")


    # Attach mineflayer events to bot
    def start_events(self):

        # Login event (Logged in)
        @On(self.bot, "login")
        def login(this):
            self.bot_socket = self.bot._client.socket
            self.log(chalk.green(
                f"Logged in to {self.bot_socket.server if self.bot_socket.server else self.bot_socket._host}"
            ))


        @On(self.bot, "spawn")
        def spawn(this):
            # Sicherstellen, dass der Prozess startet
            self.log(chalk.green("Bot gestartet!"))

            self.my_chest = Chest(self, self.mcData, chesttype="Chest")

            if self.my_chest.object:
                print(f"Truhe gefunden bei: {self.my_chest.object.position}")

            # We start the asynchronous main loop
            asyncio.run_coroutine_threadsafe(self.main_agent_loop(), loop)


        # Kicked event (Got kicked from server)
        @On(self.bot, "kicked")
        def kicked(this, reason, loggedIn):
            if loggedIn:
                self.log(chalk.red(f"Kicked whilst trying to connect: {reason}"))


        # Chat event: Triggers on chat message
        @On(self.bot, "messagestr")
        def messagestr(this, message, messagePosition, jsonMsg, sender, verified=None):
            if not sender:
                return
            # 1. NACHRICHT ZUM PUFFER HINZUFÜGEN
            self.chat_history.append(message)
            if len(self.chat_history) > 10:
                self.chat_history.pop(0)
            print("message: " + str(self.chat_history))


        @On(self.bot, "death")
        def death(this):
            self.event_history.append("death")


        # End event (Disconnected from server)
        @On(self.bot, "end")
        def end(this, reason):
            self.log(chalk.red(f"Disconnected: {reason}"))

            # Turn off event listeners
            off(self.bot, "login", login)
            off(self.bot, "kicked", kicked)
            off(self.bot, "end", end)
            off(self.bot, "messagestr", messagestr)


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method('spawn', force=True)

    # Start the periodic LTM update in the background.
    asyncio.run_coroutine_threadsafe(periodic_ltm_update(), loop)

    # Create the bots
    # bot = MCBot("bot-1")
    # bot_2 = MCBot("bot-2")
    # bot_3 = MCBot("bot-3")
    # bot_4 = MCBot("bot-4")
    bot_5 = MCBot("bot-5")

    try:
        # This keeps Python awake and processes all tasks in the loop.
        loop.run_forever()
    except KeyboardInterrupt:
        print("Bot wird beendet...")
