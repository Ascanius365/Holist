from javascript import require
vec3 = require("vec3")
from datetime import datetime, timedelta

top_blocks = 30


def observe(bot, mcData, chat_history_buffer, chat_history, event_history):
    observation_data = {}

    # Append entities
    entities = bot.entities

    nearby_players = []

    # Go through all entities
    for entity_id in entities:
        entity = entities[entity_id]

        # Check if not self
        if entity and hasattr(entity, 'username') and entity.username:
            if entity.username != bot.username:
                nearby_players.append(entity.username)

    if nearby_players:
        names_list = ", ".join(nearby_players)
        observation_data["entities"] = (f"   - Nearby players/bots: {names_list}.")


    # Append tool feedback
    clean_chat = [str(msg) for msg in chat_history_buffer if msg is not None]
    if clean_chat:
        observation_data["Tool feedback"] = ("   - Recent tool feedback: " + " | ".join(clean_chat))

    print("observation_data: " + str(observation_data))

    # Append chat history
    observation_data["Chat history"] = chat_history

    # Hunger status
    if bot.food <= 18:
        observation_data["food"] = (f"   - Food: {bot.food}/20 (If < 18, you won't heal. Eat something!)")

    # Inventory
    items = bot.inventory.items()
    if items:
        items = bot.inventory.items()
        inventory_list = [f"{i.count}x {i.name}" for i in items] if items else []
        observation_data["inventory"] = (f"\n2. YOUR INVENTORY (Items you carry): {inventory_list}")

    # Scan nearby blocks
    """Scannt jeden Block in einem Kubus um den Bot."""
    x_radius = 16
    y_radius = 2
    z_radius = 16
    block_stats = {}

    # Position of the bot
    pos = bot.entity.position

    # Loop through X, Y and Z coordinates
    for x in range(int(pos.x) - x_radius, int(pos.x) + x_radius + 1):
        for y in range(int(pos.y) - y_radius, int(pos.y) + y_radius + 1):
            for z in range(int(pos.z) - z_radius, int(pos.z) + z_radius + 1):
                block = bot.blockAt(vec3(x, y, z))

                # Ignore air
                if block and block.name != 'air':
                    if block.name in block_stats:
                        block_stats[block.name] += 1
                    else:
                        block_stats[block.name] = 1

    if block_stats:
        top_blocks = sorted(block_stats.items(), key=lambda item: item[1], reverse=True)[:30]
        blocks_str = ", ".join([f"{count}x {name}" for name, count in top_blocks])
        observation_data["surroundings"] = (f"   - Visible blocks in the environment, not in your inventory: {blocks_str}")

    else:
        observation_data["surroundings"] = (f"   - You don't see any interesting blocks nearby.")


    # Add Minecraft Time
    current_time = get_minecraft_time(bot)
    observation_data["time"] = f"{current_time}"
    print(observation_data["time"])


    # Append event history
    observation_data["Event history"] = event_history

    # Append the position
    current_pos = bot.entity.position
    block = bot.blockAt(current_pos)
    biomeId = block.biome.id
    biome_name = mcData.biomes[biomeId].name

    observation_data["Current position"] = (f"You are at the coordinates {current_pos} "
                                            f"in biome '{biome_name}'")
    print(observation_data["Current position"])

    return observation_data


def get_minecraft_time(bot):
    """
    Translates Minecraft ticks into a readable time and date.

    time_of_day (int): Ticks in the current day (0-23999)
    total_world_age (int): Total ticks in the world
    """

    time_of_day = bot.time.timeOfDay
    day_count = bot.time.day + 1

    # 1. Uhrzeit berechnen (Start um 06:00 Uhr bei 0 Ticks)
    hours = int((time_of_day / 1000 + 6) % 24)
    minutes = int((time_of_day % 1000) * 60 / 1000)
    time_string = f"{hours:02d}:{minutes:02d}"

    # 3. Wochentag simulieren
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = weekdays[(day_count - 1) % 7]

    # 4. Datum simulieren (Start am 01.01.2026)
    start_date = datetime(2026, 1, 1)
    current_date = start_date + timedelta(days=day_count - 1)
    date_string = current_date.strftime("%Y-%m-%d")

    # Ergebnis im Format: "2026-01-07 Wednesday 16:58:00"
    full_format = f"{date_string} {weekday} {time_string}:00"

    return full_format

