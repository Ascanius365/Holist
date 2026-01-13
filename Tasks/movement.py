from utils.vec3_conversion import vec3_to_str
from simple_chalk import chalk
from javascript import require
vec3 = require("vec3")
import asyncio
goals = require("@miner-org/mineflayer-baritone").goals


# Mineflayer: Pathfind to goal
async def pathfind_to_goal(bot, mineflayer_pathfinder, block, item):

    try:
        # Get the block position
        pos = block.position

        block_location = vec3(
            pos.x, pos.y + 1, pos.z
        )
        goal = goals.GoalNear(block_location, 3)

        # Check the distance
        dist = bot.entity.position.distanceTo(block.position)
        if dist > 3:
            if block_location:
                print(chalk.magenta(f"Laufe zu {item} bei {vec3_to_str(block_location)}"))
                try:
                    # Go to block
                    await bot.ashfinder.goto(goal, timeout=60)
                except Exception as e:
                    print(chalk.yellow(f"⚠️ Fehler beim Pathfinding: {e}"))
        else:
            print(f"✅ Goal {item} already reached.")

        await asyncio.sleep(0.5)

    except Exception as e:
        print(f"Error while trying to run pathfind_to_goal: {e}")


async def goToCoordinates(bot, mineflayer_pathfinder, x, y, z):

    try:
        goal_location = vec3(
            x, y + 1, z
        )

        goal = goals.GoalNear(goal_location, 3)

        # Check the distance
        current_pos = bot.entity.position
        dist = current_pos.distanceTo(goal_location)

        if dist <= 100:
            if dist > 3:
                if goal_location:
                    print(chalk.magenta(f"Walk to {vec3_to_str(goal_location)}"))
                    try:
                        # Go to block
                        await bot.ashfinder.goto(goal, timeout=60)
                    except Exception as e:
                        print(chalk.yellow(f"⚠️ Fehler beim Pathfinding: {e}"))
                        msg = f"You did not reach your goal 'x: {x}, y: {y}, z: {z}'!"
                        return msg

                    msg = f"You have successfully reached your goal 'x: {x}, y: {y}, z: {z}'!"
                    print(msg)
                    return msg

            else:
                msg = f"Goal 'x: {x}, y: {y}, z: {z}' already reached."
                print(msg)
                return msg
        else:
            msg = f"Goal 'x: {x}, y: {y}, z: {z}' is too far away ({dist} blocks)."
            print(msg)
            return msg

        await asyncio.sleep(0.5)

    except Exception as e:
        msg = f"Error while trying to run pathfind_to_goal: {e}"
        print(msg)
        return msg


async def collect_drops(bot, mcData, item):
    """Search and collect all items in the vicinity."""

    while True:
        # 1. Liste aller Item-Entities im Umkreis holen
        drops = []
        for entity_id in bot.entities:
            entity = bot.entities[entity_id]
            if entity.name == 'item':
                dist = bot.entity.position.distanceTo(entity.position)
                if dist <= 20:
                    drops.append((entity, dist))

        # Wenn keine Drops mehr da sind, abbrechen
        if not drops:
            break

        # 2. Sortieren nach Distanz (immer zum nächsten Item laufen)
        drops.sort(key=lambda x: x[1])
        target_item, distance = drops[0]

        print(f"📦 Sammle {target_item.name} auf ({distance:.1f}m entfernt)...")

        try:
            # 3. Baritone zum Item schicken
            goal = goals.GoalExact(target_item.position)
            await bot.ashfinder.goto(goal, timeout=60)

            # Kurze Pause, damit Mineflayer das Inventar-Update registriert
            await asyncio.sleep(0.5)

        except Exception as e:
            print(f"⚠️ Fehler beim Aufheben von {target_item.name}: {e}")
            break  # Bei schwerem Fehler (z.B. unerreichbar) abbrechen
