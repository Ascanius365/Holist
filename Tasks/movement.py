from utils.vec3_conversion import vec3_to_str
from simple_chalk import chalk
from javascript import require
vec3 = require("vec3")
import asyncio


# Mineflayer: Pathfind to goal
async def pathfind_to_goal(bot, mineflayer_pathfinder, block, item):

    try:
        # Get the block position
        pos = block.position

        block_location = vec3(
            pos.x, pos.y + 1, pos.z
        )

        # Check the distance
        dist = bot.entity.position.distanceTo(block.position)
        if dist > 3:
            if block_location:
                print(chalk.magenta(f"Laufe zu {item} bei {vec3_to_str(block_location)}"))

                # Go to block
                bot.pathfinder.setGoal(mineflayer_pathfinder.pathfinder.goals.GoalNear(
                    block_location.x, block_location.y, block_location.z, 1
                ))

                # Wait for movement to complete (with timeout)
                timeout = 30
                elapsed = 0
                while bot.pathfinder.isMoving() and elapsed < timeout:
                    await asyncio.sleep(0.2)
                    elapsed += 0.2

                if elapsed >= timeout:
                    print(chalk.yellow(f"Movement timeout after {timeout}s, continuing anyway..."))

        else:
            print(f"✅ Goal {item} already reached.")

        await asyncio.sleep(0.5)

    except Exception as e:
        print(f"Error while trying to run pathfind_to_goal: {e}")


async def goToBase(bot, mineflayer_pathfinder):

    try:
        location_name = "Base"

        base_x = 25
        base_y = 72
        base_z = 0

        goal_location = vec3(
            base_x, base_y + 1, base_z
        )

        # Check the distance
        current_pos = bot.entity.position
        dist = current_pos.distanceTo(goal_location)

        if dist > 3:
            if goal_location:
                print(chalk.magenta(f"Walk to {location_name} at {vec3_to_str(goal_location)}"))

                # 1. Go to block
                bot.pathfinder.setGoal(mineflayer_pathfinder.pathfinder.goals.GoalNear(
                    goal_location.x, goal_location.y, goal_location.z, 1
                ))

                # Wait for movement to complete (with timeout)
                timeout = 30
                elapsed = 0
                while bot.pathfinder.isMoving() and elapsed < timeout:
                    await asyncio.sleep(0.2)
                    elapsed += 0.2

                if elapsed >= timeout:
                    print(chalk.yellow(f"Movement timeout after {timeout}s, continuing anyway..."))
                    msg = f"You did not reach your goal '{location_name}'!"
                    print(msg)
                    return msg

                else:
                    msg = f"You have successfully reached your goal '{location_name}'."
                    print(msg)
                    return msg

        else:
            msg = f"Goal {location_name} already reached."
            print(msg)
            return msg

        await asyncio.sleep(0.5)

    except Exception as e:
        msg = f"Error while trying to run pathfind_to_goal: {e}"
        print(msg)
        return msg