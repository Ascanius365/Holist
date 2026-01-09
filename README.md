# Holist
Mineflayer Bot with LLM and PREMem Memory

Holist, like Mindcraft, is a bot that uses an agent to call tools, which are then executed in Mineflayer. However, Mindcraft cannot perform reflections like those used in the Stanford study, which is what I implemented. To address this, I added PREMem as a new module. PREMem is a new memory that can cluster long-term memory in the vector database, thereby generating more abstract entries. All entries emerge organically through clustering, and there are no predefined rules.

https://arxiv.org/abs/2509.10852

An important change is that with each call, retrieval now delivers not just individual memory fragments, but a multi-layered representation of the world:

1. Macro level (The world): In which village does the bot live? What are the characteristics of the neighborhood (market, residential area)?
2. Meso level (Social role): Where is its workplace (workshop)? What is its job there?
3. Micro-level (habits): What are its typical morning routines? What is the current status of its tools?

The unique aspect is the dynamic management of the RAG process. The text above is not a static prompt, and the entries are not completely different every time.

- Marketplace example: If the bot wants to buy something, the system hides irrelevant morning routines and instead provides information about the vendors and their stalls, the rest of the text above remains the same.

Here is an example of the entries in memory:

Awareness

- [current environment]: All activities take place in a Minecraft world
- [awareness response]: Group transitions from unaware to aware of the warning
- [repetitive activity frustration]: Bot expresses frustration with repetitive activities multiple times
- [seeking variety]: Bot aims to break cycles by switching to different activities
- [project scope]: Oak log requirement suggests large-scale construction
- [construction planning]: Oak log accumulation implies shelter/structure construction

Development and identity

- [potential activity]: The bot switched from tool crafter to miner
- [base-building proficiency]: Scout improved base-building skills over time
- [base-building experience]: Scout has prior base-building experience
- [resource strategy shift]: Transition from opportunistic gathering to systematic collection
- [operational shift]: Bot evolving from passive gatherer to strategic planner

Recognizing patterns over long periods

- [base construction frequency]: Base construction achieved multiple times within a week
- [resource gathering repetition]: Bot repeatedly gathers 8 oak logs in 50 minutes

Strategies and problem solving

- [resource specialization]: Bot focuses on oak log acquisition for structural projects
- [resource dependency]: Oak logs are critical for base-building operations
- [oak log gathering]: All consistently gathers oak logs towards goal
- [crafting motivation]: All aims to improve mining efficiency with diamond pickaxe
