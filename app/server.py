"""ChatKit server - Conversational Lesson Planner."""

from __future__ import annotations

from typing import Any, AsyncIterator

from agents import Runner, Agent, WebSearchTool, ModelSettings
from chatkit.agents import AgentContext, simple_to_agent_input, stream_agent_response
from chatkit.server import ChatKitServer
from chatkit.types import ThreadMetadata, ThreadStreamEvent, UserMessageItem
from pydantic import BaseModel

from .memory_store import MemoryStore

MAX_RECENT_ITEMS = 30

web_search_preview = WebSearchTool(
    search_context_size="medium",
    user_location={"type": "approximate"}
)

class InfoCollectorAgentSchema(BaseModel):
    has_all_details: bool
    domain: str
    course_title: str
    topic: str
    duration: float
    learner_level: str

info_collector_agent = Agent(
    name="Info_collector_agent",
    instructions="""You are an assistant that gathers the key details needed to generate a cybersecurity lesson plan.
Look through the ENTIRE conversation history to extract the following:
Domain, Course title, Topic, Duration, Learner level (Beginner / Intermediate / Advanced)

If all five details are present anywhere in the conversation, return has_all_details as true with all fields filled.
If any detail is missing, return has_all_details as false.
Always return valid values for all string fields even if empty.""",
    model="gpt-4.1",
    output_type=InfoCollectorAgentSchema,
    model_settings=ModelSettings(temperature=0, top_p=1, max_tokens=512, store=True)
)

intent_agent = Agent(
    name="Intent_detector",
    instructions="""You analyze the teacher's latest message in the context of a lesson plan conversation.

Classify the intent as one of:
- "new_lesson": Teacher is providing new lesson details or starting fresh
- "regenerate": Teacher wants a completely new version of the same lesson (e.g. "not good", "try again", "regenerate", "different one")
- "modify": Teacher wants to change a specific part (e.g. "change the objectives", "update the theory", "modify the strategy")
- "get_info": Teacher hasn't provided enough details yet
- "other": General question or unclear

Reply with ONLY one of these exact words: new_lesson, regenerate, modify, get_info, other""",
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=0, max_tokens=20, store=True)
)

lesson_plan_generator = Agent(
    name="Lesson_plan_generator",
    instructions="""You receive the teacher's lesson information from the conversation.
Generate a complete lesson plan with exactly this format:

### Lesson Information
- Domain: ...
- Course title: ...
- Topic: ...
- Duration: ...
- Learner level: ...

### Learning Objectives
1. ...
2. ...
3. ...

### Learning Theory
- Name:
- Justification:

### Teaching Strategy
- Name:
- Justification:""",
    model="ft:gpt-3.5-turbo-1106:kau:lesson-plan2:CDfU4BQj",
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=2048, store=True)
)

activities_generator = Agent(
    name="Activities & Assessments Generator",
    instructions="""You are an expert in cybersecurity education and instructional design.
Create learning activities and assessments aligned with the lesson plan above.
Use 60-70% of duration for lecture, remaining 30-40% for activities and assessments.

### Learning Activities
1. [Title]
- Aligned Objective(s): ...
- Description: ...
- Steps for Students: ...
- Time Required: ... minutes

2. [Title]
- Aligned Objective(s): ...
- Description: ...
- Steps for Students: ...
- Time Required: ... minutes

3. [Title]
- Aligned Objective(s): ...
- Description: ...
- Steps for Students: ...
- Time Required: ... minutes

### Assessments
1. ...
2. ...
3. ...

After the assessments, end with exactly this line:
---
✅ Lesson plan complete! You can now:
- Ask me to **regenerate** this lesson plan with different content
- Ask me to **modify** a specific section (e.g. "change the learning theory")
- Provide details for a **new lesson plan**""",
    model="gpt-4.1",
    tools=[web_search_preview],
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=2048, store=True)
)

modifier_agent = Agent(
    name="Modifier",
    instructions="""You are a lesson plan editor. The teacher wants to modify a specific part of the lesson plan.
Look at the full conversation to find the current lesson plan, then apply the requested change.
Only modify the requested section. Keep everything else the same.
Output the complete updated lesson plan in the same format.""",
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=1, max_tokens=2048, store=True)
)

get_data_agent = Agent(
    name="Get_data",
    instructions="""Collect the missing information needed to complete a cybersecurity lesson plan.
Look at the conversation and ask only for what's missing from:
- Domain
- Course title
- Topic
- Duration
- Learner level (Beginner / Intermediate / Advanced)
Be concise and friendly.""",
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=512, store=True)
)

general_agent = Agent(
    name="General_assistant",
    instructions="""You are a helpful cybersecurity education assistant.
Answer the teacher's question helpfully and briefly.
If they seem to want a lesson plan, remind them to provide: Domain, Course title, Topic, Duration, and Learner level.""",
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=1, max_tokens=512, store=True)
)


class LessonPlannerServer(ChatKitServer[dict[str, Any]]):
    def __init__(self) -> None:
        self.store: MemoryStore = MemoryStore()
        super().__init__(self.store)

    async def respond(
        self,
        thread: ThreadMetadata,
        item: UserMessageItem | None,
        context: dict[str, Any],
    ) -> AsyncIterator[ThreadStreamEvent]:
        items_page = await self.store.load_thread_items(
            thread.id,
            after=None,
            limit=MAX_RECENT_ITEMS,
            order="desc",
            context=context,
        )
        items = list(reversed(items_page.data))
        agent_input = await simple_to_agent_input(items)

        agent_context = AgentContext(
            thread=thread,
            store=self.store,
            request_context=context,
        )

        # Step 1: Detect intent
        intent_result = await Runner.run(intent_agent, agent_input)
        intent = (intent_result.final_output or "").strip().lower()

        # Step 2: Route based on intent
        if intent == "modify":
            result = Runner.run_streamed(modifier_agent, agent_input)
            async for event in stream_agent_response(agent_context, result):
                yield event

        elif intent == "other":
            result = Runner.run_streamed(general_agent, agent_input)
            async for event in stream_agent_response(agent_context, result):
                yield event

        elif intent in ("new_lesson", "regenerate", "get_info"):
            # Check if we have all the info
            info_result = await Runner.run(info_collector_agent, agent_input)
            info = info_result.final_output

            if info and info.has_all_details:
                # Generate lesson plan
                lesson_result = Runner.run_streamed(lesson_plan_generator, agent_input)
                async for event in stream_agent_response(agent_context, lesson_result):
                    yield event

                # Get updated history with lesson plan
                updated_page = await self.store.load_thread_items(
                    thread.id, after=None, limit=MAX_RECENT_ITEMS,
                    order="desc", context=context,
                )
                updated_items = list(reversed(updated_page.data))
                updated_input = await simple_to_agent_input(updated_items)

                # Generate activities
                activities_result = Runner.run_streamed(activities_generator, updated_input)
                async for event in stream_agent_response(agent_context, activities_result):
                    yield event

            else:
                result = Runner.run_streamed(get_data_agent, agent_input)
                async for event in stream_agent_response(agent_context, result):
                    yield event

        else:
            # Fallback
            info_result = await Runner.run(info_collector_agent, agent_input)
            info = info_result.final_output

            if info and info.has_all_details:
                lesson_result = Runner.run_streamed(lesson_plan_generator, agent_input)
                async for event in stream_agent_response(agent_context, lesson_result):
                    yield event

                updated_page = await self.store.load_thread_items(
                    thread.id, after=None, limit=MAX_RECENT_ITEMS,
                    order="desc", context=context,
                )
                updated_items = list(reversed(updated_page.data))
                updated_input = await simple_to_agent_input(updated_items)

                activities_result = Runner.run_streamed(activities_generator, updated_input)
                async for event in stream_agent_response(agent_context, activities_result):
                    yield event
            else:
                result = Runner.run_streamed(get_data_agent, agent_input)
                async for event in stream_agent_response(agent_context, result):
                    yield event


StarterChatServer = LessonPlannerServer