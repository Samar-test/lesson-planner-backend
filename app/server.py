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

class IntentSchema(BaseModel):
    intent: str
    changed_element: str

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
- "modify": Teacher wants to change a specific part of the lesson plan
- "get_info": Teacher hasn't provided enough details yet
- "other": General question or unclear

If intent is "modify", also identify what changed in changed_element as one of:
topic, learner_level, duration, objectives, learning_theory, teaching_strategy, activities, assessments

Reply with ONLY valid JSON like: {"intent": "modify", "changed_element": "objectives"}
or {"intent": "new_lesson", "changed_element": ""}""",
    model="gpt-4.1",
    output_type=IntentSchema,
    model_settings=ModelSettings(temperature=0, max_tokens=100, store=True)
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
Create learning activities and assessments aligned strictly with:
- Topic
- The three Learning Objectives
- Learner Level (Beginner / Intermediate / Advanced)
- Total Lesson Duration

Do NOT add or modify objectives. Always state which objective(s) each activity or assessment aligns with.

TIME REQUIREMENT (MANDATORY)
Allocate 60–70% of total duration to lecture (not available for activities).
Use ONLY the remaining 20–30% for activities + assessments.
Explicitly calculate and display:
- Total duration
- Lecture time
- Remaining time
The combined time for activities + assessments MUST NOT exceed the remaining time.
If time is limited (<20 min), reduce the number and complexity of activities.

SECTION 1 — Learning Activities
Create 2–4 varied activities.
Each activity must:
- Match learner level
- Directly support one or more objectives
- Use appropriate cybersecurity pedagogy (lab, log analysis, simulation, threat modeling, secure config, etc.)
- Include clear student steps
- Include a precise time estimate

SECTION 2 — Assessments
Create exactly 3 assessment items that:
- Align with the objectives
- Match learner level
- Fit within remaining time
- Use simple single-lesson formats (MCQ, short answer, scenario, mini task)
- Include time estimates

REQUIRED OUTPUT FORMAT:

### Time Allocation Calculation
- Total Duration: … minutes
- Lecture (60–70%): … minutes
- Remaining for Activities + Assessments: … minutes

### Learning Activities

1. Title
- Aligned Objective(s): …
- Description: …
- Steps: …
- Time Required: … minutes

2. Title
- Aligned Objective(s): …
- Description: …
- Steps: …
- Time Required: … minutes

3. Title (if time permits)
- Aligned Objective(s): …
- Description: …
- Steps: …
- Time Required: … minutes

### Assessments

1.
- Aligned Objective(s): …
- Format: …
- Description: …
- Time Required: … minutes

2.
- Aligned Objective(s): …
- Format: …
- Description: …
- Time Required: … minutes

3.
- Aligned Objective(s): …
- Format: …
- Description: …
- Time Required: … minutes

### Time Summary
- Total Activity Time: … minutes
- Total Assessment Time: … minutes
- Grand Total: … minutes

After the time summary, end with exactly:
---
✅ Lesson plan complete! You can:
- Ask me to **regenerate** this lesson plan
- Ask me to **modify** a specific section
- Provide details for a **new lesson plan**""",
    model="gpt-4.1",
    tools=[web_search_preview],
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=2048, store=True)
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

modifier_agent = Agent(
    name="Modifier",
    instructions="""You are a lesson plan editor. Look at the full conversation to find the current lesson plan.

Apply the requested change according to these strict rules:

1. If Topic changes → Regenerate: objectives, learning theory, teaching strategy, activities, assessments. Keep: lesson information structure.
2. If Learner Level changes → Regenerate: objectives, teaching strategy, activities, assessments. Keep: learning theory, lesson information.
3. If Duration changes → Regenerate: activities, assessments. Keep: objectives, learning theory, teaching strategy, lesson information.
4. If Objectives change → First update the objectives as requested, then regenerate: teaching strategy, activities, assessments. Keep: learning theory, lesson information.
5. If Learning Theory changes → Regenerate: teaching strategy, activities, assessments. Keep: objectives, lesson information.
6. If Teaching Strategy changes → Regenerate: activities, assessments. Keep: objectives, learning theory, lesson information.
7. If Activities change → Regenerate activities only. Keep everything else.
8. If Assessments change → Regenerate assessments only. Keep everything else.

Always output the COMPLETE updated lesson plan in the same format including all sections.
For activities and assessments, follow the time allocation rules:
- Lecture: 60-70% of duration
- Activities + Assessments: remaining 20-30%
- Include Time Allocation Calculation and Time Summary sections.""",
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=1, max_tokens=4096, store=True)
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

        # Detect intent
        intent_result = await Runner.run(intent_agent, agent_input)
        intent_output = intent_result.final_output
        intent = intent_output.intent.strip().lower() if intent_output else "other"
        changed_element = intent_output.changed_element.strip().lower() if intent_output else ""

        if intent == "modify":
            result = Runner.run_streamed(modifier_agent, agent_input)
            async for event in stream_agent_response(agent_context, result):
                yield event

        elif intent == "other":
            result = Runner.run_streamed(general_agent, agent_input)
            async for event in stream_agent_response(agent_context, result):
                yield event

        else:
            # new_lesson, regenerate, get_info
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