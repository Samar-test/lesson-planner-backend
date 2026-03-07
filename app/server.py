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

class InfoSchema(BaseModel):
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
    instructions="""You extract lesson plan details from the conversation history.
Extract: Domain, Course title, Topic, Duration, Learner level (Beginner / Intermediate / Advanced).
If all five are present → has_all_details = true, fill all fields.
If any is missing → has_all_details = false.
Always return valid string values even if empty.""",
    model="gpt-4.1",
    output_type=InfoSchema,
    model_settings=ModelSettings(temperature=0, max_tokens=256, store=True)
)

intent_agent = Agent(
    name="Intent_detector",
    instructions="""You classify the teacher's LATEST message. Read the full conversation for context.

INTENT OPTIONS:
- "new_lesson": Teacher provides new lesson details or wants a lesson on a new topic
- "regenerate": Teacher is unhappy with the CURRENT lesson and wants it redone with SAME details (e.g. "not good", "try again", "give me a different one", "regenerate", "redo this")
- "modify": Teacher wants to change ONE specific section of the current lesson plan
- "get_info": Not enough details provided yet
- "other": Off-topic or general question

IMPORTANT RULES:
- "regenerate" = same topic, completely new content
- "modify" = keep most of the plan, change ONE part only
- If teacher says "not good" or "different one" → ALWAYS "regenerate", never "modify"
- If teacher says "change the [section]" → ALWAYS "modify"

If intent is "modify", set changed_element to ONE of:
topic, learner_level, duration, objectives, learning_theory, teaching_strategy, activities, assessments

Return ONLY valid JSON:
{"intent": "regenerate", "changed_element": ""}
{"intent": "modify", "changed_element": "objectives"}""",
    model="gpt-4.1",
    output_type=IntentSchema,
    model_settings=ModelSettings(temperature=0, max_tokens=50, store=True)
)

ft_full_generator = Agent(
    name="FT_Full_Generator",
    instructions="""You generate a cybersecurity lesson plan from the teacher's details.
Output EXACTLY this format, nothing more, nothing less:

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
- Name: ...
- Justification: ...

### Teaching Strategy
- Name: ...
- Justification: ...""",
    model="ft:gpt-3.5-turbo-1106:kau:lesson-plan2:CDfU4BQj",
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=2048, store=True)
)

ft_objectives_generator = Agent(
    name="FT_Objectives_Generator",
    instructions="""You update ONLY the Learning Objectives section of a cybersecurity lesson plan.
Apply the teacher's requested change to the objectives.
Output ONLY the objectives in this exact format:

### Learning Objectives
1. ...
2. ...
3. ...

Nothing else. No other sections.""",
    model="ft:gpt-3.5-turbo-1106:kau:lesson-plan2:CDfU4BQj",
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=512, store=True)
)

ft_theory_generator = Agent(
    name="FT_Theory_Generator",
    instructions="""You update ONLY the Learning Theory section of a cybersecurity lesson plan.
Apply the teacher's requested change.
Output ONLY this section:

### Learning Theory
- Name: ...
- Justification: ...

Nothing else.""",
    model="ft:gpt-3.5-turbo-1106:kau:lesson-plan2:CDfU4BQj",
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=256, store=True)
)

ft_strategy_generator = Agent(
    name="FT_Strategy_Generator",
    instructions="""You update ONLY the Teaching Strategy section of a cybersecurity lesson plan.
Apply the teacher's requested change.
Output ONLY this section:

### Teaching Strategy
- Name: ...
- Justification: ...

Nothing else.""",
    model="ft:gpt-3.5-turbo-1106:kau:lesson-plan2:CDfU4BQj",
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=256, store=True)
)

activities_generator = Agent(
    name="Activities_Generator",
    instructions="""You generate ONLY learning activities for a cybersecurity lesson plan.
You will receive the full lesson plan above. Use the objectives, topic, learner level, and duration.

TIME CALCULATION (MANDATORY - follow exactly):
1. Lecture time = 65% of total duration (round to nearest minute)
2. Remaining time = total duration - lecture time
3. Activities budget = 70% of remaining time (round to nearest minute)
4. Each activity time must sum to EXACTLY the activities budget

NUMBER OF ACTIVITIES:
- Remaining < 15 min → 2 activities
- Remaining 15-25 min → 3 activities
- Remaining > 25 min → 4 activities

OUTPUT EXACTLY THIS FORMAT:

### Time Allocation Calculation
- Total Duration: X minutes
- Lecture (65%): X minutes
- Remaining for Activities + Assessments: X minutes
- Activities budget (70% of remaining): X minutes
- Assessments budget (30% of remaining): X minutes

### Learning Activities

1. [Title]
- Aligned Objective(s): ...
- Description: ...
- Steps: ...
- Time Required: X minutes

[repeat for each activity]""",
    model="o4-mini",
    model_settings=ModelSettings(max_tokens=2048, store=True)
)

assessments_generator = Agent(
    name="Assessments_Generator",
    instructions="""You generate ONLY assessments for a cybersecurity lesson plan.
You will receive the full lesson plan above. Use the objectives, topic, learner level, and duration.

TIME CALCULATION (MANDATORY - follow exactly):
1. Lecture time = 65% of total duration (round to nearest minute)
2. Remaining time = total duration - lecture time
3. Assessments budget = 30% of remaining time (round to nearest minute)
4. Each assessment time must sum to EXACTLY the assessments budget
5. Create exactly 3 assessments

OUTPUT EXACTLY THIS FORMAT:

### Assessments

1. [Title]
- Aligned Objective(s): ...
- Format: ...
- Description: ...
- Time Required: X minutes

2. [Title]
- Aligned Objective(s): ...
- Format: ...
- Description: ...
- Time Required: X minutes

3. [Title]
- Aligned Objective(s): ...
- Format: ...
- Description: ...
- Time Required: X minutes

### Time Summary
- Total Activity Time: X minutes
- Total Assessment Time: X minutes
- Grand Total: X minutes

---
✅ Lesson plan complete! You can:
- Ask me to **regenerate** this lesson plan
- Ask me to **modify** a specific section
- Provide details for a **new lesson plan**""",
    model="o4-mini",
    model_settings=ModelSettings(max_tokens=1024, store=True)
)

get_data_agent = Agent(
    name="Get_data",
    instructions="""Ask only for the missing lesson plan details from:
Domain, Course title, Topic, Duration, Learner level (Beginner / Intermediate / Advanced).
Be concise and friendly. Do not ask for details already provided.""",
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=1, max_tokens=256, store=True)
)

general_agent = Agent(
    name="General_assistant",
    instructions="""You are a helpful cybersecurity education assistant.
Answer briefly and helpfully.
If the teacher wants a lesson plan, remind them to provide: Domain, Course title, Topic, Duration, Learner level.""",
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
            thread.id, after=None, limit=MAX_RECENT_ITEMS,
            order="desc", context=context,
        )
        items = list(reversed(items_page.data))
        agent_input = await simple_to_agent_input(items)
        agent_context = AgentContext(thread=thread, store=self.store, request_context=context)

        def append_context(base_input, text: str):
            return base_input + [{"role": "assistant", "content": text}]

        # Detect intent
        intent_result = await Runner.run(intent_agent, agent_input)
        intent_output = intent_result.final_output
        intent = intent_output.intent.strip().lower() if intent_output else "other"
        changed_element = intent_output.changed_element.strip().lower() if intent_output else ""

        if intent == "other":
            async for event in stream_agent_response(agent_context, Runner.run_streamed(general_agent, agent_input)):
                yield event
            return

        if intent in ("new_lesson", "regenerate", "get_info"):
            info_result = await Runner.run(info_collector_agent, agent_input)
            info = info_result.final_output
            if not info or not info.has_all_details:
                async for event in stream_agent_response(agent_context, Runner.run_streamed(get_data_agent, agent_input)):
                    yield event
                return

            ft_out = str((await Runner.run(ft_full_generator, agent_input)).final_output or "")
            async for event in stream_agent_response(agent_context, Runner.run_streamed(ft_full_generator, agent_input)):
                yield event

            act_input = append_context(agent_input, ft_out)
            act_out = str((await Runner.run(activities_generator, act_input)).final_output or "")
            async for event in stream_agent_response(agent_context, Runner.run_streamed(activities_generator, act_input)):
                yield event

            assess_input = append_context(act_input, act_out)
            async for event in stream_agent_response(agent_context, Runner.run_streamed(assessments_generator, assess_input)):
                yield event
            return

        if intent == "modify":

            if changed_element in ("topic", "learner_level"):
                ft_out = str((await Runner.run(ft_full_generator, agent_input)).final_output or "")
                async for event in stream_agent_response(agent_context, Runner.run_streamed(ft_full_generator, agent_input)):
                    yield event
                act_input = append_context(agent_input, ft_out)
                act_out = str((await Runner.run(activities_generator, act_input)).final_output or "")
                async for event in stream_agent_response(agent_context, Runner.run_streamed(activities_generator, act_input)):
                    yield event
                assess_input = append_context(act_input, act_out)
                async for event in stream_agent_response(agent_context, Runner.run_streamed(assessments_generator, assess_input)):
                    yield event

            elif changed_element == "objectives":
                ft_out = str((await Runner.run(ft_objectives_generator, agent_input)).final_output or "")
                async for event in stream_agent_response(agent_context, Runner.run_streamed(ft_objectives_generator, agent_input)):
                    yield event
                act_input = append_context(agent_input, ft_out)
                act_out = str((await Runner.run(activities_generator, act_input)).final_output or "")
                async for event in stream_agent_response(agent_context, Runner.run_streamed(activities_generator, act_input)):
                    yield event
                assess_input = append_context(act_input, act_out)
                async for event in stream_agent_response(agent_context, Runner.run_streamed(assessments_generator, assess_input)):
                    yield event

            elif changed_element == "learning_theory":
                ft_out = str((await Runner.run(ft_theory_generator, agent_input)).final_output or "")
                async for event in stream_agent_response(agent_context, Runner.run_streamed(ft_theory_generator, agent_input)):
                    yield event
                strat_input = append_context(agent_input, ft_out)
                strat_out = str((await Runner.run(ft_strategy_generator, strat_input)).final_output or "")
                async for event in stream_agent_response(agent_context, Runner.run_streamed(ft_strategy_generator, strat_input)):
                    yield event
                act_input = append_context(strat_input, strat_out)
                act_out = str((await Runner.run(activities_generator, act_input)).final_output or "")
                async for event in stream_agent_response(agent_context, Runner.run_streamed(activities_generator, act_input)):
                    yield event
                assess_input = append_context(act_input, act_out)
                async for event in stream_agent_response(agent_context, Runner.run_streamed(assessments_generator, assess_input)):
                    yield event

            elif changed_element == "teaching_strategy":
                ft_out = str((await Runner.run(ft_strategy_generator, agent_input)).final_output or "")
                async for event in stream_agent_response(agent_context, Runner.run_streamed(ft_strategy_generator, agent_input)):
                    yield event
                act_input = append_context(agent_input, ft_out)
                act_out = str((await Runner.run(activities_generator, act_input)).final_output or "")
                async for event in stream_agent_response(agent_context, Runner.run_streamed(activities_generator, act_input)):
                    yield event
                assess_input = append_context(act_input, act_out)
                async for event in stream_agent_response(agent_context, Runner.run_streamed(assessments_generator, assess_input)):
                    yield event

            elif changed_element == "duration":
                act_out = str((await Runner.run(activities_generator, agent_input)).final_output or "")
                async for event in stream_agent_response(agent_context, Runner.run_streamed(activities_generator, agent_input)):
                    yield event
                assess_input = append_context(agent_input, act_out)
                async for event in stream_agent_response(agent_context, Runner.run_streamed(assessments_generator, assess_input)):
                    yield event

            elif changed_element == "activities":
                async for event in stream_agent_response(agent_context, Runner.run_streamed(activities_generator, agent_input)):
                    yield event

            elif changed_element == "assessments":
                async for event in stream_agent_response(agent_context, Runner.run_streamed(assessments_generator, agent_input)):
                    yield event

            else:
                async for event in stream_agent_response(agent_context, Runner.run_streamed(general_agent, agent_input)):
                    yield event


StarterChatServer = LessonPlannerServer
