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


# ── Schemas ──────────────────────────────────────────────────────────────

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


# ── Helper: extract only teacher requirement messages ────────────────────
# Used during regenerate so the ft model gets clean input (no old plans).

def extract_teacher_requirements(agent_input: list) -> list:
    """
    Filter conversation input to remove assistant messages that contain
    lesson plan content. This ensures the ft model gets the same clean
    input as the first generation — preventing style/format contamination.
    """
    filtered = []
    for msg in agent_input:
        role = msg.get("role", "")
        content = str(msg.get("content", ""))
        # Keep all user messages
        if role == "user":
            filtered.append(msg)
        # Skip assistant messages that contain lesson plan sections
        elif role == "assistant":
            if any(marker in content for marker in [
                "### Learning Objectives",
                "### Lesson Information",
                "### Time Allocation",
                "### Learning Activities",
                "### Assessments",
                "### Time Summary",
                "### Learning Theory",
                "### Teaching Strategy",
            ]):
                continue
            # Keep non-lesson-plan assistant messages
            filtered.append(msg)
    return filtered


# ── Agents ───────────────────────────────────────────────────────────────

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
    instructions="""You analyze the teacher's LATEST message (the most recent one only).

Classify the intent as one of:
- "new_lesson": Teacher is providing NEW lesson details (different topic, domain, duration, etc.)
- "regenerate": Teacher is unhappy with the current plan and wants a COMPLETELY NEW version.
- "modify": Teacher wants to change a SPECIFIC NAMED section of the lesson plan.
- "get_info": Teacher hasn't provided enough details yet.
- "other": General question or unclear.

═══════════════════════════════════════════════════════════════
ABSOLUTE RULE FOR MODIFY vs REGENERATE:

"modify" requires the teacher to EXPLICITLY NAME a section in their latest message.
Valid section names: topic, learner_level, duration, objectives, learning_theory, teaching_strategy, activities, assessments.

If the latest message does NOT contain an explicit section name → classify as "regenerate".

DO NOT look at what was discussed in previous turns to infer a section.
DO NOT assume the teacher means the most recently changed section.
ONLY the words in the latest message matter for this classification.

If there is ANY doubt → choose "regenerate" over "modify".
═══════════════════════════════════════════════════════════════

REGENERATE examples (vague dissatisfaction, NO section named):
- "This is not good, give me a different one" → regenerate
- "I don't like this, try again" → regenerate
- "Not what I wanted, start over" → regenerate
- "Can you redo this?" → regenerate
- "Generate another one" → regenerate
- "This doesn't work for my class" → regenerate
- "Give me a better one" → regenerate
- "No, try again" → regenerate
- "Do it again" → regenerate
- "I need something different" → regenerate

MODIFY examples (section EXPLICITLY named in the message):
- "Change the learning objectives to focus on practical skills" → modify, objectives
- "Make the assessments shorter" → modify, assessments
- "Change only the assessments" → modify, assessments
- "I want different activities" → modify, activities
- "Give me different activities" → modify, activities
- "Update the teaching strategy to use PBL" → modify, teaching_strategy
- "Change the duration to 90 minutes" → modify, duration
- "Switch the topic to cloud security" → modify, topic
- "Change learner level to advanced" → modify, learner_level

NEW_LESSON examples (entirely new lesson details):
- "I want a lesson on phishing" → new_lesson
- "Create a 90-minute lesson on malware for beginners" → new_lesson
- "New lesson: Cloud Security, 45 min, advanced" → new_lesson

If intent is "modify", set changed_element to one of:
topic, learner_level, duration, objectives, learning_theory, teaching_strategy, activities, assessments

Reply ONLY with valid JSON.""",
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

# ── ft_modify_agent: used during modify to regenerate ft-model sections ──
ft_modify_agent = Agent(
    name="FT_modify_generator",
    instructions="""You are a lesson plan editor using your expertise in cybersecurity education.

Look at the FULL conversation history to find the current lesson plan and the teacher's requested change.

You MUST output the following sections based on the instructions given to you in the user's context message.
Use exactly this format for whichever sections you are told to regenerate:

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
- Justification:

IMPORTANT: Always output ALL FOUR sections above. For sections you are told to KEEP, copy them exactly from the existing lesson plan. For sections you are told to REGENERATE, create new content that reflects the teacher's requested change.""",
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

TIME REQUIREMENT (STRICTLY ENFORCED)
Step 1: Calculate total duration from the lesson info.
Step 2: Lecture time = 65% of total duration (round to nearest minute).
Step 3: Remaining time = total duration - lecture time.
Step 4: Activities + assessments MUST use EXACTLY the remaining time. Not more, not less.
Step 5: Distribute remaining time: 70% for activities, 30% for assessments (round to nearest minute).

If remaining time is less than 15 minutes: create only 2 activities and 2 assessments.
If remaining time is 15-25 minutes: create 3 activities and 3 assessments.
If remaining time is more than 25 minutes: create 4 activities and 3 assessments.

Each activity and assessment MUST have a time estimate. The sum of all times MUST equal the remaining time exactly.

REQUIRED OUTPUT FORMAT (follow this order exactly):

### Time Allocation Calculation
- Total Duration: X minutes
- Lecture (65%): X minutes
- Remaining for Activities + Assessments: X minutes
- Activities budget (70% of remaining): X minutes
- Assessments budget (30% of remaining): X minutes

### Learning Activities

1. [Title]
- Aligned Objective(s): …
- Description: …
- Steps: …
- Time Required: X minutes

2. [Title]
- Aligned Objective(s): …
- Description: …
- Steps: …
- Time Required: X minutes

3. [Title] (if time permits)
- Aligned Objective(s): …
- Description: …
- Steps: …
- Time Required: X minutes

### Assessments

1.
- Aligned Objective(s): …
- Format: …
- Description: …
- Time Required: X minutes

2.
- Aligned Objective(s): …
- Format: …
- Description: …
- Time Required: X minutes

3.
- Aligned Objective(s): …
- Format: …
- Description: …
- Time Required: X minutes

### Time Summary
- Total Activity Time: X minutes
- Total Assessment Time: X minutes
- Grand Total: X minutes (MUST equal remaining time)

After the time summary, end with exactly:
---
✅ Lesson plan complete! You can:
- Ask me to **regenerate** this lesson plan
- Ask me to **modify** a specific section
- Provide details for a **new lesson plan**""",
    model="o4-mini",
    tools=[web_search_preview],
    model_settings=ModelSettings(max_tokens=4096, store=True)
)

# ── Assessments-only agent ──
assessments_only_agent = Agent(
    name="Assessments_only_generator",
    instructions="""You are an expert in cybersecurity education assessment design.

Look at the FULL conversation history to find the current lesson plan. You must regenerate ONLY the assessments section while keeping everything else the same.

Your output must include the COMPLETE lesson plan with ALL sections. Copy the existing sections exactly, and ONLY create new content for the Assessments section.

Output format (ALL sections, in this order):

### Lesson Information
(copy exactly from existing plan)

### Learning Objectives
(copy exactly from existing plan)

### Learning Theory
(copy exactly from existing plan)

### Teaching Strategy
(copy exactly from existing plan)

### Time Allocation Calculation
(copy exactly from existing plan)

### Learning Activities
(copy exactly from existing plan)

### Assessments

(CREATE NEW assessments here. They must:
- Align with the existing learning objectives
- Fit within the assessments time budget from the Time Allocation
- Each assessment must specify: Aligned Objective(s), Format, Description, Time Required
- Total assessment time must equal the assessments budget exactly)

1.
- Aligned Objective(s): …
- Format: …
- Description: …
- Time Required: X minutes

2.
- Aligned Objective(s): …
- Format: …
- Description: …
- Time Required: X minutes

3.
- Aligned Objective(s): …
- Format: …
- Description: …
- Time Required: X minutes

### Time Summary
- Total Activity Time: X minutes (same as before)
- Total Assessment Time: X minutes (must equal assessments budget)
- Grand Total: X minutes (MUST equal remaining time)

After the time summary, end with exactly:
---
✅ Lesson plan complete! You can:
- Ask me to **regenerate** this lesson plan
- Ask me to **modify** a specific section
- Provide details for a **new lesson plan**""",
    model="o4-mini",
    model_settings=ModelSettings(max_tokens=4096, store=True)
)

# ── Activities-only agent ──
activities_only_agent = Agent(
    name="Activities_only_generator",
    instructions="""You are an expert in cybersecurity education and instructional design.

Look at the FULL conversation history to find the current lesson plan. You must regenerate ONLY the Learning Activities section while keeping everything else the same.

Your output must include the COMPLETE lesson plan with ALL sections. Copy the existing sections exactly, and ONLY create new content for the Learning Activities section.

Output format (ALL sections, in this order):

### Lesson Information
(copy exactly from existing plan)

### Learning Objectives
(copy exactly from existing plan)

### Learning Theory
(copy exactly from existing plan)

### Teaching Strategy
(copy exactly from existing plan)

### Time Allocation Calculation
(copy exactly from existing plan)

### Learning Activities

(CREATE NEW activities here. They must:
- Align with the existing learning objectives
- Fit within the activities time budget from the Time Allocation
- Each activity must specify: Aligned Objective(s), Description, Steps, Time Required
- Total activity time must equal the activities budget exactly)

### Assessments
(copy exactly from existing plan)

### Time Summary
- Total Activity Time: X minutes (must equal activities budget)
- Total Assessment Time: X minutes (same as before)
- Grand Total: X minutes (MUST equal remaining time)

After the time summary, end with exactly:
---
✅ Lesson plan complete! You can:
- Ask me to **regenerate** this lesson plan
- Ask me to **modify** a specific section
- Provide details for a **new lesson plan**""",
    model="o4-mini",
    model_settings=ModelSettings(max_tokens=4096, store=True)
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


# ── Cascade Logic ────────────────────────────────────────────────────────

def get_cascade(changed_element: str) -> dict:
    """Determine which agents need to run based on what changed."""
    cascades = {
        "topic": {
            "needs_ft_model": True,
            "ft_regenerate": ["objectives", "learning_theory", "teaching_strategy"],
            "needs_activities_and_assessments": True,
            "needs_activities_only": False,
            "needs_assessments_only": False,
        },
        "learner_level": {
            "needs_ft_model": True,
            "ft_regenerate": ["objectives", "teaching_strategy"],
            "needs_activities_and_assessments": True,
            "needs_activities_only": False,
            "needs_assessments_only": False,
        },
        "duration": {
            "needs_ft_model": False,
            "ft_regenerate": [],
            "needs_activities_and_assessments": True,
            "needs_activities_only": False,
            "needs_assessments_only": False,
        },
        "objectives": {
            "needs_ft_model": True,
            "ft_regenerate": ["objectives", "teaching_strategy"],
            "needs_activities_and_assessments": True,
            "needs_activities_only": False,
            "needs_assessments_only": False,
        },
        "learning_theory": {
            "needs_ft_model": True,
            "ft_regenerate": ["teaching_strategy"],
            "needs_activities_and_assessments": True,
            "needs_activities_only": False,
            "needs_assessments_only": False,
        },
        "teaching_strategy": {
            "needs_ft_model": False,
            "ft_regenerate": [],
            "needs_activities_and_assessments": True,
            "needs_activities_only": False,
            "needs_assessments_only": False,
        },
        "activities": {
            "needs_ft_model": False,
            "ft_regenerate": [],
            "needs_activities_and_assessments": False,
            "needs_activities_only": True,
            "needs_assessments_only": False,
        },
        "assessments": {
            "needs_ft_model": False,
            "ft_regenerate": [],
            "needs_activities_and_assessments": False,
            "needs_activities_only": False,
            "needs_assessments_only": True,
        },
    }

    return cascades.get(changed_element, {
        "needs_ft_model": False,
        "ft_regenerate": [],
        "needs_activities_and_assessments": True,
        "needs_activities_only": False,
        "needs_assessments_only": False,
    })


# ── Server ───────────────────────────────────────────────────────────────

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

        # ── Step 1: Detect intent ──
        intent_result = await Runner.run(intent_agent, agent_input)
        intent_output = intent_result.final_output
        intent = intent_output.intent.strip().lower() if intent_output else "other"
        changed_element = intent_output.changed_element.strip().lower() if intent_output else ""

        # ── Step 2: Route based on intent ──

        if intent == "modify" and changed_element:
            cascade = get_cascade(changed_element)

            if cascade["needs_assessments_only"]:
                result = Runner.run_streamed(assessments_only_agent, agent_input)
                async for event in stream_agent_response(agent_context, result):
                    yield event

            elif cascade["needs_activities_only"]:
                result = Runner.run_streamed(activities_only_agent, agent_input)
                async for event in stream_agent_response(agent_context, result):
                    yield event

            elif cascade["needs_ft_model"]:
                # ft model first (non-streamed), then stream activities
                regen_list = ", ".join(cascade["ft_regenerate"])
                ft_context_msg = (
                    f"The teacher wants to change: {changed_element}. "
                    f"Regenerate these sections: {regen_list}. "
                    f"Keep all other sections exactly as they are in the current lesson plan. "
                    f"Output the complete Lesson Information, Learning Objectives, Learning Theory, and Teaching Strategy sections."
                )
                ft_input = agent_input + [{"role": "user", "content": ft_context_msg}]

                ft_result = await Runner.run(ft_modify_agent, ft_input)
                ft_output = str(ft_result.final_output) if ft_result.final_output else ""

                activities_context_msg = (
                    f"Here is the updated lesson plan (Lesson Information, Objectives, Theory, Strategy):\n\n"
                    f"{ft_output}\n\n"
                    f"Now generate the Time Allocation, Learning Activities, Assessments, and Time Summary "
                    f"sections based on the above. Follow the time rules strictly."
                )
                activities_input = agent_input + [
                    {"role": "assistant", "content": ft_output},
                    {"role": "user", "content": activities_context_msg}
                ]

                result = Runner.run_streamed(activities_generator, activities_input)
                async for event in stream_agent_response(agent_context, result):
                    yield event

            else:
                # Duration or teaching_strategy: just regenerate activities+assessments
                duration_context_msg = (
                    f"The teacher wants to change: {changed_element}. "
                    f"Look at the current lesson plan in the conversation. "
                    f"Keep the Lesson Information, Learning Objectives, Learning Theory, and Teaching Strategy exactly as they are. "
                    f"Regenerate ONLY the Time Allocation, Learning Activities, Assessments, and Time Summary based on the change."
                )
                activities_input = agent_input + [{"role": "user", "content": duration_context_msg}]

                result = Runner.run_streamed(activities_generator, activities_input)
                async for event in stream_agent_response(agent_context, result):
                    yield event

        elif intent == "other":
            result = Runner.run_streamed(general_agent, agent_input)
            async for event in stream_agent_response(agent_context, result):
                yield event

        else:
            # ── new_lesson, regenerate, get_info ──
            info_result = await Runner.run(info_collector_agent, agent_input)
            info = info_result.final_output

            if info and info.has_all_details:
                # ══════════════════════════════════════════════════════════
                # FIX FOR TEST 2: Strip old lesson plans from context so
                # the ft model gets CLEAN input, same as first generation.
                # This prevents style/format contamination.
                # ══════════════════════════════════════════════════════════
                clean_input = extract_teacher_requirements(agent_input)

                # Stream ft model output
                lesson_result = Runner.run_streamed(lesson_plan_generator, clean_input)
                async for event in stream_agent_response(agent_context, lesson_result):
                    yield event

                # Reload thread to get the freshly saved ft output
                updated_page = await self.store.load_thread_items(
                    thread.id, after=None, limit=MAX_RECENT_ITEMS,
                    order="desc", context=context,
                )
                updated_items = list(reversed(updated_page.data))
                updated_input = await simple_to_agent_input(updated_items)

                # ══════════════════════════════════════════════════════════
                # FIX: Clean activities input too, then add back ONLY the
                # latest ft output so o4-mini doesn't mix old/new plans.
                # ══════════════════════════════════════════════════════════
                clean_activities_input = extract_teacher_requirements(updated_input)
                # Find and append only the newest ft model output
                for msg in reversed(updated_input):
                    role = msg.get("role", "")
                    content = str(msg.get("content", ""))
                    if role == "assistant" and "### Learning Objectives" in content:
                        clean_activities_input.append(msg)
                        break

                activities_result = Runner.run_streamed(activities_generator, clean_activities_input)
                async for event in stream_agent_response(agent_context, activities_result):
                    yield event

            else:
                result = Runner.run_streamed(get_data_agent, agent_input)
                async for event in stream_agent_response(agent_context, result):
                    yield event


StarterChatServer = LessonPlannerServer