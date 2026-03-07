"""ChatKit server - Conversational Lesson Planner (Refactored)."""

from __future__ import annotations

import json
import re
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

# ── Lesson Plan JSON Schema ───────────────────────────────────────────────
EMPTY_PLAN: dict[str, Any] = {
    "domain": "",
    "course_title": "",
    "topic": "",
    "duration": "",
    "learner_level": "",
    "objectives": [],
    "learning_theory": {"name": "", "justification": ""},
    "teaching_strategy": {"name": "", "justification": ""},
    "time_allocation": {},
    "activities": [],
    "assessments": [],
    "time_summary": {},
}

PLAN_KEY = "lesson_plan"

# ── Cascade Rules ─────────────────────────────────────────────────────────
CASCADE_RULES: dict[str, dict] = {
    "topic": {
        "ft_sections": ["objectives", "learning_theory", "teaching_strategy"],
        "needs_activities": True,
        "activities_only": False,
        "assessments_only": False,
    },
    "learner_level": {
        "ft_sections": ["objectives", "teaching_strategy"],
        "needs_activities": True,
        "activities_only": False,
        "assessments_only": False,
    },
    "duration": {
        "ft_sections": [],
        "needs_activities": True,
        "activities_only": False,
        "assessments_only": False,
    },
    "objectives": {
        "ft_sections": ["objectives", "teaching_strategy"],
        "needs_activities": True,
        "activities_only": False,
        "assessments_only": False,
    },
    "learning_theory": {
        "ft_sections": ["learning_theory", "teaching_strategy"],
        "needs_activities": True,
        "activities_only": False,
        "assessments_only": False,
    },
    "teaching_strategy": {
        "ft_sections": [],
        "needs_activities": True,
        "activities_only": False,
        "assessments_only": False,
    },
    "activities": {
        "ft_sections": [],
        "needs_activities": True,
        "activities_only": True,
        "assessments_only": False,
    },
    "assessments": {
        "ft_sections": [],
        "needs_activities": True,
        "activities_only": False,
        "assessments_only": True,
    },
}


# ── Pydantic Schema ───────────────────────────────────────────────────────

class OrchestratorOutput(BaseModel):
    intent: str
    changed_element: str
    has_all_details: bool
    domain: str
    course_title: str
    topic: str
    duration: str
    learner_level: str
    missing_fields: list[str]


# ── Agents ────────────────────────────────────────────────────────────────

orchestrator_agent = Agent(
    name="Orchestrator",
    instructions="""You are the orchestrator of a cybersecurity lesson planner.

Your job is to analyze the FULL conversation history and return a structured JSON with:

1. INTENT — classify the teacher's latest message as one of:
   - "new_lesson": Teacher is providing new lesson details or starting fresh
   - "regenerate": Teacher wants a completely new version of the ENTIRE lesson plan (vague dissatisfaction, no specific section mentioned)
   - "modify": Teacher wants to change a SPECIFIC part of the lesson plan
   - "get_info": Not enough details provided yet to generate a plan
   - "other": General question or unclear request

2. CHANGED_ELEMENT — only when intent is "modify", identify exactly what changed:
   topic | learner_level | duration | objectives | learning_theory | teaching_strategy | activities | assessments

3. LESSON DETAILS — extract from the ENTIRE conversation history:
   - domain, course_title, topic, duration, learner_level
   - has_all_details: true only if ALL five fields are present
   - missing_fields: list of field names that are missing

CRITICAL DISTINCTION — regenerate vs modify:
- "regenerate" = vague dissatisfaction, wants a fresh take, NO specific section mentioned
  Examples: "I don't like this", "try again", "give me another one", "redo this"
- "modify" = targets a SPECIFIC section
  Examples: "change the objectives", "make activities shorter", "update teaching strategy"

Always return valid values for all string fields even if empty string.""",
    model="gpt-4.1",
    output_type=OrchestratorOutput,
    model_settings=ModelSettings(temperature=0, max_tokens=512, store=True)
)

ft_model_agent = Agent(
    name="FT_lesson_planner",
    instructions="""You receive the teacher's lesson information from the previous conversation.

Your task:
1. You will receive a JSON object from the previous agent with these keys: domain, course_title, topic, duration, learner_level.
2. Based on your fine-tuning, generate:
- Three learning objectives (use only the verbs you train on)
- A suitable learning theory
- A suitable teaching strategy

Final output format (exactly):

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
    name="Activities_Assessments_Generator",
    instructions="""You are an expert in cybersecurity education and instructional design.

You will receive the COMPLETE current lesson plan as structured context.
Generate ONLY the sections you are told to generate (activities, assessments, or both).

TIME REQUIREMENT (STRICTLY ENFORCED):
Step 1: Total duration is provided in the lesson info.
Step 2: Lecture time = 65% of total duration (round to nearest minute).
Step 3: Remaining time = total duration - lecture time.
Step 4: Activities + assessments MUST use EXACTLY the remaining time.
Step 5: Activities budget = 70% of remaining. Assessments budget = 30% of remaining.

Number of activities/assessments:
- Remaining < 15 min: 2 activities, 2 assessments
- Remaining 15-25 min: 3 activities, 3 assessments
- Remaining > 25 min: 4 activities, 3 assessments

Each activity/assessment MUST have a time estimate. Sums MUST be exact.

Respond ONLY with valid JSON in this exact format (no extra text, no markdown):
{
  "time_allocation": {
    "total_duration_minutes": 0,
    "lecture_minutes": 0,
    "remaining_minutes": 0,
    "activities_budget_minutes": 0,
    "assessments_budget_minutes": 0
  },
  "activities": [
    {
      "title": "...",
      "aligned_objectives": ["objective 1"],
      "description": "...",
      "steps": ["step 1", "step 2"],
      "time_minutes": 0
    }
  ],
  "assessments": [
    {
      "title": "...",
      "aligned_objectives": ["objective 1"],
      "format": "...",
      "description": "...",
      "time_minutes": 0
    }
  ],
  "time_summary": {
    "total_activity_time": 0,
    "total_assessment_time": 0,
    "grand_total": 0
  }
}""",
    model="gpt-4.1",
    tools=[web_search_preview],
    model_settings=ModelSettings(temperature=0.7, max_tokens=4096, store=True)
)

get_info_agent = Agent(
    name="Get_info",
    instructions="""You are a helpful cybersecurity lesson planner assistant.

If the teacher is missing lesson details, ask ONLY for what's missing from:
- Domain
- Course title
- Topic
- Duration (in minutes)
- Learner level (Beginner / Intermediate / Advanced)

Be concise, friendly, and ask only for missing fields.
If the teacher asks a general question, answer it helpfully and briefly.""",
    model="gpt-4.1",
    model_settings=ModelSettings(temperature=1, max_tokens=512, store=True)
)


# ── FT Model Output Parser ────────────────────────────────────────────────

def parse_ft_output(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "objectives": [],
        "learning_theory": {"name": "", "justification": ""},
        "teaching_strategy": {"name": "", "justification": ""},
    }

    if not text:
        return result

    # Objectives — only lines between ### Learning Objectives and ### Learning Theory
    obj_match = re.search(
        r"###\s*Learning Objectives\s*\n(.*?)(?=###\s*Learning Theory)",
        text, re.DOTALL | re.IGNORECASE
    )
    if obj_match:
        for line in obj_match.group(1).splitlines():
            line = line.strip()
            m = re.match(r"^\d+[\.\)]\s*(.+)", line)
            if m:
                result["objectives"].append(m.group(1).strip())
            if len(result["objectives"]) == 3:
                break

    # Learning Theory
    lt_match = re.search(
        r"###\s*Learning Theory\s*\n-\s*Name:\s*(.+?)\n-\s*Justification:\s*(.+?)(?=###|$)",
        text, re.DOTALL | re.IGNORECASE
    )
    if lt_match:
        result["learning_theory"]["name"] = lt_match.group(1).strip()
        result["learning_theory"]["justification"] = lt_match.group(2).strip()

    # Teaching Strategy
    ts_match = re.search(
        r"###\s*Teaching Strategy\s*\n-\s*Name:\s*(.+?)\n-\s*Justification:\s*(.+?)(?=###|$)",
        text, re.DOTALL | re.IGNORECASE
    )
    if ts_match:
        result["teaching_strategy"]["name"] = ts_match.group(1).strip()
        result["teaching_strategy"]["justification"] = ts_match.group(2).strip()

    return result


def parse_activities_output(text: str) -> dict[str, Any]:
    """Parse JSON response from activities agent, stripping markdown fences if present."""
    if not text:
        return {}
    try:
        clean = text.strip()
        clean = re.sub(r"^```[a-z]*\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean)
        return json.loads(clean)
    except (json.JSONDecodeError, Exception):
        return {}


# ── Helpers ───────────────────────────────────────────────────────────────

def plan_to_markdown(plan: dict[str, Any]) -> str:
    """Convert the structured JSON plan to clean markdown for display."""
    lines = []

    lines.append("### Lesson Information")
    lines.append(f"- Domain: {plan.get('domain', '')}")
    lines.append(f"- Course title: {plan.get('course_title', '')}")
    lines.append(f"- Topic: {plan.get('topic', '')}")
    lines.append(f"- Duration: {plan.get('duration', '')} minutes")
    lines.append(f"- Learner level: {plan.get('learner_level', '')}")
    lines.append("")

    lines.append("### Learning Objectives")
    for i, obj in enumerate(plan.get("objectives", []), 1):
        lines.append(f"{i}. {obj}")
    lines.append("")

    lt = plan.get("learning_theory", {})
    lines.append("### Learning Theory")
    lines.append(f"- Name: {lt.get('name', '')}")
    lines.append(f"- Justification: {lt.get('justification', '')}")
    lines.append("")

    ts = plan.get("teaching_strategy", {})
    lines.append("### Teaching Strategy")
    lines.append(f"- Name: {ts.get('name', '')}")
    lines.append(f"- Justification: {ts.get('justification', '')}")
    lines.append("")

    ta = plan.get("time_allocation", {})
    if ta:
        lines.append("### Time Allocation Calculation")
        lines.append(f"- Total Duration: {ta.get('total_duration_minutes', '')} minutes")
        lines.append(f"- Lecture (65%): {ta.get('lecture_minutes', '')} minutes")
        lines.append(f"- Remaining for Activities + Assessments: {ta.get('remaining_minutes', '')} minutes")
        lines.append(f"- Activities budget (70% of remaining): {ta.get('activities_budget_minutes', '')} minutes")
        lines.append(f"- Assessments budget (30% of remaining): {ta.get('assessments_budget_minutes', '')} minutes")
        lines.append("")

    activities = plan.get("activities", [])
    if activities:
        lines.append("### Learning Activities")
        for i, act in enumerate(activities, 1):
            lines.append(f"\n{i}. {act.get('title', '')}")
            lines.append(f"- Aligned Objective(s): {', '.join(act.get('aligned_objectives', []))}")
            lines.append(f"- Description: {act.get('description', '')}")
            steps = act.get("steps", [])
            if steps:
                lines.append("- Steps:")
                for step in steps:
                    lines.append(f"  - {step}")
            lines.append(f"- Time Required: {act.get('time_minutes', '')} minutes")
        lines.append("")

    assessments = plan.get("assessments", [])
    if assessments:
        lines.append("### Assessments")
        for i, asm in enumerate(assessments, 1):
            lines.append(f"\n{i}. {asm.get('title', '')}")
            lines.append(f"- Aligned Objective(s): {', '.join(asm.get('aligned_objectives', []))}")
            lines.append(f"- Format: {asm.get('format', '')}")
            lines.append(f"- Description: {asm.get('description', '')}")
            lines.append(f"- Time Required: {asm.get('time_minutes', '')} minutes")
        lines.append("")

    ts_summary = plan.get("time_summary", {})
    if ts_summary:
        lines.append("### Time Summary")
        lines.append(f"- Total Activity Time: {ts_summary.get('total_activity_time', '')} minutes")
        lines.append(f"- Total Assessment Time: {ts_summary.get('total_assessment_time', '')} minutes")
        lines.append(f"- Grand Total: {ts_summary.get('grand_total', '')} minutes")
        lines.append("")

    lines.append("---")
    lines.append("✅ Lesson plan complete! You can:")
    lines.append("- Ask me to **regenerate** this lesson plan")
    lines.append("- Ask me to **modify** a specific section")
    lines.append("- Provide details for a **new lesson plan**")

    return "\n".join(lines)


def build_ft_prompt(plan: dict[str, Any], sections_to_regenerate: list[str], info: OrchestratorOutput) -> str:
    """Build a clear, explicit prompt for the ft model."""
    lines = [
        "Generate lesson plan sections for the following cybersecurity lesson:",
        "",
        f"Domain: {info.domain}",
        f"Course title: {info.course_title}",
        f"Topic: {info.topic}",
        f"Duration: {info.duration} minutes",
        f"Learner level: {info.learner_level}",
        "",
    ]

    all_sections = ["objectives", "learning_theory", "teaching_strategy"]
    for section in all_sections:
        if section in sections_to_regenerate:
            lines.append(f"REGENERATE {section.upper()}: Create new content for this section.")
        else:
            if section == "objectives" and plan.get("objectives"):
                lines.append(f"KEEP OBJECTIVES: {json.dumps(plan['objectives'])}")
            elif section == "learning_theory" and plan.get("learning_theory", {}).get("name"):
                lt = plan["learning_theory"]
                lines.append(f"KEEP LEARNING_THEORY: Name={lt['name']}, Justification={lt['justification']}")
            elif section == "teaching_strategy" and plan.get("teaching_strategy", {}).get("name"):
                ts = plan["teaching_strategy"]
                lines.append(f"KEEP TEACHING_STRATEGY: Name={ts['name']}, Justification={ts['justification']}")
            else:
                lines.append(f"REGENERATE {section.upper()}: Create new content (no existing content found).")

    return "\n".join(lines)


def build_activities_prompt(
    plan: dict[str, Any],
    activities_only: bool = False,
    assessments_only: bool = False,
) -> str:
    """Build a clean, explicit prompt for the activities agent."""
    lines = [
        "Generate activities and assessments for this cybersecurity lesson.",
        "",
        "CURRENT LESSON PLAN:",
        f"- Domain: {plan.get('domain', '')}",
        f"- Course title: {plan.get('course_title', '')}",
        f"- Topic: {plan.get('topic', '')}",
        f"- Duration: {plan.get('duration', '')} minutes",
        f"- Learner level: {plan.get('learner_level', '')}",
        "",
        "Learning Objectives:",
    ]
    for i, obj in enumerate(plan.get("objectives", []), 1):
        lines.append(f"{i}. {obj}")

    lt = plan.get("learning_theory", {})
    ts = plan.get("teaching_strategy", {})
    lines.append(f"\nLearning Theory: {lt.get('name', '')}")
    lines.append(f"Teaching Strategy: {ts.get('name', '')}")
    lines.append("")

    if activities_only:
        lines.append("INSTRUCTION: Regenerate ONLY the activities. Keep assessments the same.")
        lines.append("KEEP THESE ASSESSMENTS exactly as-is in your JSON response:")
        lines.append(json.dumps(plan.get("assessments", [])))
        lines.append("KEEP THIS TIME ALLOCATION exactly as-is in your JSON response:")
        lines.append(json.dumps(plan.get("time_allocation", {})))
    elif assessments_only:
        lines.append("INSTRUCTION: Regenerate ONLY the assessments. Keep activities the same.")
        lines.append("KEEP THESE ACTIVITIES exactly as-is in your JSON response:")
        lines.append(json.dumps(plan.get("activities", [])))
        lines.append("KEEP THIS TIME ALLOCATION exactly as-is in your JSON response:")
        lines.append(json.dumps(plan.get("time_allocation", {})))
    else:
        lines.append("INSTRUCTION: Regenerate BOTH activities and assessments.")

    return "\n".join(lines)


# ── Server ────────────────────────────────────────────────────────────────

class LessonPlannerServer(ChatKitServer[dict[str, Any]]):
    def __init__(self) -> None:
        self.store: MemoryStore = MemoryStore()
        super().__init__(self.store)

    async def _load_plan(self, thread_id: str, context: dict[str, Any]) -> dict[str, Any]:
        """Load the current lesson plan from thread metadata."""
        import copy
        try:
            thread = await self.store.load_thread(thread_id, context)
            if thread.metadata and "lesson_plan" in thread.metadata:
                return thread.metadata["lesson_plan"]
        except Exception:
            pass
        return copy.deepcopy(EMPTY_PLAN)

    async def _save_plan(self, thread_id: str, plan: dict[str, Any], context: dict[str, Any]) -> None:
        """Save the current lesson plan to thread metadata."""
        try:
            thread = await self.store.load_thread(thread_id, context)
            updated_metadata = dict(thread.metadata or {})
            updated_metadata["lesson_plan"] = plan
            updated_thread = thread.model_copy(update={"metadata": updated_metadata})
            await self.store.save_thread(updated_thread, context)
        except Exception:
            pass

    async def respond(
        self,
        thread: ThreadMetadata,
        item: UserMessageItem | None,
        context: dict[str, Any],
    ) -> AsyncIterator[ThreadStreamEvent]:

        # Load conversation history
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

        # Load the current lesson plan (single source of truth)
        current_plan = await self._load_plan(thread.id, context)

        # ── Step 1: Orchestrator ──────────────────────────────────────────
        orch_result = await Runner.run(orchestrator_agent, agent_input)
        info: OrchestratorOutput = orch_result.final_output

        intent = info.intent.strip().lower()
        changed_element = info.changed_element.strip().lower()

        # ── Step 2: Route ─────────────────────────────────────────────────

        # General question
        if intent == "other":
            result = Runner.run_streamed(get_info_agent, agent_input)
            async for event in stream_agent_response(agent_context, result):
                yield event
            return

        # Missing details
        if intent == "get_info" or (intent in ("new_lesson", "regenerate") and not info.has_all_details):
            result = Runner.run_streamed(get_info_agent, agent_input)
            async for event in stream_agent_response(agent_context, result):
                yield event
            return

        # ── New lesson or full regenerate ─────────────────────────────────
        if intent in ("new_lesson", "regenerate"):
            current_plan["domain"] = info.domain
            current_plan["course_title"] = info.course_title
            current_plan["topic"] = info.topic
            current_plan["duration"] = info.duration
            current_plan["learner_level"] = info.learner_level

            # FT model: objectives, learning theory, teaching strategy
            ft_input = json.dumps({
                "domain": current_plan["domain"],
                "course_title": current_plan["course_title"],
                "topic": current_plan["topic"],
                "duration": current_plan["duration"],
                "learner_level": current_plan["learner_level"]
            })
            ft_result = await Runner.run(ft_model_agent, ft_input)
            ft_text = str(ft_result.final_output) if ft_result.final_output else ""
            ft_data = parse_ft_output(ft_text)

            current_plan["objectives"] = ft_data["objectives"]
            current_plan["learning_theory"] = ft_data["learning_theory"]
            current_plan["teaching_strategy"] = ft_data["teaching_strategy"]

            # Activities agent: time allocation, activities, assessments
            act_prompt = build_activities_prompt(current_plan)
            act_result = await Runner.run(activities_generator, act_prompt)
            act_text = str(act_result.final_output) if act_result.final_output else ""
            act_data = parse_activities_output(act_text)

            if act_data:
                current_plan["time_allocation"] = act_data.get("time_allocation", {})
                current_plan["activities"] = act_data.get("activities", [])
                current_plan["assessments"] = act_data.get("assessments", [])
                current_plan["time_summary"] = act_data.get("time_summary", {})

            await self._save_plan(thread.id, current_plan, context)

            markdown_output = plan_to_markdown(current_plan)
            display_input = [{"role": "user", "content": f"Output this lesson plan exactly as written, no changes:\n\n{markdown_output}"}]
            display_agent = Agent(
                name="Display",
                instructions="Output the content exactly as provided. Do not change, summarize, or add anything.",
                model="gpt-4.1-mini",
                model_settings=ModelSettings(temperature=0, max_tokens=4096, store=False)
            )
            result = Runner.run_streamed(display_agent, display_input)
            async for event in stream_agent_response(agent_context, result):
                yield event
            return

        # ── Modify specific section ───────────────────────────────────────
        if intent == "modify" and changed_element:
            cascade = CASCADE_RULES.get(changed_element, {
                "ft_sections": [],
                "needs_activities": True,
                "activities_only": False,
                "assessments_only": False,
            })

            # Update lesson metadata if a top-level field changed
            if changed_element == "topic":
                current_plan["topic"] = info.topic
            elif changed_element == "learner_level":
                current_plan["learner_level"] = info.learner_level
            elif changed_element == "duration":
                current_plan["duration"] = info.duration

            ft_sections = cascade.get("ft_sections", [])
            activities_only = cascade.get("activities_only", False)
            assessments_only = cascade.get("assessments_only", False)
            needs_activities = cascade.get("needs_activities", False)

            # Run FT model only if needed
            if ft_sections:
                ft_input = json.dumps({
                    "domain": current_plan["domain"],
                    "course_title": current_plan["course_title"],
                    "topic": current_plan["topic"],
                    "duration": current_plan["duration"],
                    "learner_level": current_plan["learner_level"]
                })
                ft_result = await Runner.run(ft_model_agent, ft_input)
                ft_text = str(ft_result.final_output) if ft_result.final_output else ""
                ft_data = parse_ft_output(ft_text)

                # Surgically update only regenerated sections
                if "objectives" in ft_sections:
                    current_plan["objectives"] = ft_data["objectives"]
                if "learning_theory" in ft_sections:
                    current_plan["learning_theory"] = ft_data["learning_theory"]
                if "teaching_strategy" in ft_sections:
                    current_plan["teaching_strategy"] = ft_data["teaching_strategy"]

            # Run activities agent only if needed
            if needs_activities:
                act_prompt = build_activities_prompt(
                    current_plan,
                    activities_only=activities_only,
                    assessments_only=assessments_only,
                )
                act_result = await Runner.run(activities_generator, act_prompt)
                act_text = str(act_result.final_output) if act_result.final_output else ""
                act_data = parse_activities_output(act_text)

                if act_data:
                    # Surgically update only what changed
                    if not assessments_only:
                        current_plan["time_allocation"] = act_data.get("time_allocation", current_plan["time_allocation"])
                        current_plan["activities"] = act_data.get("activities", current_plan["activities"])
                        current_plan["time_summary"] = act_data.get("time_summary", current_plan["time_summary"])
                    if not activities_only:
                        current_plan["assessments"] = act_data.get("assessments", current_plan["assessments"])
                        current_plan["time_summary"] = act_data.get("time_summary", current_plan["time_summary"])

            await self._save_plan(thread.id, current_plan, context)

            markdown_output = plan_to_markdown(current_plan)
            display_input = [{"role": "user", "content": f"Output this lesson plan exactly as written, no changes:\n\n{markdown_output}"}]
            display_agent = Agent(
                name="Display",
                instructions="Output the content exactly as provided. Do not change, summarize, or add anything.",
                model="gpt-4.1-mini",
                model_settings=ModelSettings(temperature=0, max_tokens=4096, store=False)
            )
            result = Runner.run_streamed(display_agent, display_input)
            async for event in stream_agent_response(agent_context, result):
                yield event
            return

        # Fallback
        result = Runner.run_streamed(get_info_agent, agent_input)
        async for event in stream_agent_response(agent_context, result):
            yield event


StarterChatServer = LessonPlannerServer