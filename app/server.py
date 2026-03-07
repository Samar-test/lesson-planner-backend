"""ChatKit server - Conversational Lesson Planner (Refactored)."""

from __future__ import annotations

import json
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
# This is the single source of truth stored between turns.
# Keys map directly to the sections of the lesson plan.
EMPTY_PLAN: dict[str, Any] = {
    "domain": "",
    "course_title": "",
    "topic": "",
    "duration": "",
    "learner_level": "",
    "objectives": [],           # list of strings
    "learning_theory": {"name": "", "justification": ""},
    "teaching_strategy": {"name": "", "justification": ""},
    "time_allocation": {},      # raw dict from activities agent
    "activities": [],           # list of dicts
    "assessments": [],          # list of dicts
    "time_summary": {},         # raw dict from activities agent
}

# Memory key for storing lesson plan JSON per thread
PLAN_KEY = "lesson_plan"


# ── What each changed element cascades to ────────────────────────────────
# "ft_sections"        → sections the ft model must regenerate
# "needs_activities"   → whether activities+assessments must be regenerated

CASCADE_RULES: dict[str, dict] = {
    "topic": {
        "ft_sections": ["objectives", "learning_theory", "teaching_strategy"],
        "needs_activities": True,
    },
    "learner_level": {
        "ft_sections": ["objectives", "teaching_strategy"],
        "needs_activities": True,
    },
    "duration": {
        "ft_sections": [],
        "needs_activities": True,
    },
    "objectives": {
        "ft_sections": ["teaching_strategy"],
        "needs_activities": True,
    },
    "learning_theory": {
        "ft_sections": ["teaching_strategy"],
        "needs_activities": True,
    },
    "teaching_strategy": {
        "ft_sections": [],
        "needs_activities": True,
    },
    "activities": {
        "ft_sections": [],
        "needs_activities": True,   # activities_only flag handled separately
        "activities_only": True,
    },
    "assessments": {
        "ft_sections": [],
        "needs_activities": True,   # assessments_only flag handled separately
        "assessments_only": True,
    },
}


# ── Pydantic Schemas ──────────────────────────────────────────────────────

class OrchestratorOutput(BaseModel):
    intent: str                  # "new_lesson" | "modify" | "regenerate" | "get_info" | "other"
    changed_element: str         # only set when intent == "modify"
    has_all_details: bool
    domain: str
    course_title: str
    topic: str
    duration: str
    learner_level: str
    missing_fields: list[str]


class FTModelOutput(BaseModel):
    objectives: list[str]
    learning_theory_name: str
    learning_theory_justification: str
    teaching_strategy_name: str
    teaching_strategy_justification: str


# ── Agent 1: Orchestrator ─────────────────────────────────────────────────
# Single agent that does intent detection + info extraction in one shot.
# gpt-4.1 is reliable, fast, and follows structured output perfectly.

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


# ── Agent 2: FT Model (kept exactly as-is in spirit) ─────────────────────
# Fine-tuned model handles: objectives, learning_theory, teaching_strategy
# We prompt it to return structured JSON so we can surgically update the plan.

ft_model_agent = Agent(
    name="FT_lesson_planner",
    instructions="""You are a cybersecurity lesson plan expert.
Given the lesson details, generate the requested sections.

You will receive a message telling you EXACTLY which sections to generate and what the current lesson details are.

Always respond with valid JSON in this exact format:
{
  "objectives": ["objective 1", "objective 2", "objective 3"],
  "learning_theory_name": "...",
  "learning_theory_justification": "...",
  "teaching_strategy_name": "...",
  "teaching_strategy_justification": "..."
}

If a section is marked as KEEP, copy it exactly from the provided current plan.
If a section is marked as REGENERATE, create new high-quality content for it.
Always generate exactly 3 learning objectives.""",
    model="ft:gpt-3.5-turbo-1106:kau:lesson-plan2:CDfU4BQj",
    output_type=FTModelOutput,
    model_settings=ModelSettings(temperature=1, top_p=1, max_tokens=1024, store=True)
)


# ── Agent 3: Activities + Assessments Generator ───────────────────────────
# Receives the COMPLETE current plan as clean context. No guessing.

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

Respond ONLY with valid JSON in this exact format:
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
}

No extra text. No markdown. Pure JSON only.""",
    model="gpt-4.1",
    tools=[web_search_preview],
    model_settings=ModelSettings(temperature=0.7, max_tokens=4096, store=True)
)


# ── Agent 4: General / Get Info ───────────────────────────────────────────

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
    prompt_lines = [
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
            prompt_lines.append(f"REGENERATE {section.upper()}: Create new content for this section.")
        else:
            # Tell it exactly what to keep
            if section == "objectives" and plan.get("objectives"):
                prompt_lines.append(f"KEEP OBJECTIVES: {json.dumps(plan['objectives'])}")
            elif section == "learning_theory" and plan.get("learning_theory", {}).get("name"):
                lt = plan["learning_theory"]
                prompt_lines.append(f"KEEP LEARNING_THEORY: Name={lt['name']}, Justification={lt['justification']}")
            elif section == "teaching_strategy" and plan.get("teaching_strategy", {}).get("name"):
                ts = plan["teaching_strategy"]
                prompt_lines.append(f"KEEP TEACHING_STRATEGY: Name={ts['name']}, Justification={ts['justification']}")
            else:
                prompt_lines.append(f"REGENERATE {section.upper()}: Create new content (no existing content found).")

    return "\n".join(prompt_lines)


def build_activities_prompt(
    plan: dict[str, Any],
    activities_only: bool = False,
    assessments_only: bool = False
) -> str:
    """Build a clean, explicit prompt for the activities agent."""
    prompt_lines = [
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
        prompt_lines.append(f"{i}. {obj}")

    lt = plan.get("learning_theory", {})
    ts = plan.get("teaching_strategy", {})
    prompt_lines.append(f"\nLearning Theory: {lt.get('name', '')}")
    prompt_lines.append(f"Teaching Strategy: {ts.get('name', '')}")
    prompt_lines.append("")

    if activities_only:
        prompt_lines.append("INSTRUCTION: Regenerate ONLY the activities. Keep assessments the same.")
        prompt_lines.append("KEEP THESE ASSESSMENTS (copy exactly into your JSON response):")
        prompt_lines.append(json.dumps(plan.get("assessments", [])))
        prompt_lines.append("KEEP THIS TIME ALLOCATION (copy exactly, only recalculate if duration changed):")
        prompt_lines.append(json.dumps(plan.get("time_allocation", {})))
    elif assessments_only:
        prompt_lines.append("INSTRUCTION: Regenerate ONLY the assessments. Keep activities the same.")
        prompt_lines.append("KEEP THESE ACTIVITIES (copy exactly into your JSON response):")
        prompt_lines.append(json.dumps(plan.get("activities", [])))
        prompt_lines.append("KEEP THIS TIME ALLOCATION (copy exactly):")
        prompt_lines.append(json.dumps(plan.get("time_allocation", {})))
    else:
        prompt_lines.append("INSTRUCTION: Regenerate BOTH activities and assessments.")

    return "\n".join(prompt_lines)


# ── Server ────────────────────────────────────────────────────────────────

class LessonPlannerServer(ChatKitServer[dict[str, Any]]):
    def __init__(self) -> None:
        self.store: MemoryStore = MemoryStore()
        super().__init__(self.store)

    async def _load_plan(self, thread_id: str, context: dict[str, Any]) -> dict[str, Any]:
        """Load the current lesson plan JSON from the store."""
        try:
            raw = await self.store.get(f"{thread_id}:{PLAN_KEY}", context)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return dict(EMPTY_PLAN)

    async def _save_plan(self, thread_id: str, plan: dict[str, Any], context: dict[str, Any]) -> None:
        """Save the current lesson plan JSON to the store."""
        await self.store.set(f"{thread_id}:{PLAN_KEY}", json.dumps(plan), context)

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

        # Load the current lesson plan from structured storage (source of truth)
        current_plan = await self._load_plan(thread.id, context)

        # ── Step 1: Orchestrator — one call, gets intent + lesson details ──
        orch_result = await Runner.run(orchestrator_agent, agent_input)
        info: OrchestratorOutput = orch_result.final_output

        intent = info.intent.strip().lower()
        changed_element = info.changed_element.strip().lower()

        # ── Step 2: Route based on intent ────────────────────────────────

        # ── CASE: Other / General question ──
        if intent == "other":
            result = Runner.run_streamed(get_info_agent, agent_input)
            async for event in stream_agent_response(agent_context, result):
                yield event
            return

        # ── CASE: Get info — missing details ──
        if intent == "get_info" or (intent in ("new_lesson", "regenerate") and not info.has_all_details):
            result = Runner.run_streamed(get_info_agent, agent_input)
            async for event in stream_agent_response(agent_context, result):
                yield event
            return

        # ── CASE: New lesson or full regenerate ──
        if intent in ("new_lesson", "regenerate"):
            # Update plan metadata from orchestrator
            current_plan["domain"] = info.domain
            current_plan["course_title"] = info.course_title
            current_plan["topic"] = info.topic
            current_plan["duration"] = info.duration
            current_plan["learner_level"] = info.learner_level

            # FT model generates all three sections
            ft_prompt = build_ft_prompt(
                current_plan,
                sections_to_regenerate=["objectives", "learning_theory", "teaching_strategy"],
                info=info
            )
            ft_result = await Runner.run(ft_model_agent, ft_prompt)
            ft_output: FTModelOutput = ft_result.final_output

            # Update plan with ft model output
            current_plan["objectives"] = ft_output.objectives
            current_plan["learning_theory"] = {
                "name": ft_output.learning_theory_name,
                "justification": ft_output.learning_theory_justification,
            }
            current_plan["teaching_strategy"] = {
                "name": ft_output.teaching_strategy_name,
                "justification": ft_output.teaching_strategy_justification,
            }

            # Activities agent generates activities + assessments
            act_prompt = build_activities_prompt(current_plan)
            act_result = await Runner.run(activities_generator, act_prompt)

            # Parse activities JSON response
            act_text = ""
            if act_result.final_output:
                act_text = str(act_result.final_output)
            else:
                for msg in act_result.messages:
                    if hasattr(msg, "content"):
                        act_text += str(msg.content)

            try:
                act_text_clean = act_text.strip()
                if act_text_clean.startswith("```"):
                    act_text_clean = act_text_clean.split("```")[1]
                    if act_text_clean.startswith("json"):
                        act_text_clean = act_text_clean[4:]
                act_data = json.loads(act_text_clean)
                current_plan["time_allocation"] = act_data.get("time_allocation", {})
                current_plan["activities"] = act_data.get("activities", [])
                current_plan["assessments"] = act_data.get("assessments", [])
                current_plan["time_summary"] = act_data.get("time_summary", {})
            except (json.JSONDecodeError, Exception):
                # If JSON parsing fails, store raw text and surface error
                current_plan["activities"] = []
                current_plan["assessments"] = []

            # Save updated plan to store
            await self._save_plan(thread.id, current_plan, context)

            # Stream the final markdown to the user
            markdown_output = plan_to_markdown(current_plan)
            final_input = [{"role": "user", "content": f"Display this lesson plan:\n\n{markdown_output}"}]

            display_agent = Agent(
                name="Display",
                instructions="Output the lesson plan exactly as provided, with no changes or additions.",
                model="gpt-4.1-mini",
                model_settings=ModelSettings(temperature=0, max_tokens=4096, store=False)
            )
            result = Runner.run_streamed(display_agent, final_input)
            async for event in stream_agent_response(agent_context, result):
                yield event
            return

        # ── CASE: Modify ──
        if intent == "modify" and changed_element:
            cascade = CASCADE_RULES.get(changed_element, {
                "ft_sections": [],
                "needs_activities": True,
            })

            # Update lesson metadata if it changed
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

            # ── Run FT model if needed ──
            if ft_sections:
                ft_prompt = build_ft_prompt(current_plan, ft_sections, info)
                ft_result = await Runner.run(ft_model_agent, ft_prompt)
                ft_output: FTModelOutput = ft_result.final_output

                # Surgically update only the sections that were regenerated
                if "objectives" in ft_sections:
                    current_plan["objectives"] = ft_output.objectives
                if "learning_theory" in ft_sections:
                    current_plan["learning_theory"] = {
                        "name": ft_output.learning_theory_name,
                        "justification": ft_output.learning_theory_justification,
                    }
                if "teaching_strategy" in ft_sections:
                    current_plan["teaching_strategy"] = {
                        "name": ft_output.teaching_strategy_name,
                        "justification": ft_output.teaching_strategy_justification,
                    }

            # ── Run activities agent if needed ──
            if needs_activities:
                act_prompt = build_activities_prompt(
                    current_plan,
                    activities_only=activities_only,
                    assessments_only=assessments_only
                )
                act_result = await Runner.run(activities_generator, act_prompt)

                act_text = ""
                if act_result.final_output:
                    act_text = str(act_result.final_output)
                else:
                    for msg in act_result.messages:
                        if hasattr(msg, "content"):
                            act_text += str(msg.content)

                try:
                    act_text_clean = act_text.strip()
                    if act_text_clean.startswith("```"):
                        act_text_clean = act_text_clean.split("```")[1]
                        if act_text_clean.startswith("json"):
                            act_text_clean = act_text_clean[4:]
                    act_data = json.loads(act_text_clean)

                    # Surgically update only what changed
                    if not assessments_only:
                        current_plan["time_allocation"] = act_data.get("time_allocation", current_plan["time_allocation"])
                        current_plan["activities"] = act_data.get("activities", current_plan["activities"])
                        current_plan["time_summary"] = act_data.get("time_summary", current_plan["time_summary"])
                    if not activities_only:
                        current_plan["assessments"] = act_data.get("assessments", current_plan["assessments"])
                        current_plan["time_summary"] = act_data.get("time_summary", current_plan["time_summary"])

                except (json.JSONDecodeError, Exception):
                    pass  # Keep existing activities/assessments if parsing fails

            # Save updated plan
            await self._save_plan(thread.id, current_plan, context)

            # Stream the final markdown
            markdown_output = plan_to_markdown(current_plan)
            final_input = [{"role": "user", "content": f"Display this lesson plan:\n\n{markdown_output}"}]

            display_agent = Agent(
                name="Display",
                instructions="Output the lesson plan exactly as provided, with no changes or additions.",
                model="gpt-4.1-mini",
                model_settings=ModelSettings(temperature=0, max_tokens=4096, store=False)
            )
            result = Runner.run_streamed(display_agent, final_input)
            async for event in stream_agent_response(agent_context, result):
                yield event
            return

        # ── Fallback ──
        result = Runner.run_streamed(get_info_agent, agent_input)
        async for event in stream_agent_response(agent_context, result):
            yield event


StarterChatServer = LessonPlannerServer