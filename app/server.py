from __future__ import annotations
import copy
import json
import re
from typing import Any, AsyncIterator
from agents import Runner, Agent, ModelSettings
from chatkit.agents import AgentContext, simple_to_agent_input, stream_agent_response
from chatkit.server import ChatKitServer
from chatkit.types import ThreadMetadata, ThreadStreamEvent, UserMessageItem
from openai import AsyncOpenAI
from .memory_store import MemoryStore

MAX_RECENT_ITEMS = 30
FT_MODEL = "ft:gpt-3.5-turbo-1106:kau:lesson-plan2:CDfU4BQj"
GPT4 = "gpt-4.1"
openai_client = AsyncOpenAI()


# ── Lesson Plan State ─────────────────────────────────────────────────────

def empty_plan() -> dict[str, Any]:
    return {
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

# ── Cascade Rules (exactly per client spec) ───────────────────────────────

CASCADE_RULES = {
    "topic": {
        "regenerate_via_ft": True,           # full ft regeneration
        "ft_keeps": [],                       # ft regenerates everything
        "regenerate_activities": True,
        "activities_only": False,
        "assessments_only": False,
    },
    "learner_level": {
        "regenerate_via_ft": True,
        "ft_keeps": ["learning_theory"],      # keep learning theory, regenerate objectives + strategy
        "regenerate_activities": True,
        "activities_only": False,
        "assessments_only": False,
    },
    "duration": {
        "regenerate_via_ft": False,
        "ft_keeps": [],
        "regenerate_activities": True,
        "activities_only": False,
        "assessments_only": False,
    },
    "objectives": {
        "regenerate_via_ft": False,           # gpt-4.1 handles objective changes
        "ft_keeps": [],
        "regenerate_activities": True,
        "activities_only": False,
        "assessments_only": False,
        "gpt4_regenerates": ["objectives", "teaching_strategy"],
    },
    "learning_theory": {
        "regenerate_via_ft": False,
        "ft_keeps": [],
        "regenerate_activities": True,
        "activities_only": False,
        "assessments_only": False,
        "gpt4_regenerates": ["learning_theory", "teaching_strategy"],
    },
    "teaching_strategy": {
        "regenerate_via_ft": False,
        "ft_keeps": [],
        "regenerate_activities": True,
        "activities_only": False,
        "assessments_only": False,
        "gpt4_regenerates": ["teaching_strategy"],
    },
    "activities": {
        "regenerate_via_ft": False,
        "ft_keeps": [],
        "regenerate_activities": True,
        "activities_only": True,
        "assessments_only": False,
    },
    "assessments": {
        "regenerate_via_ft": False,
        "ft_keeps": [],
        "regenerate_activities": True,
        "activities_only": False,
        "assessments_only": True,
    },
}

# ══════════════════════════════════════════════════════════════════════════
# NODE 1: Orchestrator
# Detects intent and extracts lesson details in one call.
# ══════════════════════════════════════════════════════════════════════════

async def node_orchestrator(conversation: list[dict]) -> dict[str, Any]:
    """
    Analyzes conversation and returns:
    - intent: new_lesson | regenerate | modify | get_info | other
    - changed_element: which section to modify (if intent=modify)
    - lesson details extracted from full conversation
    """
    system = """You are the orchestrator of a cybersecurity lesson planner.
Analyze the conversation and return a JSON object with these exact fields:

{
  "intent": "new_lesson" | "regenerate" | "modify" | "get_info" | "other",
  "changed_element": "",
  "has_all_details": true | false,
  "domain": "...",
  "course_title": "...",
  "topic": "...",
  "duration": "...",
  "learner_level": "...",
  "missing_fields": []
}

The ONLY 5 fields needed to generate a lesson plan are:
1. domain (e.g. Cybersecurity)
2. course_title (e.g. Intro to Network Defense)
3. topic (e.g. Network Security)
4. duration (e.g. 60)
5. learner_level (e.g. Beginner / Intermediate / Advanced)

has_all_details = true ONLY if all 5 fields above are present in the conversation.
missing_fields = list of whichever of the 5 fields are missing.

DO NOT require objectives, activities, assessments, or any other fields.
The system generates those automatically.

INTENT RULES:
- "new_lesson": teacher provides lesson details for the first time
- "regenerate": vague dissatisfaction with ENTIRE plan, no specific section mentioned
- "modify": teacher wants to change a SPECIFIC section → set changed_element
- "get_info": one or more of the 5 fields above are missing
- "other": general question

Return ONLY valid JSON. No extra text."""

    messages = [{"role": "system", "content": system}]
    messages += conversation

    response = await openai_client.chat.completions.create(
        model=GPT4,
        messages=messages,
        temperature=0,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


# ══════════════════════════════════════════════════════════════════════════
# NODE 2: FT Model — Fresh generation only
# Receives JSON with 5 fields, outputs markdown lesson plan header
# ══════════════════════════════════════════════════════════════════════════

async def node_ft_model(plan: dict[str, Any]) -> str:
    """
    Calls the fine-tuned model for fresh lesson generation.
    Input: JSON with domain, course_title, topic, duration, learner_level
    Output: markdown string with Lesson Info, Objectives, Theory, Strategy
    """
    ft_input = json.dumps({
        "domain": plan["domain"],
        "course_title": plan["course_title"],
        "topic": plan["topic"],
        "duration": plan["duration"],
        "learner_level": plan["learner_level"],
    })

    system = """You receive the teacher's lesson information from the previous conversation.

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
- Justification:"""

    response = await openai_client.chat.completions.create(
        model=FT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": ft_input},
        ],
        temperature=1,
        top_p=1,
        max_tokens=2048,
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════════════════════════════════
# NODE 3: GPT-4.1 Modifier — For targeted section changes
# Understands the user's specific modification request
# ══════════════════════════════════════════════════════════════════════════

async def node_gpt4_modifier(
    plan: dict[str, Any],
    sections_to_regenerate: list[str],
    user_instruction: str,
) -> dict[str, Any]:
    """
    Uses gpt-4.1 to modify specific sections based on user's instruction.
    Returns only the sections that changed.
    """
    system = f"""You are a cybersecurity lesson plan expert.
The teacher has requested a specific change to their lesson plan.

Teacher's request: "{user_instruction}"

Current lesson plan:
- Domain: {plan['domain']}
- Course title: {plan['course_title']}
- Topic: {plan['topic']}
- Duration: {plan['duration']} minutes
- Learner level: {plan['learner_level']}
- Current objectives: {json.dumps(plan['objectives'])}
- Current learning theory: {plan['learning_theory']['name']}
- Current teaching strategy: {plan['teaching_strategy']['name']}

You must REGENERATE these sections: {', '.join(sections_to_regenerate)}
You must KEEP all other sections exactly as they are.

Respond with ONLY a JSON object in this exact format:
{{
  "objectives": ["objective 1", "objective 2", "objective 3"],
  "learning_theory_name": "...",
  "learning_theory_justification": "...",
  "teaching_strategy_name": "...",
  "teaching_strategy_justification": "..."
}}

Rules:
- Always include all 5 keys in the JSON
- For sections you are KEEPING: copy the current values exactly
- For sections you are REGENERATING: create new content that reflects the teacher's request
- objectives must always be a list of exactly 3 strings
- Return ONLY the JSON, no extra text"""

    response = await openai_client.chat.completions.create(
        model=GPT4,
        messages=[{"role": "user", "content": system}],
        temperature=0.7,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


# ══════════════════════════════════════════════════════════════════════════
# NODE 4: Activities & Assessments Generator
# ══════════════════════════════════════════════════════════════════════════

async def node_activities_generator(
    plan: dict[str, Any],
    activities_only: bool = False,
    assessments_only: bool = False,
) -> dict[str, Any]:
    """
    Generates activities and/or assessments based on current plan state.
    Returns JSON with time_allocation, activities, assessments, time_summary.
    """
    keep_activities_msg = ""
    keep_assessments_msg = ""

    if activities_only:
        keep_assessments_msg = f"\nKEEP THESE ASSESSMENTS exactly (copy into your response):\n{json.dumps(plan.get('assessments', []))}"
        keep_activities_msg = f"\nKEEP THIS TIME ALLOCATION exactly:\n{json.dumps(plan.get('time_allocation', {}))}"
        instruction = "Regenerate ONLY the activities. Keep assessments unchanged."
    elif assessments_only:
        keep_activities_msg = f"\nKEEP THESE ACTIVITIES exactly (copy into your response):\n{json.dumps(plan.get('activities', []))}"
        keep_assessments_msg = f"\nKEEP THIS TIME ALLOCATION exactly:\n{json.dumps(plan.get('time_allocation', {}))}"
        instruction = "Regenerate ONLY the assessments. Keep activities unchanged."
    else:
        instruction = "Regenerate BOTH activities and assessments."

    prompt = f"""You are an expert in cybersecurity education and instructional design.

LESSON PLAN:
- Domain: {plan.get('domain', '')}
- Course title: {plan.get('course_title', '')}
- Topic: {plan.get('topic', '')}
- Duration: {plan.get('duration', '')} minutes
- Learner level: {plan.get('learner_level', '')}

Learning Objectives (use the FULL TEXT of each objective, not numbers):
{chr(10).join(f"{i+1}. {obj}" for i, obj in enumerate(plan.get('objectives', [])))}

Learning Theory: {plan.get('learning_theory', {}).get('name', '')}
Teaching Strategy: {plan.get('teaching_strategy', {}).get('name', '')}

INSTRUCTION: {instruction}
{keep_activities_msg}
{keep_assessments_msg}

TIME RULES (STRICTLY ENFORCED — violations will break the lesson plan):
- Total duration = {plan.get('duration', '')} minutes
- Lecture time = 65% of total duration, rounded to nearest minute
- Remaining time = total duration - lecture time
- Activities budget = 70% of remaining time, rounded to nearest minute
- Assessments budget = 30% of remaining time, rounded to nearest minute
- Sum of all activity times MUST equal activities budget EXACTLY
- Sum of all assessment times MUST equal assessments budget EXACTLY
- Grand total in time_summary MUST equal remaining time EXACTLY
- NEVER produce an activity or assessment with 0 minutes
- Every activity and assessment MUST have time_minutes >= 1

NUMBER OF ITEMS based on remaining time:
- Remaining LESS THAN 15 min (e.g. 10 min): EXACTLY 2 activities, EXACTLY 2 assessments. NOT 3.
- Remaining BETWEEN 15 AND 25 min inclusive (e.g. 21 min): EXACTLY 3 activities, EXACTLY 3 assessments. NOT 4.
- Remaining MORE THAN 25 min (e.g. 31 min): EXACTLY 4 activities, EXACTLY 3 assessments. NOT 3.

EXAMPLE: If remaining = 10 minutes → 2 activities + 2 assessments only.
EXAMPLE: If remaining = 21 minutes → 3 activities + 3 assessments only.
EXAMPLE: If remaining = 31 minutes → 4 activities + 3 assessments only.

STEP 1: Calculate remaining time first.
STEP 2: Select the correct number of activities and assessments from the table above.
STEP 3: You MUST generate exactly that number — no more, no less.
STEP 4: Distribute the time budget evenly across the exact number of items.

Current remaining time = total_duration - lecture_minutes
If remaining is between 15 and 25 (inclusive): generate EXACTLY 3 activities. NOT 4.
If remaining is above 25: generate EXACTLY 4 activities. NOT 3.

CRITICAL — aligned_objectives field:
- ALWAYS use the full text of the objective, copied exactly from the Learning Objectives list above
- NEVER use numbers like "1", "2", "3" — always use the full sentence

SELF-CHECK before responding:
1. Add up all activity time_minutes — does it equal activities_budget_minutes?
2. Add up all assessment time_minutes — does it equal assessments_budget_minutes?
3. Does activities + assessments = remaining_minutes?
4. Are all aligned_objectives full text strings, not numbers?
5. Does any activity or assessment have time_minutes = 0? If yes, fix it.
6. Count your activities — does the count match the NUMBER OF ITEMS rule above?
7. Count your assessments — does the count match the NUMBER OF ITEMS rule above?
If any check fails, fix it before responding.

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "time_allocation": {{
    "total_duration_minutes": 0,
    "lecture_minutes": 0,
    "remaining_minutes": 0,
    "activities_budget_minutes": 0,
    "assessments_budget_minutes": 0
  }},
  "activities": [
    {{
      "title": "...",
      "aligned_objectives": ["full objective text here"],
      "description": "...",
      "steps": ["..."],
      "time_minutes": 0
    }}
  ],
  "assessments": [
    {{
      "title": "...",
      "aligned_objectives": ["full objective text here"],
      "format": "...",
      "description": "...",
      "time_minutes": 0
    }}
  ],
  "time_summary": {{
    "total_activity_time": 0,
    "total_assessment_time": 0,
    "grand_total": 0
  }}
}}"""

    response = await openai_client.chat.completions.create(
        model=GPT4,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


# ══════════════════════════════════════════════════════════════════════════
# NODE 5: Get Info Agent
# ══════════════════════════════════════════════════════════════════════════

async def node_get_info(conversation: list[dict], missing_fields: list[str]) -> str:
    system = f"""You are a helpful cybersecurity lesson planner assistant.
The teacher needs to provide the following missing information: {', '.join(missing_fields)}

Ask only for what's missing. Be concise and friendly."""

    messages = [{"role": "system", "content": system}] + conversation
    response = await openai_client.chat.completions.create(
        model=GPT4,
        messages=messages,
        temperature=1,
        max_tokens=512,
    )
    return response.choices[0].message.content


# ══════════════════════════════════════════════════════════════════════════
# PARSERS
# ══════════════════════════════════════════════════════════════════════════

def parse_ft_output(text: str) -> dict[str, Any]:
    """Parse ft model markdown output into structured dict."""
    result = {
        "objectives": [],
        "learning_theory": {"name": "", "justification": ""},
        "teaching_strategy": {"name": "", "justification": ""},
    }

    if not text:
        return result

    # Objectives: only lines between ### Learning Objectives and ### Learning Theory
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


def apply_gpt4_modifier_output(plan: dict[str, Any], data: dict, sections: list[str]) -> None:
    """Surgically apply gpt-4.1 modifier output to plan."""
    if "objectives" in sections:
        plan["objectives"] = data.get("objectives", plan["objectives"])
    if "learning_theory" in sections:
        plan["learning_theory"] = {
            "name": data.get("learning_theory_name", plan["learning_theory"]["name"]),
            "justification": data.get("learning_theory_justification", plan["learning_theory"]["justification"]),
        }
    if "teaching_strategy" in sections:
        plan["teaching_strategy"] = {
            "name": data.get("teaching_strategy_name", plan["teaching_strategy"]["name"]),
            "justification": data.get("teaching_strategy_justification", plan["teaching_strategy"]["justification"]),
        }


def apply_activities_output(
    plan: dict[str, Any],
    data: dict,
    activities_only: bool,
    assessments_only: bool,
) -> None:
    """Surgically apply activities generator output to plan."""
    if not data:
        return
    if not assessments_only:
        plan["time_allocation"] = data.get("time_allocation", plan["time_allocation"])
        plan["activities"] = data.get("activities", plan["activities"])
        plan["time_summary"] = data.get("time_summary", plan["time_summary"])
    if not activities_only:
        plan["assessments"] = data.get("assessments", plan["assessments"])
        plan["time_summary"] = data.get("time_summary", plan["time_summary"])


# ══════════════════════════════════════════════════════════════════════════
# RENDERER
# ══════════════════════════════════════════════════════════════════════════

def render_plan(plan: dict[str, Any]) -> str:
    """Convert plan dict to formatted markdown string."""
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

    ts_s = plan.get("time_summary", {})
    if ts_s:
        lines.append("### Time Summary")
        lines.append(f"- Total Activity Time: {ts_s.get('total_activity_time', '')} minutes")
        lines.append(f"- Total Assessment Time: {ts_s.get('total_assessment_time', '')} minutes")
        lines.append(f"- Grand Total: {ts_s.get('grand_total', '')} minutes")
        lines.append("")

    lines.append("---")
    lines.append("✅ Lesson plan complete! You can:")
    lines.append("- Ask me to **regenerate** this lesson plan")
    lines.append("- Ask me to **modify** a specific section")
    lines.append("- Provide details for a **new lesson plan**")

    return "\n".join(lines)



class LessonPlannerServer(ChatKitServer[dict[str, Any]]):
    def __init__(self) -> None:
        self.store: MemoryStore = MemoryStore()
        super().__init__(self.store)

    async def _load_plan(self, thread_id: str, context: dict[str, Any]) -> dict[str, Any]:
        try:
            thread = await self.store.load_thread(thread_id, context)
            if thread.metadata and "lesson_plan" in thread.metadata:
                return thread.metadata["lesson_plan"]
        except Exception:
            pass
        return empty_plan()

    async def _save_plan(self, thread_id: str, plan: dict[str, Any], context: dict[str, Any]) -> None:
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

        items_page = await self.store.load_thread_items(
            thread.id, after=None, limit=MAX_RECENT_ITEMS, order="desc", context=context,
        )
        items = list(reversed(items_page.data))
        agent_input = await simple_to_agent_input(items)

        agent_context = AgentContext(
            thread=thread, store=self.store, request_context=context,
        )

        current_plan = await self._load_plan(thread.id, context)

        # Build conversation for node functions
        #conversation = [{"role": msg["role"], "content": msg["content"]} for msg in agent_input if isinstance(msg, dict)]
        # Build conversation for node functions
        conversation = []
        for msg in agent_input:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            # content can be a string or a list of content blocks from ChatKit
            if isinstance(content, list):
                # Extract text from content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        # handles input_text, text, and other block types
                        text_parts.append(
                            block.get("text") or
                            block.get("input_text") or
                            ""
                        )
                content = " ".join(text_parts).strip()
            if role and content:
                conversation.append({"role": role, "content": content})

        # Step 1: Orchestrator
        orch = await node_orchestrator(conversation)
        intent = orch.get("intent", "other").strip().lower()
        changed_element = orch.get("changed_element", "").lower().strip()
        changed_element = changed_element.replace("learning objectives", "objectives")
        changed_element = changed_element.replace("learning theory", "learning_theory")
        changed_element = changed_element.replace("teaching strategy", "teaching_strategy")
        changed_element = changed_element.replace("learner level", "learner_level")
        changed_element = changed_element.replace(" ", "_")

        # Step 2: Route
        if intent == "other":
            response = await openai_client.chat.completions.create(
                model=GPT4,
                messages=[{"role": "system", "content": "You are a helpful cybersecurity education assistant. Answer briefly."}] + conversation,
                temperature=1, max_tokens=512,
            )
            text = response.choices[0].message.content
        elif intent == "get_info" or (intent in ("new_lesson", "regenerate") and not orch.get("has_all_details")):
            missing = orch.get("missing_fields", ["domain", "course_title", "topic", "duration", "learner_level"])
            system = f"You are a helpful cybersecurity lesson planner. Ask only for these missing fields: {', '.join(missing)}. Be concise and friendly."
            response = await openai_client.chat.completions.create(
                model=GPT4,
                messages=[{"role": "system", "content": system}] + conversation,
                temperature=1, max_tokens=512,
            )
            text = response.choices[0].message.content
        elif intent in ("new_lesson", "regenerate"):
            current_plan["domain"] = orch["domain"]
            current_plan["course_title"] = orch["course_title"]
            current_plan["topic"] = orch["topic"]
            current_plan["duration"] = orch["duration"]
            current_plan["learner_level"] = orch["learner_level"]

            ft_text = await node_ft_model(current_plan)
            ft_data = parse_ft_output(ft_text)
            current_plan["objectives"] = ft_data["objectives"]
            current_plan["learning_theory"] = ft_data["learning_theory"]
            current_plan["teaching_strategy"] = ft_data["teaching_strategy"]

            act_data = await node_activities_generator(current_plan)
            apply_activities_output(current_plan, act_data, False, False)

            await self._save_plan(thread.id, current_plan, context)
            text = render_plan(current_plan)

        elif intent == "modify" and changed_element:
            cascade = CASCADE_RULES.get(changed_element, {
                "regenerate_via_ft": False,
                "ft_keeps": [],
                "regenerate_activities": True,
                "activities_only": False,
                "assessments_only": False,
                "gpt4_regenerates": [],
            })

            if changed_element == "topic":
                current_plan["topic"] = orch["topic"]
            elif changed_element == "learner_level":
                current_plan["learner_level"] = orch["learner_level"]
            elif changed_element == "duration":
                current_plan["duration"] = orch["duration"]

            user_instruction = ""
            for msg in reversed(conversation):
                if msg["role"] == "user":
                    user_instruction = msg["content"]
                    break

            if cascade.get("regenerate_via_ft"):
                ft_text = await node_ft_model(current_plan)
                ft_data = parse_ft_output(ft_text)
                ft_keeps = cascade.get("ft_keeps", [])
                if "objectives" not in ft_keeps:
                    current_plan["objectives"] = ft_data["objectives"]
                if "learning_theory" not in ft_keeps:
                    current_plan["learning_theory"] = ft_data["learning_theory"]
                if "teaching_strategy" not in ft_keeps:
                    current_plan["teaching_strategy"] = ft_data["teaching_strategy"]

            gpt4_sections = cascade.get("gpt4_regenerates", [])
            if gpt4_sections:
                modifier_data = await node_gpt4_modifier(current_plan, gpt4_sections, user_instruction)
                apply_gpt4_modifier_output(current_plan, modifier_data, gpt4_sections)

            if cascade.get("regenerate_activities"):
                activities_only = cascade.get("activities_only", False)
                assessments_only = cascade.get("assessments_only", False)
                act_data = await node_activities_generator(current_plan, activities_only, assessments_only)
                apply_activities_output(current_plan, act_data, activities_only, assessments_only)

            await self._save_plan(thread.id, current_plan, context)
            text = render_plan(current_plan)

        else:
            missing = ["domain", "course_title", "topic", "duration", "learner_level"]
            system = f"You are a helpful cybersecurity lesson planner. Ask only for these missing fields: {', '.join(missing)}. Be concise and friendly."
            response = await openai_client.chat.completions.create(
                model=GPT4,
                messages=[{"role": "system", "content": system}] + conversation,
                temperature=1, max_tokens=512,
            )
            text = response.choices[0].message.content

        # Stream final output
        display_agent = Agent(
            name="Display",
            instructions="Output the content exactly as provided. Do not change, summarize, or add anything.",
            model="gpt-4.1-mini",
            model_settings=ModelSettings(temperature=0, max_tokens=4096, store=False)
        )
        result = Runner.run_streamed(display_agent, [{"role": "user", "content": text}])
        async for event in stream_agent_response(agent_context, result):
            yield event


StarterChatServer = LessonPlannerServer