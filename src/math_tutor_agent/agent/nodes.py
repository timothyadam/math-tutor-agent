"""各个节点：收集信息、意图识别、检索、校验、生成。

三条贯穿所有节点的规矩：

1. 节点签名是 ``(state, deps) -> dict``，只返回自己改动的键。
   返回整个 state 会把"谁改了什么"彻底搞糊涂，也让单测没法断言。
2. 能用确定性代码判断的，绝不调模型。collect_info 的后半段和 validate
   一次模型都不调 —— 它们每次都给同样的答案，这正是算槽位和过滤题目需要的
   性质，而且不花钱、不会超时、可以直接写断言。
3. 节点不自己创建模型客户端、不写 SQL。外部依赖统一从 deps 注入，
   测试时换成假对象，不用起数据库、不用联网。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import anthropic
from pydantic import BaseModel, Field

from . import prompts
from .state import REQUIRED_SLOTS, Intent, RetrievedItem, Slots, TutorState

MODEL = "claude-opus-5-5"

# 校验阈值放模块顶部，不埋在函数里：测试要调它，调参也要调它。
MIN_SCORE = 0.62       # 低于这个相似度的召回一律丢掉
MAX_PER_POINT = 2      # 同一知识点最多两道，避免整页都是同一类题
MAX_ITEMS = 5
MAX_RETRY = 1          # 放宽阈值重检索的次数上限，条件边靠它收敛


class Retriever(Protocol):
    """检索层的接口。

    节点只认这个协议，不认具体实现 —— 真实实现在 retrieval/retriever.py，
    测试里换成一个返回固定列表的假对象。Agent 节点不直接碰数据库：
    SQL、连接和事务都留在 db/ 和 retrieval/ 层。
    """

    def search(
        self, *, query: str, grade: int, edition: str, unit: str, limit: int
    ) -> list[dict]: ...


@dataclass(frozen=True)
class Deps:
    """图运行期的外部依赖，在 graph.py 里组装一次，绑给每个节点。"""

    retriever: Retriever
    client: anthropic.Anthropic


# ---- 结构化输出的返回类型 --------------------------------------------------
# 用 Pydantic 模型，而不是"让模型吐 JSON 再自己 json.loads"：
# output_format 会把 schema 下发给模型并在服务端约束生成，
# 返回的 parsed_output 已经是校验过的对象，省掉一整类解析失败的兜底代码。


class ExtractedSlots(BaseModel):
    grade: int | None = Field(default=None, ge=1, le=6)
    edition: str | None = None
    unit: str | None = None
    lesson: str | None = None


class IntentResult(BaseModel):
    intent: Intent


def _last_user_text(state: TutorState) -> str:
    for message in reversed(state.get("messages", [])):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            return json.dumps(content, ensure_ascii=False)
    return ""


# ---- 1. 收集信息 -----------------------------------------------------------


def collect_info(state: TutorState, deps: Deps) -> dict:
    """抽取槽位并算出还缺什么。

    只有"从自然语言里挑信息"这件事用模型（规则写不动），
    "还缺哪个槽位"是纯比对，不该再花一次调用。

    effort 调到 low：这是抽取任务，不需要模型深想。Opus 5.5 的 effort
    默认是 medium，不显式写就会按 medium 的深度花钱。
    """
    known: Slots = dict(state.get("slots") or {})  # type: ignore[assignment]

    response = deps.client.messages.parse(
        model=MODEL,
        max_tokens=1024,
        system=prompts.EXTRACT_SYSTEM,
        output_config={"effort": "low"},
        messages=state.get("messages", []),
        output_format=ExtractedSlots,
    )
    extracted = response.parsed_output

    # 只补空缺，不覆盖已确认的槽位：学生说过"三年级"之后，
    # 后面随口提到"我姐姐六年级"不该把年级改掉。
    for field in ("grade", "edition", "unit", "lesson"):
        value = getattr(extracted, field, None)
        if value is not None and not known.get(field):
            known[field] = value  # type: ignore[literal-required]

    missing = [slot for slot in REQUIRED_SLOTS if not known.get(slot)]
    return {"slots": known, "missing": missing}


def ask_missing(state: TutorState, deps: Deps) -> dict:
    """缺槽位时直接追问，不调模型。

    一次只问一个 —— 一口气问三个问题，小学生大概率只回一个，
    剩下两个还得再问一轮，反而更慢。话术来自 prompts，节点不拼字符串。
    """
    slot = (state.get("missing") or ["grade"])[0]
    question = prompts.SLOT_QUESTIONS[slot]
    return {"messages": [{"role": "assistant", "content": question}], "answer": question}


# ---- 2. 意图识别 -----------------------------------------------------------


def detect_intent(state: TutorState, deps: Deps) -> dict:
    """判断学生要什么。

    为什么不和 collect_info 合成一次调用？合是能合，能省一半调用。
    分开是因为两者的失败处理完全不同：槽位抽歪了要追问，意图判错了要走兜底。
    而且这一步输入输出都很小，是整条链里最适合换便宜模型的地方 ——
    真要降成本，只把这个节点换成 claude-haiku-5-5 即可，不影响讲解质量。
    这里统一用 Opus，换不换由你定。
    """
    response = deps.client.messages.parse(
        model=MODEL,
        max_tokens=256,
        system=prompts.INTENT_SYSTEM,
        output_config={"effort": "low"},
        messages=state.get("messages", []),
        output_format=IntentResult,
    )
    return {"intent": response.parsed_output.intent}


# ---- 3. 检索 ---------------------------------------------------------------


def retrieve(state: TutorState, deps: Deps) -> dict:
    """按槽位做带过滤的召回，并把结果裁成 State 装得下的大小。"""
    slots = state.get("slots") or {}
    attempt = state.get("attempt", 0)

    rows = deps.retriever.search(
        query=_last_user_text(state),
        grade=int(slots["grade"]),
        edition=str(slots["edition"]),
        unit=str(slots["unit"]),
        # 重试时多取候选，让放宽后的校验有东西可挑
        limit=20 if attempt else 10,
    )

    items: list[RetrievedItem] = [
        {
            "question_id": int(row["id"]),
            "stem": str(row["stem"])[:120],
            "knowledge_point": str(row["knowledge_point"]),
            "grade": int(row["grade"]),
            "score": float(row["score"]),
        }
        for row in rows
    ]
    return {"retrieved": items, "attempt": attempt + 1}


# ---- 4. 校验 ---------------------------------------------------------------


def validate(state: TutorState, deps: Deps) -> dict:
    """把不该给学生看的题挡在生成之前，全是确定性规则。

    这是"分年级约束"的第二道闸：提示词管住模型的嘴，这里管住喂给模型的料。
    只靠提示词不够 —— 一旦检索召回了一道六年级的题摆在上下文里，
    模型很容易顺着它讲超纲。
    """
    student_grade = int((state.get("slots") or {}).get("grade", 6))
    threshold = MIN_SCORE if state.get("attempt", 1) <= 1 else MIN_SCORE - 0.1

    kept: list[RetrievedItem] = []
    rejected: list[str] = []
    per_point: dict[str, int] = {}

    for item in state.get("retrieved") or []:
        qid, point = item["question_id"], item["knowledge_point"]
        if item["grade"] > student_grade:
            rejected.append(f"{qid}：题目是{item['grade']}年级的，超纲")
            continue
        if item["score"] < threshold:
            rejected.append(f"{qid}：相似度 {item['score']:.2f} 低于 {threshold:.2f}")
            continue
        if per_point.get(point, 0) >= MAX_PER_POINT:
            rejected.append(f"{qid}：知识点「{point}」已经够了")
            continue
        per_point[point] = per_point.get(point, 0) + 1
        kept.append(item)
        if len(kept) >= MAX_ITEMS:
            break

    return {"retrieved": kept, "rejected": rejected}


# ---- 5. 生成 ---------------------------------------------------------------


def generate(state: TutorState, deps: Deps) -> dict:
    """带着分年级约束和已校验的题目讲解。

    题目以"参考资料"的形式追加成最后一条 user 消息，而不是塞进 system：
    system 里只放稳定不变的角色和约束，这样它能被提示词缓存命中；
    每轮都在变的检索结果放在 messages 尾部，不会把前缀缓存打掉。
    """
    slots = state.get("slots") or {}
    system = prompts.tutor_system(
        grade=int(slots.get("grade", 6)),
        edition=str(slots.get("edition", "人教版")),
        unit=str(slots.get("unit", "")),
        lesson=slots.get("lesson"),
    )

    messages = list(state.get("messages", []))
    items = state.get("retrieved") or []
    if items:
        reference = "\n".join(
            f"- [{item['knowledge_point']}] {item['stem']}" for item in items
        )
        messages.append(
            {"role": "user", "content": f"课本同类题（只作参考，不要照抄）：\n{reference}"}
        )
    else:
        messages.append(
            {"role": "user", "content": "题库里没有同类题，请直接按课本方法讲解。"}
        )

    response = deps.client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
    )
    text = next((block.text for block in response.content if block.type == "text"), "")
    return {"messages": [{"role": "assistant", "content": text}], "answer": text}


def decline(state: TutorState, deps: Deps) -> dict:
    """闲聊或超出范围时的兜底，不调模型。"""
    text = "同学，我们先把数学题弄明白好不好？你把题目发给我，或者告诉我哪一课不懂。"
    return {"messages": [{"role": "assistant", "content": text}], "answer": text}
