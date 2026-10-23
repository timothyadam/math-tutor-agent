"""LangGraph State 定义（年级/版本/单元/课时）。

State 是整张图里唯一共享的数据结构。每个节点读它，返回一个"只含本节点改动"
的 dict，LangGraph 负责合并。所以字段怎么设计，直接决定了节点之间有多耦合，
也决定了多轮对话里哪些信息能被续上。
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

# 年级用 Literal 钉死：超出 1-6 在类型检查阶段就报错，不用等运行时
Grade = Literal[1, 2, 3, 4, 5, 6]

Intent = Literal[
    "solve",     # 拿着一道具体的题问怎么做
    "explain",   # 要听某个知识点的讲解
    "practice",  # 要练习题
    "check",     # 要核对自己的答案
    "chat",      # 闲聊或与数学无关，礼貌挡回去
]

# 必须问清楚才能检索的槽位，顺序就是追问顺序：先年级，再版本，最后单元
REQUIRED_SLOTS: tuple[str, ...] = ("grade", "edition", "unit")


class Slots(TypedDict, total=False):
    """把"学生是谁、在学哪一课"单独收成一层。

    单独一层的好处：判断"还缺什么"只要比对这一个 dict 的键，
    不用在十几个平铺字段里挑；多轮对话里也只需要把这一层带着走。
    """

    grade: Grade
    edition: str  # 教材版本：人教版 / 北师大版 / 苏教版
    unit: str
    lesson: str   # 课时可选，没有就按整个单元检索


class RetrievedItem(TypedDict):
    """检索结果进 State 时只留"够下一步用"的最小信息。

    题目全文、解析、embedding 向量一律不进 State：
    State 会被序列化、被 checkpoint、被日志打印，放大对象会让这三件事都变贵，
    而且一旦有了副本，题库更新后 State 里的就是脏数据。要全文就按 id 回库里取。
    """

    question_id: int
    stem: str             # 题干摘要，截断到 120 字
    knowledge_point: str
    grade: int            # 题目本身的年级，用来挡超纲题
    score: float          # 余弦相似度


class TutorState(TypedDict, total=False):
    # messages 是唯一需要"累加"的字段，所以给它挂 reducer：
    # 节点返回 {"messages": [新消息]}，LangGraph 自动追加到历史后面。
    # 其余字段没有 reducer，节点返回什么就覆盖什么 —— 这正是我们要的：
    # 重新检索一次，retrieved 就该整个换掉，而不是越堆越多。
    messages: Annotated[list[dict], operator.add]

    slots: Slots
    missing: list[str]          # 还没问到的槽位
    intent: Intent
    retrieved: list[RetrievedItem]
    rejected: list[str]         # 被校验挡掉的原因，便于排查"为什么没给题"
    answer: str
    attempt: int                # 检索重试次数，条件边靠它防死循环
