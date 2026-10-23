"""组装 LangGraph 工作流。

这个文件只做两件事：把节点连起来，把"下一步走哪"的判断写成路由函数。
它不含任何业务计算 —— 一旦这里出现 if score > 0.6 之类的东西，
说明某个判断逃出了 nodes.py，以后就得在两个地方改。

整条路线：

    START → collect_info ─┬─(缺槽位)→ ask_missing → END
                          └─(齐了) → detect_intent ─┬─(闲聊)→ decline → END
                                                    └─(数学)→ retrieve → validate
                                                                            │
                           generate → END ←─(有题 / 重试用尽)──────────────┤
                                       retrieve ←─(没题且还能重试)─────────┘

用法::

    import anthropic
    from math_tutor_agent.agent.graph import build_graph
    from math_tutor_agent.agent.nodes import Deps
    from math_tutor_agent.retrieval.retriever import QuestionRetriever

    graph = build_graph(Deps(retriever=QuestionRetriever(), client=anthropic.Anthropic()))
    state = graph.invoke(
        {"messages": [{"role": "user", "content": "三年级人教版第五单元这道题不会"}]},
        config={"configurable": {"thread_id": "student-42"}},
    )
    print(state["answer"])
"""

from __future__ import annotations

from functools import partial

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import nodes
from .nodes import Deps
from .state import TutorState

# ---- 路由函数 --------------------------------------------------------------
# 路由只读 state、只返回一个字符串标签，不改 state、不调模型。
# 写成独立函数而不是 lambda，是因为它们是最该被单测覆盖的部分：
# 给一个构造出来的 state，断言它走哪条边，不需要跑整张图。


def route_after_collect(state: TutorState) -> str:
    """槽位不全就先问清楚。

    这里宁可多问一轮也不猜：年级猜错，后面所有的分年级约束就全错了，
    而且错得很隐蔽 —— 模型会一本正经地给三年级学生讲方程。
    """
    return "ask" if state.get("missing") else "ok"


def route_by_intent(state: TutorState) -> str:
    """闲聊直接兜底，不白跑一次检索。"""
    return "decline" if state.get("intent") == "chat" else "retrieve"


def route_after_validate(state: TutorState) -> str:
    """校验之后决定重检索还是开讲。

    三种情况：
    - 还有题 → 直接讲。
    - 没题但还有重试次数 → 回到 retrieve，validate 会自动放宽阈值。
    - 没题且重试用尽 → 还是去 generate。这是有意的：没有同类题不代表
      讲不了，generate 里有"按课本方法讲"的分支。宁可讲得朴素一点，
      也别让学生收到一句"没找到"。

    attempt 由 retrieve 自增，是这条环路唯一的收敛依据 ——
    条件边只要能绕回去，就必须有一个单调变化的计数兜底，否则就是死循环。
    """
    if state.get("retrieved"):
        return "generate"
    if state.get("attempt", 0) <= nodes.MAX_RETRY:
        return "retry"
    return "generate"


# ---- 组装 ------------------------------------------------------------------


def build_graph(deps: Deps, checkpointer: BaseCheckpointSaver | None = None):
    """把节点连成图。

    deps 用 partial 绑进节点：LangGraph 只会用 state 调用节点，
    而节点需要检索器和模型客户端。绑在这里的好处是整张图只组装一次依赖，
    测试时传一份假的 Deps 就能跑完整条链，不用打补丁、不用改全局变量。

    checkpointer 决定多轮对话能不能续上：没有它，学生上一句说过的年级
    下一句就忘了，每轮都要重新问。开发用 MemorySaver，
    线上换成 Postgres 的 saver，调用处一行都不用改。
    """
    bind = lambda node: partial(node, deps=deps)  # noqa: E731

    builder = StateGraph(TutorState)
    builder.add_node("collect_info", bind(nodes.collect_info))
    builder.add_node("ask_missing", bind(nodes.ask_missing))
    builder.add_node("detect_intent", bind(nodes.detect_intent))
    builder.add_node("retrieve", bind(nodes.retrieve))
    builder.add_node("validate", bind(nodes.validate))
    builder.add_node("generate", bind(nodes.generate))
    builder.add_node("decline", bind(nodes.decline))

    builder.add_edge(START, "collect_info")
    builder.add_conditional_edges(
        "collect_info",
        route_after_collect,
        {"ask": "ask_missing", "ok": "detect_intent"},
    )
    builder.add_edge("ask_missing", END)

    builder.add_conditional_edges(
        "detect_intent",
        route_by_intent,
        {"decline": "decline", "retrieve": "retrieve"},
    )
    builder.add_edge("decline", END)

    builder.add_edge("retrieve", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {"generate": "generate", "retry": "retrieve"},
    )
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
