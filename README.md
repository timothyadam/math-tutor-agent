## 项目结构

```text
math-tutor-agent/
├── README.md               # 最重要！项目介绍、架构图、演示、技术选型、取舍思考
├── docs/
│   ├── architecture.md     # 架构说明、数据流图
│   ├── database.md         # 建表SQL、表结构设计、ER图
│   └── demo.md             # 对话样例
├── app/
│   ├── main.py             # FastAPI 入口
│   ├── agent/
│   │   ├── state.py        # LangGraph State定义（年级/版本/单元/课时）
│   │   ├── nodes.py        # 各个节点：收集信息、意图识别、检索、校验、生成
│   │   ├── graph.py        # 组装LangGraph工作流
│   │   └── prompts.py      # 系统提示词（分年级约束）
│   ├── retrieval/
│   │   ├── embedding.py    # embedding封装（bge等）
│   │   ├── similarity.py   # ✅ 手写余弦相似度实现（面试重点文件！）
│   │   └── retriever.py    # 检索逻辑：先元过滤 → 取候选 → 算相似度排序
│   ├── db/
│   │   ├── models.py       # SQLAlchemy ORM模型（教材目录、课时、题目、学情）
│   │   └── init_data.py    # 初始化人教版1-6年级目录、样例题目
│   ├── schema/             # Pydantic 请求响应模型
│   └── service/
├── tests/                  # 单元测试！面试加分项：检索测试、相似度测试、agent路由测试
│   ├── test_similarity.py
│   ├── test_retriever.py
│   └── test_agent_graph.py
├── docker-compose.yml      # 一键启动：pg + redis + api，面试官一键跑起来
├── .env.example
├── requirements.txt
├── .gitignore
└── LICENSE
```
