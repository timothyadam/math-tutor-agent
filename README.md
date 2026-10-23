```mermaid
flowchart TD
    A[math-tutor-agent/] --> B[README.md<br/>项目介绍、架构图、演示、技术选型、取舍思考]
    A --> C[docs/]
    C --> C1[architecture.md<br/>架构说明、数据流图]
    C --> C2[database.md<br/>建表SQL、表结构设计、ER图]
    C --> C3[demo.md<br/>对话样例]

    A --> D[app/]
    D --> D1[main.py<br/>FastAPI入口]
    D --> D2[agent/]
    D2 --> D21[state.py<br/>LangGraph State定义<br/>年级/版本/单元/课时]
    D2 --> D22[nodes.py<br/>收集信息、意图识别、检索、校验、生成]
    D2 --> D23[graph.py<br/>组装LangGraph工作流]
    D2 --> D24[prompts.py<br/>系统提示词（分年级约束）]

    D --> D3[retrieval/]
    D3 --> D31[embedding.py<br/>embedding封装（bge等）]
    D3 --> D32[similarity.py<br/>✅手写余弦相似度实现，面试重点]
    D3 --> D33[retriever.py<br/>元过滤→候选→相似度排序]

    D --> D4[db/]
    D4 --> D41[models.py<br/>SQLAlchemy ORM模型<br/>教材目录、课时、题目、学情]
    D4 --> D42[init_data.py<br/>初始化人教版1-6年级目录、样例题目]

    D --> D5[schema/<br/>Pydantic 请求响应模型]
    D --> D6[service/]

    A --> E[tests/<br/>单元测试，面试加分]
    E --> E1[test_similarity.py]
    E --> E2[test_retriever.py]
    E --> E3[test_agent_graph.py]

    A --> F[docker-compose.yml<br/>一键启动 pg + redis + api]
    A --> G[.env.example]
    A --> H[requirements.txt]
    A --> I[.gitignore]
    A --> J[LICENSE]