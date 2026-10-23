# 数据库设计说明
数据库：PostgreSQL
核心表：
1. textbook_catalog：教材目录表（年级、学期、版本、单元、课时、课时标题）
2. knowledge_point：知识点表，独立编码，支持跨版本关联
3. lesson_kp_rel：课时-知识点关联表（一个课时对应多个知识点）
4. math_question：题目表，包含题干、答案、解析、难度、题型、embedding向量数组
5. student_record：学生学习记录、答题记录、正确率、错题标记

向量存储方式：使用 PostgreSQL `float[]` 存储向量；
检索逻辑：SQL WHERE 条件先过滤，取出向量数组到内存，执行余弦相似度计算。
## 📌 后续迭代计划（TODO）

- 接入 FAISS，实现 ANN 向量召回，支持更大题库
- 完善学情报表、知识点掌握图谱
- 接入 OCR，支持拍照上传题目自动解析
- 增加前端交互页面（Streamlit）
- 增加单元测试覆盖率、日志链路追踪