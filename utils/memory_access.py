import json
import sqlite3


def process_query(query_str: str, db_path: str) -> str:
    """
    根据输入的JSON查询字符串和SQLite数据库路径，从数据库中查找匹配的第三者。

    :param query_str: JSON格式的查询字符串（包含relation + object或subject + relation等组合）
    :param db_path: SQLite数据库文件路径
    :return: JSON格式字符串，如 {"result": ["xxx", "yyy"]}
    """
    # 先去掉tool_call标签，提取中间内容
    query_str = query_str.split("<tool_call>")[1].split("</tool_call>")[0].strip()
    query = json.loads(query_str)
    args = query.get("arguments", {})
    result = []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if "subject" in args and "relation" in args:
        subject = args["subject"]
        relation = args["relation"]
        cursor.execute("""
            SELECT E2.Entity_name
            FROM TripleMemory TM
            JOIN Entities E1 ON TM.Subj_Entity_ID = E1.Entity_ID
            JOIN Entities E2 ON TM.Obj_Entity_ID = E2.Entity_ID
            JOIN Relations R ON TM.Relation_ID = R.Relation_ID
            WHERE E1.Entity_name = ? AND R.Relation_name = ?
        """, (subject, relation))
        result = [row[0] for row in cursor.fetchall()]

    elif "relation" in args and "object" in args:
        obj = args["object"]
        relation = args["relation"]
        cursor.execute("""
            SELECT E1.Entity_name
            FROM TripleMemory TM
            JOIN Entities E1 ON TM.Subj_Entity_ID = E1.Entity_ID
            JOIN Entities E2 ON TM.Obj_Entity_ID = E2.Entity_ID
            JOIN Relations R ON TM.Relation_ID = R.Relation_ID
            WHERE E2.Entity_name = ? AND R.Relation_name = ?
        """, (obj, relation))
        result = [row[0] for row in cursor.fetchall()]

    elif "subject" in args and "object" in args:
        subject = args["subject"]
        obj = args["object"]
        cursor.execute("""
            SELECT R.Relation_name
            FROM TripleMemory TM
            JOIN Entities E1 ON TM.Subj_Entity_ID = E1.Entity_ID
            JOIN Entities E2 ON TM.Obj_Entity_ID = E2.Entity_ID
            JOIN Relations R ON TM.Relation_ID = R.Relation_ID
            WHERE E1.Entity_name = ? AND E2.Entity_name = ?
        """, (subject, obj))
        result = [row[0] for row in cursor.fetchall()]

    cursor.close()
    conn.close()

    return json.dumps({"result": result})

if __name__ == '__main__':
    db_path = "../memory/memory.db"
    # query = "{\"name\": \"MEM_READ\", \"arguments\": {\"relation\": \"headquarters location\", \"object\": \"Pasay City\"}}"
    query = '{"name": "MEM_READ", "arguments": {"subject": "Pasay City", "object": "Philippines"}}'
    print(process_query(query, db_path))