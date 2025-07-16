import re
import sqlparse
import sqlglot

def normalize_sql(sql):
    """Normalize SQL for exact string matching."""
    try:
        formatted = sqlparse.format(sql, keyword_case='upper', reindent=True)
        return re.sub(r'\s+', ' ', formatted).strip().lower()
    except Exception:
        return sql.strip().lower()

def extract_sql(raw_output):
    """Extract SQL code block or fallback to first semicolon-terminated SQL."""
    match = re.search(r"```sql\s*(.*?)```", raw_output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"(SELECT|INSERT|UPDATE|DELETE).*?;", raw_output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()
    return raw_output.strip()

def ast_equal_ignore_alias(sql1, sql2):
    try:
        def strip_aliases(expression):
            for node in expression.find_all(sqlglot.exp.Alias):
                node.replace(node.this)
            return expression

        tree1 = strip_aliases(sqlglot.parse_one(sql1))
        tree2 = strip_aliases(sqlglot.parse_one(sql2))
        return tree1 == tree2
    except Exception:
        return False

def extract_json_block(text):
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    return text  # fallback to whole string

def remove_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)