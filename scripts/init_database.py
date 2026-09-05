"""为全新 PostgreSQL 初始化两级知识库表结构和兼容迁移。"""

import sys
from pathlib import Path

# 允许从仓库根目录直接执行 `python scripts/init_database.py`。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.bootstrap import initialize_application


if __name__ == "__main__":
    initialize_application()
    print("Database initialization complete")
