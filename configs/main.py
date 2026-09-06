import os
import asyncio
import aiomysql
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

async def init_mysql_database():
    """初始化MySQL数据库（只在表不存在时创建）"""

    config = {
        "host": os.getenv("DB_HOST", "8.155.1.111"),
        "port": int(os.getenv("DB_PORT", 3306)),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", "root"),
        "charset": "utf8mb4"
    }

    db_name = os.getenv("DB_NAME", "agent_db")

    # 1. 确保数据库存在
    async with aiomysql.connect(**config) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = %s",
                (db_name,)
            )
            exists = await cursor.fetchone()

            if not exists:
                await cursor.execute(
                    f"CREATE DATABASE {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
                print(f"✅ 数据库 '{db_name}' 创建成功")
            else:
                print(f"ℹ️ 数据库 '{db_name}' 已存在")

    # 2. 连接到目标数据库，检查并创建表
    config["db"] = db_name
    async with aiomysql.connect(**config) as conn:
        async with conn.cursor() as cursor:

            # 检查表是否存在
            await cursor.execute(
                "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
                "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'tool_definitions'",
                (db_name,)
            )
            table_exists = await cursor.fetchone()

            if not table_exists:
                # 表不存在，创建
                await cursor.execute("""
                    CREATE TABLE tool_definitions (
                        id VARCHAR(64) PRIMARY KEY,
                        name VARCHAR(100) NOT NULL UNIQUE,
                        description TEXT NOT NULL,
                        type VARCHAR(50) NOT NULL DEFAULT 'api',
                        api_config JSON,
                        parameters JSON NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                    )
                """)
                print("✅ 表 'tool_definitions' 创建成功")

                # 插入示例数据
                await cursor.execute("""
                    INSERT INTO tool_definitions (id, name, description, type, api_config, parameters) 
                    VALUES (
                        'tool_001',
                        'get_weather',
                        '查询中国城市实时天气信息，包括温度、湿度、风向等',
                        'api',
                        '{"url": "https://uapis.cn/api/v1/misc/weather", "method": "GET", "timeout": 10}',
                        '[{"name": "city", "type": "string", "description": "城市名称，如北京、上海", "required": true}]'
                    )
                """)
                print("✅ 示例数据插入成功")
            else:
                print("ℹ️ 表 'tool_definitions' 已存在，跳过创建")

    print("🎉 MySQL 数据库初始化完成（无需重复创建）！")