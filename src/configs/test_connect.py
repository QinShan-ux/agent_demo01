# test_mysql.py
import aiomysql
import asyncio


async def test_connection():
    config = {
        "host": "8.155.1.111",
        "port": 3306,
        "user": "root",
        "password": "root",
        "charset": "utf8mb4"
    }

    try:
        conn =  aiomysql.connect(**config)
        print("✅ MySQL 连接成功！")

        async with conn.cursor() as cursor:
            await cursor.execute("SELECT VERSION()")
            version = await cursor.fetchone()
            print(f"📌 MySQL 版本: {version[0]}")

        conn.close()

    except Exception as e:
        print(f"❌ 连接失败: {e}")


if __name__ == "__main__":
    asyncio.run(test_connection())