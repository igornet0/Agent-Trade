import asyncio
from core.database import db_helper
from core import data_helper
from core.database.orm import orm_get_coins, orm_get_data_timeseries, orm_get_timeseries_by_coin

import pandas as pd

async def export_data_db_to_csv_all():
    async with db_helper.get_session() as session:
        coins = await orm_get_coins(session)
        for coin in coins:
            data_coin = {
                "datetime": [],
                "open": [],
                "max": [],
                "min": [],
                "close": [],
                "volume": []
            }
            tm = await orm_get_timeseries_by_coin(session, coin)
            # print(tm)
            for t in tm:
                datas = await orm_get_data_timeseries(session, t.id)
                for data in datas:
                    data_coin["datetime"].append(data.datetime)
                    data_coin["open"].append(data.open)
                    data_coin["max"].append(data.max)
                    data_coin["min"].append(data.min)
                    data_coin["close"].append(data.close)
                    data_coin["volume"].append(data.volume)
                df = pd.DataFrame(data_coin)
                df.to_csv(data_helper["raw"] / f"{coin.name}_{t.timestamp}.csv", index=False)
      
if __name__ == "__main__":
    asyncio.run(export_data_db_to_csv_all())