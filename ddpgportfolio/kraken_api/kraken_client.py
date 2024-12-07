from typing import List, Optional, Dict, Any
import aiohttp
import asyncio
from dataclasses import dataclass, field
import datetime

KRAKEN_URL: str = "https://api.kraken.com/0/public/OHLC"


@dataclass
class KrakenClient:
    session: Optional[aiohttp.ClientSession] = field(init=False, default=None)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()

    async def fetch(self, url: str, params: Dict[str, Any]):
        if not self.session:
            raise ValueError(
                "Session has not been initialized.  Please use async with AiohttpClient() to initialize"
            )
        headers = {"Accept": "application/json"}
        async with self.session.get(url, params=params, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Error fetching {url}, status code: {response.status}")

    async def fetch_ohlc_data(self, currency_pair: str, start: int, period: int = 30):
        """fetches the ohlc data for a particular currency pair

        Parameters
        ----------
        currency_pair : str
            the currency pair to fetch data for
        start : int
            start of time period
        end : int
            end of time period
        period : int, optional
            by default 1800 seconds (30 minutes)
        """
        print("fetching data")
        params: Dict[str, Any] = {
            "pair": currency_pair,
            "since": start,
            "interval": period,
        }

        data = await self.fetch(KRAKEN_URL, params)
        return data["result"]


async def fetch_multiple_currency_pairs(
    currency_pairs: List[str],
    days_from_now: int = 14,
    period: int = 30,
):
    end_time = datetime.datetime.now(datetime.timezone.utc)
    start_time = end_time - datetime.timedelta(days=days_from_now)

    # convert to unix timestamps
    start_timestamp = 1704067200

    async with KrakenClient() as client:
        tasks = [
            asyncio.create_task(
                client.fetch_ohlc_data(
                    currency_pair, start=start_timestamp, period=period
                )
            )
            for currency_pair in currency_pairs
        ]
        for task in tasks:
            result = await task
            last = result.pop("last")
            for currency_pair in result:
                data = result[currency_pair]
                if data:
                    print(f"Data for {currency_pair}")
                    for candle in data:
                        timestamp = datetime.datetime.fromtimestamp(
                            candle[0], tz=datetime.timezone.utc
                        )
                        open_price = candle[1]
                        high_price = candle[2]
                        low_price = candle[3]
                        close_price = candle[4]
                        print(
                            f" Time: {timestamp} | Open:  {open_price} | High: {high_price} | Low: {low_price} | Close: {close_price}"
                        )
                    print()


if __name__ == "__main__":
    print("startiing program")
    currency_pairs: List[str] = [
        "ETH/USD",
        "ADA/USD",
        "BTC/USD",
        "XRP/USD",
        "LINK/USD",
        "DOT/USD",
        "LTC/USD",
        "USDT/USD",
        "SOL/USD",
        "USDC/USD",
        "MATIC/USD",
        "PEPE/USD",
        "SHIB/USD",
    ]
    asyncio.run(fetch_multiple_currency_pairs(currency_pairs=currency_pairs))
