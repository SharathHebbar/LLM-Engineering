from .agent import execute
from shared.schemas import TravelRequest
async def run(payload: TravelRequest):
    print(payload)
    return await execute(payload)