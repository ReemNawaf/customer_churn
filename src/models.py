from pydantic import BaseModel


class UsersLogs(BaseModel):
    logs: list[dict]
