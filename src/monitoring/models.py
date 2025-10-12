from pydantic import BaseModel


class LogsPayload(BaseModel):
    logs: list[dict]


class UsersLogs(BaseModel):
    logs: list[dict]
