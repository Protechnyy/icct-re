import time


class BudgetExhausted(RuntimeError):
    pass


class ServiceUnavailable(RuntimeError):
    pass


class Budget:
    def __init__(self, max_calls, timeout_seconds, clock=time.monotonic, sleep=time.sleep, trace=None):
        self.max_calls = max_calls
        self.clock, self.sleep = clock, sleep
        self.started_at = clock()
        self.deadline = self.started_at + timeout_seconds
        self.calls = 0
        self.tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self.trace = trace

    def check(self):
        if self.calls >= self.max_calls:
            raise BudgetExhausted("模型调用预算耗尽")
        if self.clock() >= self.deadline:
            raise BudgetExhausted("核查阶段超时")

    def invoke(self, model, messages):
        for attempt in range(3):
            self.check()
            self.calls += 1
            started = self.clock()
            call_id = f"model-{self.calls}"
            task_id = self.trace.current_task["id"] if self.trace and self.trace.current_task else None
            if self.trace:
                self.trace.emit("model_start", task_id, call_id=call_id, summary="正在等待模型返回")
            completed = False
            usage = {}
            try:
                response = model.invoke(messages, timeout=max(0.001, self.deadline - self.clock()))
                usage = getattr(response, "usage_metadata", None) or {}
                for field in self.tokens:
                    self.tokens[field] += int(usage.get(field) or 0)
                if self.clock() >= self.deadline:
                    raise BudgetExhausted("核查阶段超时")
                completed = True
                return response
            except BudgetExhausted:
                raise
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if status == 429:
                    if attempt == 2:
                        raise ServiceUnavailable("百炼持续限流") from exc
                    self.check()
                    delay = 2 ** attempt
                    if self.clock() + delay >= self.deadline:
                        raise BudgetExhausted("限流重试将超过核查时限") from exc
                    self.sleep(delay)
                    continue
                if status in (401, 403, 404) or (isinstance(status, int) and status >= 500) or type(exc).__name__ in (
                    "APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout"):
                    raise ServiceUnavailable("模型服务不可用：" + type(exc).__name__) from exc
                raise
            finally:
                if self.trace:
                    self.trace.emit("model_end", task_id, result={"usage": usage}, call_id=call_id,
                                    status="completed" if completed else "error",
                                    summary="模型已返回" if completed else "模型请求未完成",
                                    elapsed_seconds=self.clock() - started)

    @property
    def elapsed_seconds(self):
        return round(self.clock() - self.started_at, 4)
