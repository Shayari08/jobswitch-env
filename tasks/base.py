from env.environment import JobSwitchEnvironment


class BaseTask:
    SEED: int = 42
    TASK_ID: int = 0
    MAX_STEPS: int = 30

    def __init__(self, env: JobSwitchEnvironment):
        self.env = env

    async def reset(self) -> dict:
        obs = await self.env.reset(
            seed=self.SEED, task_id=self.TASK_ID, max_steps=self.MAX_STEPS
        )
        self._configure_scenario()
        # Rebuild observation after scenario configuration
        return self.env._build_observation().model_dump()

    def _configure_scenario(self):
        raise NotImplementedError

    def grade(self) -> float:
        raise NotImplementedError
