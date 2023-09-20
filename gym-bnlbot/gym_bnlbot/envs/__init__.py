from gym.envs.registration import register


register(
     id="gym_bnlbot/Bnlbot-v0",
     entry_point="gym_bnlbot.envs:BnlbotEnv",
     max_episode_steps=3000,
)

