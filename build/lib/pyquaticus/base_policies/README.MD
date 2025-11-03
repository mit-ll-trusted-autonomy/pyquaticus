## Updated Base Policies *(recommended)*

Base policies are heuristics for CTF derived from `BaseAgentPolicy` in `base_policy.py`. The standard base policies are:

- `BaseAttacker`
- `BaseDefender`
- `Heuristic_CTF_Agent` *(a heuristic combination of* `BaseAttacker` *and* `BaseDefender`*)*

Additionally, `WaypointPolicy` is provided, which uses an RRT*-based algorithm to plan paths to waypoints while avoiding obstacles. 

The `BaseAgentPolicy` class can be extended with inheritance to create custom heuristics. Custom policies should override `compute_action(obs, info)`

The new base policies use only the global state from the `info` dictionary to make decisions, whereas the old versions require the full unnormalized observation dictionary *(a dictionary keyed by agent IDs containing each agent's unnormalized observation)*.

***We have tested these policies to ensure that they match the behavior of the deprecated ones, but please open issues for any bugs or discrepancies found.***

## Deprecated Base Policies

It is still possible to use the old base policies, as the Pyquaticus `info` dictionary can return the unnormalized observation of each agent:

```unnorm_full_obs = {agent_id: info[agent_id]["unnrorm_obs"] for agent_id in info}```

However, if the observations incorporate a history *(e.g. stacking previous observations with the current)*, you will need to extract only the **most recent** unnormalized observation before passing `unnorm_full_obs` to the base policy's `compute_action()` method:

```unnorm_full_obs = {agent_id: info[agent_id]["unnrorm_obs"][0] for agent_id in info}```