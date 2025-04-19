## Base Policies

Base policies are heuristics for CTF derived from `BaseAgentPolicy` in `base.py`. The standard base policies are:

- `BaseAttacker`
- `BaseDefender`
- `Heuristic_CTF_Agent` *(a heuristic combination of `BaseAttacker` and `BaseDefender`)*

The `BaseAgentPolicy` class can be extended with inheritance to create custom heuristics.

> The old versions of `BaseAgentPolicy`, `BaseAttacker`, `BaseDefender`, and `Heuristic_CTF_Agent` can be found in the `./deprecated/` folder.

The new base policies use the global state from the `info` dictionary to make decisions, whereas the old versions require the full unnormalized observation dictionary *(a dictionary keyed by agent IDs containing each agent's unnormalized observation)*.

---

## Deprecated Base Policies (**currently recommended**)

It is still possible to use the old base policies, as the Pyquaticus `info` dictionary can return the unnormalized observation of each agent:

```unnorm_full_obs = {agent_id: info[agent_id]["unnrorm_obs"] for agent_id in info}```

However, if the observations incorporate a history *(e.g. stacking previous observations with the current)*, you will need to extract only the **most recent** unnormalized observation before passing `unnorm_full_obs` to the base policy's `compute_action()` method:

```unnorm_full_obs = {agent_id: info[agent_id]["unnrorm_obs"][0] for agent_id in info}```

While in theory the behaviors of the new and old versions of the standard base policies should match, the **new versions have not been thoroughly tested** and there may be bugs or inconsistencies.

> **We recommend using the old versions for now and submitting issues for the new ones as they are found.**


<!--
# ## Base Policies
# Base policies are heuristics for CTF derived from `BaseAgentPolicy` in `base.py`. The standard base policies are `BaseAttacker`, `BaseDefender`, and `Heuristic_CTF_Agent` (a heuristic combination of `BaseAttacker` and `BaseDefender`). The `BaseAgentPolicy` class can be extended with inheritance to create custom hueristics. The old versions of `BaseAgentPolicy`, `BaseAttacker`, `BaseDefender`, and `Heuristic_CTF_Agent` can be found in the `./deprecated/` folder. The new base policies use the global state from the `info` dictionary to make decisions, wheras the old versions require the full unnormalized observation dictionary (a dictionary eyed by agent id's containing each agent's unnormalized observation).

# ## Deprecated Base Policies (currently recommended)
# It is still possible to use the old base policies as the Pyquaticus info dictionary can return the unnormalized observation of each agent (e.g. `unnorm_full_obs = {agent_id: info[agent_id]["unnrorm_obs"] for agent_id in info`), however if a the obervations incorporate a history (stacking previous observations with the current), you will have to extract only the most recent unnormalized observation for each agent before passing `unnorm_full_obs` to the base policy's `compute_action()` method (e.g. `unnorm_full_obs = {agent_id: info[agent_id]["unnrorm_obs"][0] for agent_id in info`). While in theory the behaviors of the new and old versions of the standard base policies should match, the new versions have not been thoroughly tested and there may be bugs and inconsistencies with the old versions. We recommend using the old for the present and submitting issues for the new as they are found.
-->