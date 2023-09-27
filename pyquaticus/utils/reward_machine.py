# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.
#
# (C) 2023 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from transitions import EventData, Machine

TAG_PENALTY = -5.0


class RewardMachine:
    # states = ['defend', 'seek_flag', 'return_flag', 'out_of_bounds', 'recover_flag']
    states = ["seek_flag", "getting_flag", "return_flag", "defend", "tagged_opp"]
    states_set = set(states)
    num_states = 0
    state_to_idx = {}
    for s in states:
        state_to_idx[s] = num_states
        num_states += 1

    # TODO: possibly have nonzero additions for other transitions
    # maps trigger to (team reward, other team reward)
    reward_map = {
        "to_getting_flag": (1.0, -1.0),
        # the two tag transitions are coupled
        # easier to reason about by only modifying own reward
        "tagged": (TAG_PENALTY, 0.0),
        "did_tag": (-TAG_PENALTY, 0),
    }

    def __init__(self, init_state: str, env, agent_ind: int, debug: bool = True):
        raise DeprecationWarning("RewardMachine is deprecated. Needs to be updated for use with Pyquaticus if needed.")

        if init_state not in RewardMachine.states_set:
            raise ValueError(
                f"Got initial state {init_state}, but expecting state from:"
                f" {RewardMachine.states}"
            )

        self._env = env
        self._agent_ind = agent_ind
        self._debug = debug

        self._automata = Machine(
            model=self,
            states=RewardMachine.states,
            initial=init_state,
            auto_transitions=False,
        )

        # add these back in when ready to have rewards per automata state
        self._automata.add_transition(
            "to_getting_flag", "seek_flag", "getting_flag", conditions=["have_flag"]
        )
        self._automata.add_transition(
            "to_return_flag", "getting_flag", "return_flag", conditions=["have_flag"]
        )
        self._automata.add_transition(
            "tagged",
            ["seek_flag", "return_flag"],
            "seek_flag",
            conditions=["got_tagged"],
        )
        # self._automata.add_transition('team_won', 'return_flag', 'won', conditions=['is_done'])

        # defend states are unconnected to others for now (basically a separate automata)
        self._automata.add_transition(
            "did_tag", "defend", "tagged_opp", conditions="tagged_someone"
        )
        self._automata.add_transition("defend_again", "tagged_opp", "defend")

    @property
    def have_flag(self):
        return self._env.state["agent_has_flag"][self._agent_ind]

    @property
    def got_tagged(self):
        res = bool(self._env.state["agent_tagged"][self._agent_ind])
        assert not (res and self.have_flag)
        return res

    @property
    def tagged_someone(self):
        return self._env.state["agent_captures"][self._agent_ind] is not None

    @property
    def captured_flag(self):
        team = self.get_team()
        assert team in {"blue", "red"}
        return (team == "blue" and self._env.blue_team_flag_capture) or (
            team == "red" and self._env.red_team_flag_capture
        )

    @property
    def is_done(self):
        return self._env.dones[self.get_team()]

    # function based on PeekMachine here:
    # https://github.com/pytransitions/transitions/blob/master/examples/Frequently%20asked%20questions.ipynb
    def can_trigger(self, *args, **kwargs):
        e = EventData(None, None, self._automata, self, args, kwargs)

        for trigger_name in self._automata.get_triggers(self.state):
            if any(
                all(c.check(e) for c in t.conditions)
                for t in self._automata.events[trigger_name].transitions[self.state]
            ):
                yield trigger_name

    def transition(self):
        if self._debug:
            # check for multiple triggers
            triggers = list(self.can_trigger())
        else:
            # only get the first one
            triggers = []
            for trigger in self.can_trigger():
                triggers.append(trigger)
                break

        if len(triggers) > 1:
            raise RuntimeError(
                f'Nondeterminism in state "{self.state}",'
                " got following matching triggers:\n\t"
                + "\n\t".join(triggers)
            )
        elif triggers:
            self.trigger(triggers[0])
            return triggers[0]

    def get_team(self):
        """Return the team affiliated with this agent."""
        team, _ = self._env.get_agent_team_idx(self._agent_ind)
        return team

    @property
    def state_idx(self):
        return RewardMachine.state_to_idx[self.state]

    def shortest_dist_to(self, team, poi):
        """
        Args:
            poi: point of interest (x, y) position.

        Returns
        -------
            (float, int): shortest distance and index of closest agent of the
                          given team to the point of interest
        """
        agents = self._env.agents_of_team[team]
        ag_id = agents[0]
        shortest_dist = self._env.get_distance_between_2_points(
            self._env.state["agent_position"][ag_id], poi
        )
        for ag in agents[1:]:
            ag_dis_2_flag = self._env.get_distance_between_2_points(
                self._env.state["agent_position"][ag], poi
            )
            if ag_dis_2_flag < shortest_dist:
                shortest_dist = ag_dis_2_flag
                ag_id = ag
        return shortest_dist, ag_id

    def compute_reward(self):
        this_reward = 0.0
        other_reward = 0.0
        ag_pos = self._env.state["agent_position"][self._agent_ind]
        prev_ag_pos = self._env.state["prev_agent_position"][self._agent_ind]
        velocity_direc = self._env.state["agent_velocity"][self._agent_ind]
        velocity_norm = np.linalg.norm(velocity_direc)
        if velocity_norm > 1e-6:
            velocity_direc = velocity_direc / velocity_norm
        team = self.get_team()
        team_idx = self._env._get_team_idx(team)
        other_team = "red" if team == "blue" else "blue"
        ret_flag_pos = self._env.state["flag_locations"][int(not team_idx)][0]
        scrimmage = self._env.scrimmage

        if self.state == "seek_flag":
            to_flag_direc = np.subtract(ret_flag_pos, ag_pos)
            to_flag_direc = to_flag_direc / np.linalg.norm(to_flag_direc)
            direc_rew = np.dot(to_flag_direc, velocity_direc)
            this_reward += direc_rew
            other_reward -= 0.01 * this_reward

        elif self.state == "return_flag":
            if team == "blue":
                # towards own side is West
                direc_rew = -velocity_direc[0]
            else:
                # towards own side is East
                direc_rew = velocity_direc[0]
            this_reward += direc_rew
            other_reward -= 0.01 * this_reward

        elif self.state == "getting_flag":
            this_reward += 1.0
            other_reward -= 0.01 * this_reward

        elif self.state == "defend":
            prot_flag_pos = np.asarray(self._env.state["flag_locations"][team_idx][0])
            # handle the two cases of being on the wrong side as blue or red
            if team == "blue" and (prev_ag_pos[0] > scrimmage or ag_pos[0] > scrimmage):
                to_scrimmage = np.asarray([-1, 0])
                direc_rew = np.dot(to_scrimmage, velocity_direc)
                this_reward += direc_rew
                other_reward -= 0.01 * this_reward
            elif team == "red" and (
                prev_ag_pos[0] < scrimmage or ag_pos[0] < scrimmage
            ):
                to_scrimmage = np.asarray([1, 0])
                direc_rew = np.dot(to_scrimmage, velocity_direc)
                this_reward += direc_rew
                other_reward -= 0.01 * this_reward
            else:
                _, nearest_other_id = self.shortest_dist_to(other_team, prot_flag_pos)
                nearest_agent_pos = self._env.state["agent_position"][nearest_other_id]
                to_other_direc = np.subtract(nearest_agent_pos, ag_pos)
                to_other_norm = np.linalg.norm(to_other_direc)
                if to_other_norm > 1e-6:
                    to_other_direc = to_other_direc / to_other_norm

                direc_rew = np.dot(to_other_direc, velocity_direc)

                this_reward += direc_rew
                other_reward -= 0.01 * this_reward

        on_opponent_side = not self._env.state["agent_on_sides"][self._agent_ind]
        if on_opponent_side and self.state != "defend":
            shortest_dist, _ = self.shortest_dist_to(other_team, ag_pos)
            dist_penalty = -TAG_PENALTY - (shortest_dist - self._env.catch_radius) ** 2
            if dist_penalty > 0:
                this_reward -= dist_penalty

        return this_reward, other_reward

    def transition_reward(self, trigger):
        return RewardMachine.reward_map.get(trigger, (0.0, 0.0))
