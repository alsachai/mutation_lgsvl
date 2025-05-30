from enum import Enum
import math
import pickle
from environs import Env
import lgsvl
import random



def replay(scenario, sim):

    if sim.current_scene == "SanFrancisco_correct":
        sim.reset()
    else:
        sim.load("SanFrancisco_correct")


    # create ego
    ego_state = lgsvl.AgentState()
    ego_state.transform = lgsvl.Transform(position=lgsvl.Vector(x=scenario[0][0][1], y=scenario[0][0][2], z=scenario[0][0][3]), rotation=lgsvl.Vector(x=scenario[0][0][4], y=scenario[0][0][5], z=scenario[0][0][6]))
    ego = sim.add_agent('SUV', lgsvl.AgentType.NPC, ego_state)

    # create npc
    n = len(scenario)
    m = len(scenario[0])
    print(n,m)
    npc = []
    name = ["Jeep", "Sedan", "SUV"]
    for i in range(1, n):
        npc_state = lgsvl.AgentState()
        npc_state.transform = lgsvl.Transform(position=lgsvl.Vector(x=scenario[i][0][1], y=scenario[i][0][2], z=scenario[i][0][3]), rotation=lgsvl.Vector(x=scenario[i][0][4], y=scenario[i][0][5], z=scenario[i][0][6]))
        npc.append(sim.add_agent(name[i-1], lgsvl.AgentType.NPC, npc_state))

    # util function
    def cal_speed(speed):
        return math.sqrt(speed.x ** 2 + speed.y ** 2 + speed.z ** 2)

    def on_waypoint(agent, index):
        print("waypoint {} reached".format(index))

    # ego waypoints
    ego_waypoints = []
    for i in range(1, m):
        wp = lgsvl.DriveWaypoint(position=lgsvl.Vector(x=scenario[0][i][1], y=scenario[0][i][2], z=scenario[0][i][3]),
                                 angle=lgsvl.Vector(x=scenario[0][i][4], y=scenario[0][i][5], z=scenario[0][i][6]),
                                 speed=scenario[0][i][0])
        ego_waypoints.append(wp)
    ego.follow(ego_waypoints)

    # npc wypoints
    for i in range(1, n):
        npc_waypoints = []
        for j in range(1, m):
            wp = lgsvl.DriveWaypoint(position=lgsvl.Vector(x=scenario[i][j][1], y=scenario[i][j][2], z=scenario[i][j][3]),
                                     angle=lgsvl.Vector(x=scenario[i][j][4], y=scenario[i][j][5], z=scenario[i][j][6]),
                                     speed=scenario[i][j][0])
            npc_waypoints.append(wp)
        npc[i-1].follow(npc_waypoints)


    # run simulation
    cnt = 0
    while cnt < m:
        for _ in range(100):
            ego_record_state = ego.state.transform
            up = lgsvl.utils.transform_to_up(ego_record_state)
            forward = lgsvl.utils.transform_to_forward(ego_record_state)
            ego_record_state.position += up * 10 - forward * 10
            ego_record_state.rotation += lgsvl.Vector(30,0,0)
            sim.set_sim_camera(ego_record_state)
            sim.run(0.01)
        cnt += 1


def main(path):
    with open(path, 'rb') as f:
        scenario = pickle.load(f)

    # initial simulation
    env = Env()
    sim = lgsvl.Simulator(env.str("LGSVL__SIMULATOR_HOST", lgsvl.wise.SimulatorSettings.simulator_host),
                          env.int("LGSVL__SIMULATOR_PORT", lgsvl.wise.SimulatorSettings.simulator_port))

    replay(scenario, sim)


if __name__ == '__main__':
    main('/home/tieriv/alsachai/mutation_lgsvl/outputs/ds_1_v2-at-12-13-2024-16-59-46/simulation/replays/scenario_88.obj')
