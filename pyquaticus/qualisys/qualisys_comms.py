
import asyncio
import logging

import qtm_rt
import multiprocessing
#from multiprocessing import Process, Array

def on_packet(packet):
    """ Callback function that is called everytime a data packet arrives from QTM """
    print("Framenumber: {}".format(packet.framenumber))
    header, markers = packet.get_3d_markers()
    print("Component info: {}".format(header))
    for marker in markers:
        print("\t", marker)
    position = [0.0,0.0]
    heading = 155.5
    return position, heading

async def setup():
    """ Main function """
    connection = await qtm_rt.connect("192.168.8.231", version='1.13')#("127.0.0.1")
    if connection is None:
        return
    
    test = await connection.stream_frames(components=["3d"], on_packet=on_packet)
    print("Test Return: ", test)


def gather_agents(event, agents, agent_mapper):
    init = False
    while True:
        if self.event.is_set():
            return
    if init == False:
        asyncio.ensure_future(setup())
        asyncio.get_event_loop().run_forever()

class QualisysComms:
    def __init__(self, agent_mapper={'blue_one':0,}):
        self.num_agents = num_agents
        manager = multiprocessing.Manager()
        self.event = manager.Event()
	    self.agents = manager.dict()
        self.agent_mapper = agent_mapper
        for a in self.agent_mapper:
            self.agents[self.agent_mapper[a]]['pos'] = [0,0]
            self.agents[self.agent_mapper[a]]['heading'] = [0.0]
        self.p = None
    def end(self,):
        if self.p == None:
            return True
        self.event.set()
        self.p.join()
        return True
    def connect(self,):
        if not self.p == None:
            self.end()
        self.p = manager.Process(target=gather_agents, args=(self.event, self.agents, self.agent_mapper))
        self.p.start()
    def close(self):
        self.end()
        return
    def get_positions(self):
        return self.agents

