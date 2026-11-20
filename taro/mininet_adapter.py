"""Mininet data-plane adapter for the shared TARO topology format."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import time
import os
from .graph import GraphSpec


@dataclass
class ControllerSample:
    timestamp: float
    port_bytes: Dict[str, int]
    cpu_percent: float | None = None
    memory_bytes: int | None = None


class MininetAdapter:
    def __init__(self, topology: GraphSpec, controller=None, interval=1.0):
        self.topology = topology
        self.controller = controller
        self.interval = interval
        self.samples = []
        self.started = False
        self.net = None

    def start(self):
        try:
            from mininet.net import Mininet
            from mininet.node import OVSKernelSwitch, OVSBridge, RemoteController
            from mininet.topo import Topo
        except ImportError as exc:
            raise RuntimeError("Mininet is required; run this adapter in Ubuntu/WSL2") from exc
        graph = self.topology

        class GraphTopo(Topo):
            def build(self):
                switches = {i: self.addSwitch(f"s{i + 1}") for i in range(graph.num_nodes)}
                for u, v, _, _ in graph.edges:
                    if u < v:
                        self.addLink(switches[u], switches[v])
                for i in range(graph.num_nodes):
                    host = self.addHost(f"h{i + 1}", ip=f"10.0.0.{i + 1}/24")
                    self.addLink(host, switches[i])

        switch_cls = OVSKernelSwitch if self.controller else OVSBridge
        kwargs = {"topo": GraphTopo(), "switch": switch_cls, "autoSetMacs": True,
                  "cleanup": True, "build": True, "controller": None}
        if self.controller:
            host, _, port = self.controller.partition(":")
            kwargs["controller"] = lambda name: RemoteController(name, ip=host, port=int(port or 6653))
        self.net = Mininet(**kwargs)
        self.net.start()
        self.started = True

    def ping_all(self) -> float:
        if not self.started or self.net is None:
            raise RuntimeError("adapter is not started")
        return float(self.net.pingAll())

    def install_forwarding(self, table: dict, destination: int) -> int:
        """Install simple destination rules through ovs-ofctl.

        The method is intentionally explicit about the destination and counts
        successful flow-mod commands, allowing controller-update overhead to be
        measured in emulation runs.
        """
        if not self.started or self.net is None:
            raise RuntimeError("adapter is not started")
        updates = 0
        for (node, dest), weights in table.items():
            if dest != destination:
                continue
            switch = self.net.get(f"s{node + 1}")
            for edge, weight in weights.items():
                out_node = edge[1] + 1
                command = f"ovs-ofctl add-flow {switch.name} priority=100,ip,nw_dst=10.0.0.{dest + 1},actions=output:{out_node}"
                switch.cmd(command)
                updates += 1
        return updates

    def controller_resources(self) -> dict:
        """Return process and interface counters for the current emulation."""
        sample = self.poll()
        rss = None
        try:
            with open("/proc/self/status", encoding="utf-8") as stream:
                for line in stream:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024
                        break
        except OSError:
            pass
        return {"interfaces": len(sample.port_bytes), "bytes": sum(sample.port_bytes.values()),
                "timestamp": sample.timestamp, "process_cpu_seconds": time.process_time(), "memory_bytes": rss}
    def poll(self) -> ControllerSample:
        if not self.started:
            raise RuntimeError("adapter is not started")
        bytes_by_interface = {}
        for switch in self.net.switches:
            for intf in switch.intfList():
                name = intf.name
                if not name or name == "lo":
                    continue
                try:
                    rx = int(switch.cmd(f"cat /sys/class/net/{name}/statistics/rx_bytes").strip() or 0)
                    tx = int(switch.cmd(f"cat /sys/class/net/{name}/statistics/tx_bytes").strip() or 0)
                    bytes_by_interface[name] = rx + tx
                except (ValueError, OSError):
                    continue
        sample = ControllerSample(time.time(), bytes_by_interface)
        self.samples.append(sample)
        return sample

    def stop(self):
        if self.net is not None:
            self.net.stop()
        self.started = False
        self.net = None
