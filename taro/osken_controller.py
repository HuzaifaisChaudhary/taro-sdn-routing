"""Minimal OS-Ken OpenFlow 1.3 controller for Mininet validation."""
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib.packet import packet, ethernet


class TaroOSKenController(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.mac_to_port = {}

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features(self, ev):
        dp = ev.msg.datapath; parser = dp.ofproto_parser; ofp = dp.ofproto
        match = parser.OFPMatch(); actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        dp.send_msg(parser.OFPFlowMod(datapath=dp, priority=0, match=match,
                                      instructions=[parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]))

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in(self, ev):
        msg = ev.msg; dp = msg.datapath; parser = dp.ofproto_parser; ofp = dp.ofproto
        pkt = packet.Packet(msg.data); eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None: return
        port = msg.match.get("in_port"); self.mac_to_port.setdefault(dp.id, {})[eth.src] = port
        out = self.mac_to_port[dp.id].get(eth.dst, ofp.OFPP_FLOOD); actions = [parser.OFPActionOutput(out)]
        if out != ofp.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=port, eth_dst=eth.dst)
            dp.send_msg(parser.OFPFlowMod(datapath=dp, priority=10, match=match,
                                          instructions=[parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]))
        dp.send_msg(parser.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id, in_port=port, actions=actions,
                                        data=msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None))
