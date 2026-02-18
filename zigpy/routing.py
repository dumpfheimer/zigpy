from __future__ import annotations

import logging
from abc import abstractmethod
from datetime import UTC, datetime

import zigpy.device
import zigpy.types as t
from zigpy.zdo import ZDO

LOGGER = logging.getLogger(__name__)

class RouteBase:
    packages_received: int
    packages_lost: int
    average_lqi: float

    def __init__(self, device: zigpy.device.Device, name: str) -> None:
        self.device = device
        self.name = name
        self.packages_received = 0
        self.packages_lost = 0
        self.average_lqi = 0
        self.last_packet_received = datetime.now(UTC)
        self.last_was_successful = False
        self.last_lqi = 0

    def packet_received(self, packet: t.ZigbeePacket) -> None:
        LOGGER.debug("Received packet for %s (%s) tsn %s (aps tsn %s)", self.device.nwk, self.name, packet.tsn, packet.data.value[0])
        self.average_lqi = ((self.average_lqi * self.packages_received) + float(packet.lqi)) / (self.packages_received + 1)
        self.packages_received += 1
        self.last_was_successful = True
        self.last_lqi = packet.lqi

    @abstractmethod
    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        raise NotImplementedError

    def notify_route_error(self, tsn):
        LOGGER.debug("Received route error for %s (%s) tsn %s", self.device.nwk, self.name, tsn)
        self.last_was_successful = False
        self.packages_lost += 1

    def notify_timeout(self, tsn):
        LOGGER.debug("Received timeout for %s (%s) tsn %s", self.device.nwk, self.name, tsn)
        self.last_was_successful = False
        self.packages_lost += 1


class DirectRoute(RouteBase):
    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        return []

class AutomaticRoute(RouteBase):
    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        return None

class TopologyRoute(RouteBase):
    def __init__(self, device: zigpy.device.Device, name: str) -> None:
        super().__init__(device, name)
        self.success_rate: dict[t.NWK, float] = {}
        self.last_route: list[t.NWK] | None = None
        self.last_successful_route: list[t.NWK] | None = None

    def one_hop_route(self):
        coordinator_neighbors = self.device.application.topology.neighbors.get(self.device.application.get_device_with_address(t.AddrModeAddress(t.AddrMode.NWK, t.NWK(0x0000))).ieee)
        LOGGER.debug("Found coordinator neighbors %s", coordinator_neighbors)
        device_neighbors = self.device.application.topology.neighbors.get(self.device.ieee)
        LOGGER.debug("Found device neighbors %s", device_neighbors)
        if coordinator_neighbors is None or device_neighbors is None or len(coordinator_neighbors) == 0 or len(device_neighbors) == 0:
            return None

        # filter for lqi > 80
        coordinator_neighbors = [n for n in coordinator_neighbors if n.lqi > 80]
        device_neighbors = [n for n in device_neighbors if n.lqi > 80]
        shared_neighbors = []
        for cn in coordinator_neighbors:
            for dn in device_neighbors:
                if cn.nwk == dn.nwk:
                    shared_neighbors.append(cn)
        LOGGER.debug("Found shared neighbors %s", shared_neighbors)
        if len(shared_neighbors) == 0:
            return None
        # sort by lqi
        shared_neighbors = sorted(shared_neighbors, key=lambda n: n.lqi)
        self.last_route = [shared_neighbors[0].nwk]
        # return highest lqi neighbor
        return self.last_route

    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        if not ping and self.last_successful_route is not None:
            LOGGER.debug("Returning last successful route for %s", self.device.nwk)
            return self.last_successful_route
        LOGGER.debug("Building route for %s", self.device.nwk)
        one_hop_route = self.one_hop_route()
        if one_hop_route is not None:
            LOGGER.debug("Returning one hop route for %s: %s", self.device.nwk, one_hop_route)
            return one_hop_route
        LOGGER.debug("No one hop route found for %s", self.device.nwk)
        return None

    def packet_received(self, packet: t.ZigbeePacket) -> None:
        previous_lqi = self.last_lqi
        super().packet_received(packet)
        # make this route the new successful route
        if packet.lqi > previous_lqi:
            LOGGER.debug("New successful topology route for %s: %s", self.device.nwk, self.last_route)
            self.last_successful_route = self.last_route

class DeviceRouting:
    device: zigpy.device.Device

    def __init__(self, device: zigpy.device.Device) -> None:
        self.device = device

        self.direct_route: DirectRoute = DirectRoute(device, "direct")
        self.automatic_route: AutomaticRoute = AutomaticRoute(device, "automatic")
        self.topology_route: TopologyRoute = TopologyRoute(device, "topology")
        self.tsn_route: dict[int, RouteBase] = {}
        self.last_ping_route: RouteBase | None = None
        self.last_ping_tsn: int | None = None
        self.packages_received: int = 0

    def _build_route_ping(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        if attempt == 1:
            # only change once per ping
            if self.last_ping_route is None:
                LOGGER.debug("Using automatic route as ping route for %s because of lack of data", self.device.nwk)
                self.last_ping_route = self.automatic_route

            elif self.last_ping_route == self.automatic_route:
                LOGGER.debug("Using direct route as ping route for %s", self.device.nwk)
                self.last_ping_route = self.direct_route

            elif self.last_ping_route == self.direct_route:
                LOGGER.debug("Using topology route as ping route for %s", self.device.nwk)
                self.last_ping_route = self.topology_route

            elif self.last_ping_route == self.topology_route:
                LOGGER.debug("Using automatic route as ping route for %s", self.device.nwk)
                self.last_ping_route = self.automatic_route

            else:
                LOGGER.debug("Using direct route as ping route for %s", self.device.nwk)
                self.last_ping_route = self.direct_route
        else:
            LOGGER.debug("Using last ping route for %s", self.device.nwk)

        self.last_ping_tsn = tsn
        self.tsn_route[tsn] = self.last_ping_route
        return self.last_ping_route.build_route(tsn, ping, attempt, max_attempts)

    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        LOGGER.debug("build_route called with parameters: tsn=%s, ping=%s, attempt=%s, max_attempts=%s for %s", tsn, ping, attempt, max_attempts, self.device.nwk)
        # if we are pinging the device, we can try routes. on requests we want the highest success rate
        if ping:
            return self._build_route_ping(tsn, ping, attempt, max_attempts)

        route = None

        # let ping determine if direct route is possible
        if self.direct_route.average_lqi >= 80 and self.direct_route.last_was_successful:
            LOGGER.debug("Using direct route for %s because lqi is good", self.device.nwk)
            self.tsn_route[tsn] = self.direct_route
            return []
        # TODO: utilize topology
        # default route is automatic route
        elif self.topology_route.last_was_successful and self.topology_route.last_lqi >= 80:
            LOGGER.debug("Using topology route for %s because other routes failed or had bad lqi", self.device.nwk)
            route = self.topology_route
        elif self.automatic_route.last_was_successful and self.automatic_route.last_lqi >= 80:
            LOGGER.debug("Using automatic route for %s because direct route failed or had bad lqi", self.device.nwk)
            route = self.automatic_route
        elif self.direct_route.average_lqi >= 80:
            LOGGER.debug("Using direct route for %s because lqi is good and both routes failed", self.device.nwk)
            route = self.direct_route
        else:
            LOGGER.debug("Using automatic route for %s as default", self.device.nwk)
            route = self.automatic_route

        # remember route we took for tsn
        self.tsn_route[tsn] = route

        return route.build_route(tsn, ping, attempt, max_attempts)

    def _notify_route_error_ping(self) -> None:
        LOGGER.warning("Ping failed")
        self.last_ping_route.packages_lost += 1
        self.last_ping_route = self.direct_route

    def notify_route_error(self, tsn: int) -> None:
        LOGGER.debug("Received route error for %s tsn %s", self.device.nwk, tsn)
        if tsn == self.last_ping_tsn:
            self._notify_route_error_ping()

        packet_route = self.tsn_route.get(tsn)
        if packet_route is not None:
            packet_route.notify_route_error(tsn)

    def _notify_timeout_ping(self) -> None:
        LOGGER.warning("Ping failed")
        self.last_ping_route.packages_lost += 1
        self.last_ping_route = self.direct_route

    def notify_timeout(self, tsn: int) -> None:
        LOGGER.debug("Received timeout for %s tsn %s", self.device.nwk, tsn)
        if tsn == self.last_ping_tsn:
            self._notify_route_error_ping()

        packet_route = self.tsn_route.get(tsn)
        if packet_route is not None:
            packet_route.notify_timeout(tsn)

    def _packet_received_ping(self, packet: t.ZigbeePacket) -> None:
        LOGGER.debug("Received packet for ping of %s tsn %s", self.device.nwk, packet.tsn)
        if self.last_ping_route is not None:
            self.last_ping_route.packet_received(packet)

    def packet_received(self, packet: t.ZigbeePacket, endpoint, zcl_cluster) -> None:
        LOGGER.debug("Received packet for %s tsn %s endpoint %s cluster %s", self.device.nwk, packet.tsn, endpoint, zcl_cluster)
        if packet.src_ep == 0 and len(packet.data.value) > 0 and packet.data.value[0] == self.last_ping_tsn:
            LOGGER.debug("Received packet for ping of %s data %s", self.device.nwk, packet.data)
            self._packet_received_ping(packet)

        packet_route = self.tsn_route.get(packet.tsn)
        if packet_route is not None:
            packet_route.packet_received(packet)
        return

