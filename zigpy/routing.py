from __future__ import annotations

import logging
from abc import abstractmethod
from datetime import UTC, datetime

import zigpy.device
import zigpy.types as t
import zigpy.zdo.types as zdo_t

LOGGER = logging.getLogger(__name__)

class RouteBase:
    """Base class for route generation"""

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
        """A packet was received (in response to a ping)"""
        LOGGER.debug("Received packet for %s (%s) tsn %s (aps tsn %s)", self.device.nwk, self.name, packet.tsn, packet.data.value[0])
        self.average_lqi = ((self.average_lqi * self.packages_received) + float(packet.lqi)) / (self.packages_received + 1)
        self.packages_received += 1
        self.last_was_successful = True
        self.last_lqi = packet.lqi

    @abstractmethod
    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        """Build a route for the given parameters"""
        raise NotImplementedError

    def notify_route_error(self, tsn):
        """A route error message was received"""
        LOGGER.debug("Received route error for %s (%s) tsn %s", self.device.nwk, self.name, tsn)
        self.last_was_successful = False
        self.packages_lost += 1

    def notify_timeout(self, tsn):
        """A timeout message was received"""
        LOGGER.debug("Received timeout for %s (%s) tsn %s", self.device.nwk, self.name, tsn)
        self.last_was_successful = False
        self.packages_lost += 1


class DirectRoute(RouteBase):
    """This route sends the packet directly to the device without any relays"""
    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        return []

class AutomaticRoute(RouteBase):
    """This route lets the coordinator choose the route"""
    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        return None

class TopologyRoute(RouteBase):
    """This rout utilizes the topology information within zigpy to detect the best route (currently only one and two hops supported)"""
    def __init__(self, device: zigpy.device.Device, name: str) -> None:
        super().__init__(device, name)
        self.success_rate: dict[t.NWK, float] = {}
        self.last_route: list[t.NWK] | None = None
        self.last_successful_route: list[t.NWK] | None = None
        self.timeouts: dict[t.NWK, int] = {}

    def _neighbors_array_contains_nwk(self, arr: list[zdo_t.Neighbor], nwk: t.NWK) -> bool:
        for neighbor in arr:
            if neighbor.nwk == nwk:
                return True
        return False

    def _intersect_neighbors(self, arr1: list[zdo_t.Neighbor], arr2: list[zdo_t.Neighbor]) -> list[zdo_t.Neighbor]:
        shared_neighbors: list[zdo_t.Neighbor] = []
        for n1 in arr1:
            for n2 in arr2:
                if n1.nwk == n2.nwk:
                    shared_neighbors.append(n1)
        return shared_neighbors

    def _add_if_missing(self, arr: list[zdo_t.Neighbor], neighbor: zdo_t.Neighbor) -> None:
        for n in arr:
            if n.nwk == neighbor.nwk:
                return
        arr.append(neighbor)

    def _all_neighbors(self, nwk: t.NWK) -> list[zdo_t.Neighbor]:
        neighbors: list[zdo_t.Neighbor] = []
        own_neighbors = self.device.application.topology.neighbors.get(self.device.application.get_device_with_address(t.AddrModeAddress(t.AddrMode.NWK, nwk)).ieee)
        if own_neighbors is not None:
            for n in own_neighbors:
                if n.lqi > 80:
                    self._add_if_missing(neighbors, n)
        for device in self.device.application.devices.values():
            if device.nwk == nwk:
                continue
            device_neighbors = self.device.application.topology.neighbors.get(device.ieee)
            if device_neighbors is not None:
                for n in device_neighbors:
                    if n.lqi > 80 and n.nwk == nwk and not self._neighbors_array_contains_nwk(neighbors, n.nwk):
                        self._add_if_missing(neighbors, n)

        return neighbors

    def _best_lqi_neighbor(self, neighbors: list[zdo_t.Neighbor]) -> zdo_t.Neighbor | None:
        if len(neighbors) == 0:
            return None
        return sorted(neighbors, key=lambda n: n.lqi)[len(neighbors) - 1]

    def _filter_bad(self, neighbors: list[zdo_t.Neighbor]) -> list[zdo_t.Neighbor]:
        return [n for n in neighbors if n.nwk not in self.timeouts]

    def _best_relay_for(self, src: t.NWK, dst: t.NWK, allow_bad: bool = False) -> tuple[t.NWK | None, int]:
        src_neighbors = self._all_neighbors(src)
        dest_neighbors = self._all_neighbors(dst)

        best_combined_lqi = 0
        best_relay = None
        best_banned_lqi = 0
        best_banned_relay = None

        for src_neighbor in src_neighbors:
            if src_neighbor.device_type == zdo_t.DeviceType.Router:
                for dest_neighbor in dest_neighbors:
                    if dest_neighbor.device_type == zdo_t.DeviceType.Router:
                        if src_neighbor.nwk == dest_neighbor.nwk:
                            combined_lqi = src_neighbor.lqi + dest_neighbor.lqi
                            if combined_lqi > best_combined_lqi:
                                if src_neighbor.nwk not in self.timeouts:
                                    best_combined_lqi = combined_lqi
                                    best_relay = src_neighbor.nwk
                                best_banned_lqi = src_neighbor.lqi
                                best_banned_relay = src_neighbor.nwk

        if best_relay is None and allow_bad:
            LOGGER.debug("No non-bad relay found for %s -> %s using %s", src, dst, best_banned_relay)
            best_relay = best_banned_relay
            best_combined_lqi = best_banned_lqi
        LOGGER.debug("Best relay for %s -> %s is %s with combined lqi %s", src, dst, best_relay, best_combined_lqi)
        return best_relay, best_combined_lqi


    def one_hop_route(self, allow_bad: bool = False) -> list[t.NWK] | None:
        best_relay, _ = self._best_relay_for(t.NWK(0x0000), self.device.nwk, allow_bad=allow_bad)
        return [best_relay] if best_relay is not None else None

    def _best_two_hop_route(self, src: t.NWK, dst: t.NWK, allow_bad: bool = False) -> tuple[t.NWK | None, t.NWK | None, int]:
        best_combined_lqi = 0
        best_hop1 = None
        best_hop2 = None

        src_neighbors = self._all_neighbors(src)
        for src_neighbor in src_neighbors:
            best_relay, combined_lqi = self._best_relay_for(src_neighbor.nwk, self.device.nwk, allow_bad=allow_bad)
            if combined_lqi > best_combined_lqi:
                best_combined_lqi = combined_lqi
                best_hop1 = src_neighbor
                best_hop2 = best_relay

        dst_neighbors = self._all_neighbors(dst)
        for dst_neighbor in dst_neighbors:
            best_relay, combined_lqi = self._best_relay_for(dst_neighbor.nwk, self.device.nwk, allow_bad=allow_bad)
            if combined_lqi > best_combined_lqi:
                best_combined_lqi = combined_lqi
                best_hop1 = dst_neighbor
                best_hop2 = best_relay

        return best_hop1, best_hop2, best_combined_lqi


    def two_hop_route(self, allow_bad: bool = False):
        hop1, hop2, _ = self._best_two_hop_route(t.NWK(0x0000), self.device.nwk, allow_bad=allow_bad)
        if hop1 is not None and hop2 is not None:
            return [hop1, hop2]
        return None

    def notify_timeout(self, tsn):
        LOGGER.debug("Received timeout for %s (%s) tsn %s", self.device.nwk, self.name, tsn)
        self.last_was_successful = False
        self.packages_lost += 1
        LOGGER.warning("Timeout on n hop route for %s (%s) tsn %s (route %s)", self.device.nwk, self.name, tsn, self.last_route)
        for nwk in self.last_route:
            if not nwk in self.timeouts:
                self.timeouts[nwk] = 0
            self.timeouts[nwk] += 1
        LOGGER.warning("%s current bad relays: %s", self.device.nwk, self.timeouts)

    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        if not ping and self.last_successful_route is not None:
            LOGGER.debug("Returning last successful route for %s", self.device.nwk)
            return self.last_successful_route
        LOGGER.debug("Building route for %s", self.device.nwk)
        route: list[t.NWK] | None = self.one_hop_route()
        if route is not None:
            LOGGER.debug("Returning one hop route for %s: %s", self.device.nwk, route)
        else:
            LOGGER.debug("No one hop route found for %s", self.device.nwk)
            route = self.two_hop_route()
            if route is not None:
                LOGGER.debug("Returning two hop route for %s: %s", self.device.nwk, route)
            else:
                LOGGER.debug("No two hop route found for %s", self.device.nwk)
                route = self.one_hop_route(allow_bad=True)
                LOGGER.debug(
                    "Returning one hop route (allow_bad=True) for %s: %s", self.device.nwk, route
                )
                if route is not None:
                    LOGGER.debug("Returning two hop route for %s: %s", self.device.nwk, route)
                else:
                    LOGGER.debug("No one hop route found for %s using bad", self.device.nwk)
                    route = self.two_hop_route(allow_bad=True)
                    if route is not None:
                        LOGGER.debug("Returning two hop route for %s: %s", self.device.nwk, route)

        self.last_route = route
        return route

    def packet_received(self, packet: t.ZigbeePacket) -> None:
        previous_lqi = self.last_lqi
        super().packet_received(packet)
        # make this route the new successful route
        if packet.lqi > previous_lqi and self.last_route is not None:
            LOGGER.debug("New successful topology route for %s: %s", self.device.nwk, self.last_route)
            self.last_successful_route = self.last_route

class ReportedRoute(RouteBase):
    def __init__(self, device: zigpy.device.Device, name: str) -> None:
        super().__init__(device, name)
        self.success_rate: dict[t.NWK, float] = {}
        self.last_route: list[t.NWK] | None = None
        self.last_successful_route: list[t.NWK] | None = None


    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        return self.device.relays[::-1]


class DeviceRouting:
    device: zigpy.device.Device

    def __init__(self, device: zigpy.device.Device) -> None:
        self.device = device

        self.direct_route: DirectRoute = DirectRoute(device, "direct")
        self.automatic_route: AutomaticRoute = AutomaticRoute(device, "automatic")
        self.topology_route: TopologyRoute = TopologyRoute(device, "topology")
        self.reported_route: ReportedRoute = ReportedRoute(device, "reported")
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
                LOGGER.debug("Using reported route as ping route for %s", self.device.nwk)
                self.last_ping_route = self.reported_route

            elif self.last_ping_route == self.reported_route:
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
        elif self.reported_route.average_lqi >= 80 and self.reported_route.last_was_successful:
            LOGGER.debug("Using reported route for %s", self.device.nwk)
            route = self.reported_route
        else:
            LOGGER.debug("Using automatic route for %s as default", self.device.nwk)
            route = self.automatic_route

        if attempt > 2 and attempt == max_attempts - 1:
            LOGGER.debug("Using direct route for %s because its close to max_attempts", self.device.nwk)
            route = self.direct_route
        if attempt > 2 and attempt == max_attempts:
            LOGGER.debug("Using topology route for %s because its the last attempt", self.device.nwk)
            route = self.topology_route

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

