from __future__ import annotations

import logging
from abc import abstractmethod
from datetime import UTC, datetime

import zigpy.device
import zigpy.types as t
import zigpy.zdo.types as zdo_t
from zigpy.zdo.types import Neighbor

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

    async def establish_route(self, routes: list[list[t.NWK]]) -> list[t.NWK] | None:
        """Try to establish a route using the given routes"""
        for route in routes:
            LOGGER.debug("Trying route for %s: %s", self.device.nwk, route)
            if await self.device.application.establish_route(self.device.nwk, route):
                LOGGER.debug("Establishing route to %s via %s succeeded", self.device.nwk, route)
                if await self.device.ping_using_route_works(route):
                    LOGGER.debug("Ping to %s succeeded using route %s", self.device.nwk, route)
                    return route
                else:
                    LOGGER.debug("Ping to %s failed using route %s", self.device.nwk, route)
            else:
                LOGGER.debug("Establishing route to %s via %s failed", self.device.nwk, route)
        return None

    def is_usable(self):
        return self.last_was_successful and self.last_lqi > 80


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
        self.tsn_route: dict[int, list[t.NWK]] = {}
        self.bad_routes: list[list[t.NWK]] = []

    def has_good_route(self):
        return self.last_successful_route is not None \
                and len(self.last_successful_route) <= 2 \
                and self.last_was_successful

    def _get_routes_to_coordinator(self, max_hops=2) -> list[list[t.NWK]] | None:
        if self.device._routing.direct_route.is_usable():
            return []
        if max_hops == 0: return None

        ret = []
        all_hops: list[zdo_t.Route] = self.device.application.topology.routes.get(self.device.ieee)
        if all_hops is None: return None
        for h in all_hops:
            if h.RouteStatus == zdo_t.RouteStatus.Active:
                try:
                    device = self.device.application.get_device(nwk=h.NextHop)
                    if device is not None and device.node_desc.is_router:
                        child_routes: list[list[t.NWK]] = device._routing.topology_route._get_routes_to_coordinator(max_hops - 1)
                        if child_routes is not None:
                            if len(child_routes) == 0:
                                ret.append([h.NextHop])
                            else:
                                for child_route in child_routes:
                                    route = child_route + [h.NextHop]
                                    LOGGER.debug("Route to coordinator: %s", route)
                                    if route is not None: ret.append(route)
                except KeyError:
                    # device not found
                    pass

        return None if len(ret) == 0 else ret


    def reported_routes(self) -> list[list[t.NWK]]:
        routes = self._get_routes_to_coordinator()
        if routes is None:
            LOGGER.debug("No routes found for %s", self.device.nwk)
            return []
        routes = [route for route in routes if route not in self.bad_routes]

        LOGGER.debug("Topology routes for %s: %s", self.device.nwk, routes)
        return routes

    def notify_timeout(self, tsn):
        LOGGER.debug("Received timeout for %s (%s) tsn %s", self.device.nwk, self.name, tsn)
        self.last_was_successful = False
        self.packages_lost += 1
        LOGGER.warning("Timeout on n hop route for %s (%s) tsn %s (route %s)", self.device.nwk, self.name, tsn, self.last_route)
        self.bad_routes.append(self.last_route)
        LOGGER.warning("%s current bad routes: %s", self.device.nwk, self.bad_routes)

    async def scan_routes(self) -> bool:
        LOGGER.debug("Scanning routes for %s", self.device.nwk)

        reported_routes = self.reported_routes()
        working_route = await self.establish_route(reported_routes)
        if working_route is not None:
            LOGGER.debug("Established route for %s: %s", self.device.nwk, working_route)
            self.last_successful_route = working_route
            self.last_was_successful = True
            self.last_lqi = 100 # TODO: do something better
            return True
        LOGGER.debug("No working route found for %s", self.device.nwk)
        return False


    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        if not ping and self.last_successful_route is not None:
            LOGGER.debug("Returning last successful route for %s", self.device.nwk)
            self.last_route = self.last_successful_route
            return self.last_successful_route
        LOGGER.debug("Building route for %s", self.device.nwk)
        #route: list[t.NWK] | None = self.one_hop_route()
        #if route is not None:
        #    LOGGER.debug("Returning one hop route for %s: %s", self.device.nwk, route)
        #else:
        #    LOGGER.debug("No one hop route found for %s", self.device.nwk)
        #    route = self.two_hop_route()
        #    if route is not None:
        #        LOGGER.debug("Returning two hop route for %s: %s", self.device.nwk, route)
        #    else:
        #        LOGGER.debug("No two hop route found for %s", self.device.nwk)
        #        route = self.one_hop_route(allow_bad=True)
        #        LOGGER.debug(
        #            "Returning one hop route (allow_bad=True) for %s: %s", self.device.nwk, route
        #        )
        #        if route is not None:
        #            LOGGER.debug("Returning two hop route for %s: %s", self.device.nwk, route)
        #        else:
        #            LOGGER.debug("No one hop route found for %s using bad", self.device.nwk)
        #            route = self.two_hop_route(allow_bad=True)
        #            if route is not None:
        #                LOGGER.debug("Returning two hop route for %s: %s", self.device.nwk, route)

        #route = self.reported_routes()

        self.last_route = self.last_successful_route
        return self.last_route

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
        self.next_hop: zigpy.device.Device | None = None

    def _build_route_ping(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        if attempt == 1:
            # only change once per ping
            if self.last_ping_route is None:
                LOGGER.debug("Using automatic route as ping route for %s because of lack of data", self.device.nwk)
                self.last_ping_route = self.automatic_route
            elif isinstance(self.last_ping_route, DirectRoute) and self.last_ping_route.last_was_successful:
                LOGGER.debug("Usint direct route as ping route for %s (because it works and is the best)", self.device.nwk)

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