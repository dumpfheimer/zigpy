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
    def build_route(self, tsn: int, ping: bool) -> list[t.NWK] | None:
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

    async def establish_route(self, routes: list[list[t.NWK]], max_tries=3) -> list[t.NWK] | None:
        """Try to establish a route using the given routes"""
        for route in routes:
            if max_tries == 0: return None
            max_tries -= 1
            LOGGER.debug("Trying route for %s: %s", self.device.nwk, route)
            try:
                if await self.device.application.establish_route(self.device.nwk, route):
                    LOGGER.debug("Establishing route to %s via %s succeeded", self.device.nwk, route)
                    try:
                        if await self.device.ping_using_route_works(route):
                            LOGGER.debug("Ping to %s succeeded using route %s", self.device.nwk, route)
                            return route
                        else:
                            LOGGER.debug("Ping to %s failed using route %s", self.device.nwk, route)
                    except TimeoutError:
                        LOGGER.debug("Ping to %s timed out using route %s", self.device.nwk, route)
                else:
                    LOGGER.debug("Establishing route to %s via %s failed", self.device.nwk, route)
            except Exception as e:
                LOGGER.debug("Error establishing route to %s via %s: %s", self.device.nwk, route, e, exc_info=e)
        return None

    def is_usable(self):
        return self.last_was_successful and self.last_lqi > 80


class DirectRoute(RouteBase):
    """This route sends the packet directly to the device without any relays"""
    def build_route(self, tsn: int, ping: bool) -> list[t.NWK] | None:
        return []

class AutomaticRoute(RouteBase):
    """This route lets the coordinator choose the route"""
    def build_route(self, tsn: int, ping: bool) -> list[t.NWK] | None:
        return None

class ReportedRoute(RouteBase):
    def __init__(self, device: zigpy.device.Device, name: str) -> None:
        super().__init__(device, name)
        self.success_rate: dict[t.NWK, float] = {}
        self.last_route: list[t.NWK] | None = None
        self.last_successful_route: list[t.NWK] | None = None


    def build_route(self, tsn: int, ping: bool) -> list[t.NWK] | None:
        return self.device.relays[::-1]


class DeviceRouting:
    device: zigpy.device.Device

    def __init__(self, device: zigpy.device.Device) -> None:
        self.device = device

        self.direct_route: DirectRoute = DirectRoute(device, "direct")
        self.automatic_route: AutomaticRoute = AutomaticRoute(device, "automatic")
        self.reported_route: ReportedRoute = ReportedRoute(device, "reported")
        self.tsn_route: dict[int, RouteBase] = {}
        self.last_ping_route: RouteBase | None = None
        self.last_ping_tsn: int | None = None
        self.packages_received: int = 0
        self.next_hop: zigpy.device.Device | None = None

    def _build_route_ping(self, tsn: int) -> list[t.NWK] | None:
        LOGGER.debug("First ping for %s. current ping route: %s", self.device.nwk, self.last_ping_route)
        # only change once per ping
        if self.last_ping_route is None:
            LOGGER.debug("Using direct route as ping route for %s because of lack of data", self.device.nwk)
            self.last_ping_route = self.direct_route
        elif isinstance(self.last_ping_route, DirectRoute) and self.last_ping_route.is_usable():
            LOGGER.debug("Using direct route as ping route for %s (because it works and is the best)", self.device.nwk)

        elif self.last_ping_route == self.automatic_route:
            LOGGER.debug("Using direct route as ping route for %s", self.device.nwk)
            self.last_ping_route = self.direct_route

        elif self.last_ping_route == self.direct_route:
            LOGGER.debug("Using reported route as ping route for %s", self.device.nwk)
            self.last_ping_route = self.reported_route

        elif self.last_ping_route == self.reported_route:
            LOGGER.debug("Using automatic route as ping route for %s", self.device.nwk)
            self.last_ping_route = self.automatic_route

        else:
            LOGGER.debug("Using direct route as ping route for %s", self.device.nwk)
            self.last_ping_route = self.direct_route

        self.last_ping_tsn = tsn
        self.tsn_route[tsn] = self.last_ping_route
        return self.last_ping_route.build_route(tsn, True)

    def build_route(self, tsn: int, ping: bool) -> list[t.NWK] | None:
        LOGGER.debug("build_route called with parameters: tsn=%s, ping=%s, attempt=%s, max_attempts=%s for %s", tsn, ping, attempt, max_attempts, self.device.nwk)
        # if we are pinging the device, we can try routes. on requests we want the highest success rate
        if ping:
            return self._build_route_ping(tsn)

        route = None

        # let ping determine if direct route is possible
        if self.direct_route.average_lqi >= 80 and self.direct_route.last_was_successful:
            LOGGER.debug("Using direct route for %s because lqi is good", self.device.nwk)
            self.tsn_route[tsn] = self.direct_route
            return []
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

        # remember route we took for tsn
        self.tsn_route[tsn] = route

        return route.build_route(tsn, ping, attempt, max_attempts)

    def _notify_route_error_ping(self) -> None:
        LOGGER.warning("Ping failed")
        self.last_ping_route.packages_lost += 1
        #self.last_ping_route = self.direct_route

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
        #self.last_ping_route = self.direct_route

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