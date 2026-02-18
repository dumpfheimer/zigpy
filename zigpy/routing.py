from __future__ import annotations

import logging
from abc import abstractmethod
from datetime import UTC, datetime

import zigpy.device
import zigpy.types as t


LOGGER = logging.getLogger(__name__)

class RouteBase:
    packages_sent: int
    packages_received: int
    packages_lost: int
    average_lqi: float

    def __init__(self, device: zigpy.device.Device) -> None:
        self.packages_sent = 0
        self.packages_received = 0
        self.packages_lost = 0
        self.average_lqi = 0
        self.last_packet_received = datetime.now(UTC)
        self.last_was_successful = False

    def packet_received(self, packet: t.ZigbeePacket) -> None:
        self.average_lqi = ((self.average_lqi * self.packages_received) + float(packet.lqi)) / (self.packages_received + 1)
        self.packages_received += 1
        self.last_was_successful = True

    @abstractmethod
    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        raise NotImplementedError

    def notify_route_error(self, tsn):
        self.last_was_successful = False
        self.packages_lost += 1

    def notify_timeout(self, tsn):
        self.last_was_successful = False
        self.packages_lost += 1


class DirectRoute(RouteBase):
    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        return []

class AutomaticRoute(RouteBase):
    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        return None

class TopologyRoute(RouteBase):
    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        # TODO: actually build a route
        return None

class DeviceRouting:
    device: zigpy.device.Device

    def __init__(self, device: zigpy.device.Device) -> None:
        self.device = device

        self.direct_route: DirectRoute = DirectRoute(device)
        self.automatic_route: AutomaticRoute = AutomaticRoute(device)
        self.topology_route: TopologyRoute = TopologyRoute(device)
        self.tsn_route: dict[int, RouteBase] = {}
        self.last_ping_route: RouteBase | None = None
        self.last_ping_tsn: int | None = None
        self.packages_received: int = 0

    def _build_route_ping(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        if attempt == 1:
            # only change once per ping
            if self.last_ping_route is None:
                self.last_ping_route = self.automatic_route

            elif self.last_ping_route == self.direct_route:
                self.last_ping_route = self.automatic_route

            elif self.last_ping_route == self.automatic_route:
                self.last_ping_route = self.topology_route

            else:
                self.last_ping_route = self.direct_route

        self.last_ping_tsn = tsn
        self.tsn_route[tsn] = self.last_ping_route
        return self.last_ping_route.build_route(tsn, ping, attempt, max_attempts)

    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        # if we are pinging the device, we can try routes. on requests we want the highest success rate
        if ping:
            return self._build_route_ping(tsn, ping, attempt, max_attempts)

        route = None

        # let ping determine if direct route is possible
        if self.direct_route.packages_sent == 0:
            LOGGER.debug("Using automatic route because of lack of data")
            route = self.automatic_route

        # use direct route if lqi is high enough
        elif self.direct_route.average_lqi >= 80 and self.direct_route.last_was_successful:
            LOGGER.debug("Using direct route because lqi is good")
            self.tsn_route[tsn] = self.direct_route
            return []
        # TODO: utilize topology
        # default route is automatic route
        elif self.automatic_route.last_was_successful:
            LOGGER.debug("Using automatic route because direct route failed or had bad lqi")
            route = self.automatic_route
        elif self.direct_route.average_lqi >= 80:
            LOGGER.debug("Using direct route because lqi is good and both routes failed")
            route = self.direct_route
        else:
            LOGGER.debug("Using automatic route as default")
            route = self.automatic_route

        # remember route we took for tsn
        self.tsn_route[tsn] = route

        return route.build_route(tsn, ping, attempt, max_attempts)

    def _notify_route_error_ping(self) -> None:
        LOGGER.warning("Ping failed")
        self.last_ping_route.packages_lost += 1
        self.last_ping_route = self.direct_route

    def notify_route_error(self, tsn: int) -> None:
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
        if tsn == self.last_ping_tsn:
            self._notify_route_error_ping()

        packet_route = self.tsn_route.get(tsn)
        if packet_route is not None:
            packet_route.notify_timeout(tsn)

    def _packet_received_ping(self) -> None:
        pass

    def packet_received(self, packet: t.ZigbeePacket) -> None:
        if packet.tsn == self.last_ping_tsn:
            self._packet_received_ping()

        packet_route = self.tsn_route.get(packet.tsn)
        if packet_route is not None:
            packet_route.packet_received(packet)
        return

