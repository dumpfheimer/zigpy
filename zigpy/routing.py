from __future__ import annotations

import logging
from abc import abstractmethod
from datetime import UTC, datetime, timedelta

import zigpy.config as conf
import zigpy.device
import zigpy.types as t

LOGGER = logging.getLogger(__name__)

# How long a route stays blacklisted after a failure before it is retried
BAD_ROUTE_TTL = timedelta(minutes=5)

# Background ping backoff for routers without a usable route. Starts at
# PING_BACKOFF_BASE and doubles after every failed steady-phase probe, up to
# PING_BACKOFF_MAX, so an unreachable device is probed less and less often
# instead of once per second. Reset to the base whenever we hear from the
# device (see DeviceRouting.notify_seen).
PING_BACKOFF_BASE = timedelta(seconds=10)
PING_BACKOFF_MAX = timedelta(minutes=5)

# Number of background ping attempts after which a device is considered to have
# concluded its initial convergence. Until then we leave routing to the
# coordinator (automatic route) and do not penalize delivery failures; once
# converged we have a usable picture of the network and start being deliberate.
# Should match ControllerApplication._ping_loop's ``fast_attempts``.
PING_CONVERGENCE_ATTEMPTS = 4

# How many delivery failures a route may accumulate (without an intervening
# success) before it is demoted. Tolerates a single transient failure but
# reacts before the full retry budget is exhausted.
DELIVERY_FAILURE_THRESHOLD = 2

# Route discovery is expensive (network-wide broadcasts), so only spend it on a
# device we have actually heard from this recently. A device silent for longer
# is treated as probably offline; any received traffic re-enables discovery.
DISCOVERY_OFFLINE_AFTER = timedelta(minutes=5)

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
        # Consecutive delivery failures since the last success, used to demote a
        # route that has stopped delivering (see DeviceRouting.notify_delivery_failure).
        self.consecutive_delivery_failures = 0

    def packet_received(self, packet: t.ZigbeePacket) -> None:
        """A packet was received (in response to a ping)"""
        LOGGER.debug("Received packet for %s (%s) tsn %s (aps tsn %s)", self.device.nwk, self.name, packet.tsn, packet.data.value[0] if packet.data.value else None)
        self.average_lqi = ((self.average_lqi * self.packages_received) + float(packet.lqi)) / (self.packages_received + 1)
        self.packages_received += 1
        self.last_was_successful = True
        self.last_lqi = packet.lqi
        self.consecutive_delivery_failures = 0

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
    """Route computed from zigpy's topology (neighbor) tables.

    Builds a bidirectional neighbor graph and searches it for good source routes
    up to ``CONF_NWK_ROUTING_MAX_HOPS`` relays, preferring fewer hops then higher
    worst-link LQI, then verifies candidates end to end before adopting one.
    """
    def __init__(self, device: zigpy.device.Device, name: str) -> None:
        super().__init__(device, name)
        self.last_route: list[t.NWK] | None = None
        self.last_successful_route: list[t.NWK] | None = None
        # blacklisted routes -> time the failure was recorded (aged out via TTL)
        self.bad_routes: dict[tuple[t.NWK, ...], datetime] = {}

    def is_usable(self):
        return self.last_was_successful and self.last_lqi > 80 and self.has_good_route()

    def _max_hops(self) -> int:
        return self.device.application.config[conf.CONF_NWK_ROUTING_MAX_HOPS]

    def has_good_route(self):
        return self.last_successful_route is not None \
                and len(self.last_successful_route) <= self._max_hops() \
                and self.last_was_successful

    def _neighbor_graph(self, lqi_threshold: int) -> dict[t.NWK, dict[t.NWK, int]]:
        """Build a bidirectional neighbor graph from the latest topology scan.

        An edge ``a <-> b`` exists only when both nodes list each other in their
        neighbor tables above ``lqi_threshold``. Zigbee LQI is directional, so
        requiring both directions avoids picking asymmetric links. The edge
        weight is the worst (minimum) of the two directional LQIs. Only scanned
        routers have neighbor tables, so graph nodes are inherently routers (plus
        the coordinator).
        """
        app = self.device.application
        # node nwk -> {neighbor nwk: lqi as seen by node}
        seen: dict[t.NWK, dict[t.NWK, int]] = {}
        for ieee, neighbors in app.topology.neighbors.items():
            try:
                node = app.get_device(ieee=ieee).nwk
            except KeyError:
                continue
            seen[node] = {
                neighbor.nwk: neighbor.lqi
                for neighbor in neighbors
                if neighbor.lqi >= lqi_threshold
            }

        graph: dict[t.NWK, dict[t.NWK, int]] = {}
        for a, neighbors in seen.items():
            for b, lqi_ab in neighbors.items():
                lqi_ba = seen.get(b, {}).get(a)
                if lqi_ba is None:
                    # not bidirectional (or b was not scanned) -> skip
                    continue
                graph.setdefault(a, {})[b] = min(lqi_ab, lqi_ba)
        return graph

    def _build_candidate_routes(
        self, max_hops: int, lqi_threshold: int = 80, max_candidates: int = 5
    ) -> list[list[t.NWK]]:
        """Find candidate source routes from the coordinator to this device.

        Enumerates simple paths through the bidirectional neighbor graph (up to
        ``max_hops`` relays) and ranks them by fewest relays, then highest
        bottleneck (worst-link) LQI. Returns relay lists excluding the
        coordinator and the destination, e.g. ``[A, B]`` for COORD -> A -> B ->
        device. An empty list is a direct (no-relay) route.
        """
        graph = self._neighbor_graph(lqi_threshold)
        coordinator = t.NWK(0x0000)
        target = self.device.nwk

        found: list[tuple[list[t.NWK], int]] = []  # (relays, bottleneck lqi)

        def visit(current: t.NWK, path: list[t.NWK], bottleneck: int) -> None:
            for neighbor, lqi in graph.get(current, {}).items():
                if neighbor in path:
                    continue
                hop_bottleneck = min(bottleneck, lqi)
                if neighbor == target:
                    # only relayed routes; the direct route is handled separately
                    if len(path) > 1:
                        found.append((path[1:], hop_bottleneck))
                elif len(path) <= max_hops:
                    visit(neighbor, path + [neighbor], hop_bottleneck)

        visit(coordinator, [coordinator], 255)

        # fewest relays first, then best bottleneck LQI
        found.sort(key=lambda c: (len(c[0]), -c[1]))

        ranked: list[list[t.NWK]] = []
        seen_keys: set[tuple[t.NWK, ...]] = set()
        for relays, _ in found:
            key = tuple(relays)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            ranked.append(relays)
            if len(ranked) >= max_candidates:
                break

        if ranked:
            LOGGER.debug("Candidate routes for %s: %s", self.device.nwk, ranked)
        return ranked

    def _is_bad_route(self, route: list[t.NWK]) -> bool:
        """Whether a route is currently blacklisted (failures age out via TTL)."""
        key = tuple(route)
        added = self.bad_routes.get(key)
        if added is None:
            return False
        if datetime.now(UTC) - added > BAD_ROUTE_TTL:
            del self.bad_routes[key]
            return False
        return True

    def _mark_bad_route(self, route: list[t.NWK] | None) -> None:
        if route is None:
            return
        LOGGER.debug("Marking route to %s as bad: %s", self.device.nwk, route)
        self.bad_routes[tuple(route)] = datetime.now(UTC)

    def notify_timeout(self, tsn):
        LOGGER.debug("Received timeout for %s (%s) tsn %s", self.device.nwk, self.name, tsn)
        self.last_was_successful = False
        self.packages_lost += 1
        LOGGER.warning("Timeout on n hop route for %s (%s) tsn %s (route %s)", self.device.nwk, self.name, tsn, self.last_route)
        self._mark_bad_route(self.last_route)

    async def scan_routes(self) -> bool:
        """Find and verify a good source route to this device.

        Generates ranked candidate routes from the neighbor graph (best first),
        verifies each end to end via the application's per-hop establishment,
        and keeps the first one that works -- recording its real measured LQI.
        Failed candidates are blacklisted (with a TTL) to avoid re-trying them.
        """
        LOGGER.debug("Scanning routes for %s", self.device.nwk)

        candidates = [
            route
            for route in self._build_candidate_routes(max_hops=self._max_hops())
            if not self._is_bad_route(route)
        ]

        if not candidates:
            LOGGER.debug("No candidate routes found for %s", self.device.nwk)
            return False

        for route in candidates:
            try:
                established = await self.device.application.establish_route(
                    self.device.nwk, route
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug(
                    "Error establishing route to %s via %s: %s",
                    self.device.nwk,
                    route,
                    exc,
                    exc_info=exc,
                )
                established = False

            if established:
                LOGGER.debug("Established route for %s: %s", self.device.nwk, route)
                self.last_successful_route = route
                self.last_route = route
                self.last_was_successful = True
                # Real measured LQI of the verification ping reply
                self.last_lqi = self.device.lqi if self.device.lqi is not None else 0
                return True

            self._mark_bad_route(route)

        LOGGER.debug("No working route found for %s", self.device.nwk)
        return False


    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        if not ping and self.last_successful_route is not None:
            LOGGER.debug("Returning last successful topology route for %s", self.device.nwk)
            self.last_route = self.last_successful_route
            return self.last_successful_route

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
        if self.device.relays is None:
            return None

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

        # Background ping bookkeeping (see ControllerApplication._ping_loop)
        self.ping_attempts: int = 0
        self.ping_in_flight: bool = False

        # Steady-phase ping backoff for an unreachable device. ``next_ping_at``
        # is the earliest time the device is eligible for another steady ping;
        # ``ping_backoff`` is the current interval, grown on each failed probe.
        self.ping_backoff: timedelta = PING_BACKOFF_BASE
        self.next_ping_at: datetime = datetime.now(UTC)

    def ping_due(self, now: datetime | None = None) -> bool:
        """Whether the device is eligible for another steady-phase ping."""
        return (now or datetime.now(UTC)) >= self.next_ping_at

    def schedule_next_ping(self, now: datetime | None = None) -> None:
        """Record that a steady ping was just dispatched and grow the backoff.

        The next ping is pushed out by the current backoff, then the backoff is
        doubled (capped at ``PING_BACKOFF_MAX``) so a device that keeps failing
        is probed progressively less often.
        """
        now = now or datetime.now(UTC)
        self.next_ping_at = now + self.ping_backoff
        self.ping_backoff = min(self.ping_backoff * 2, PING_BACKOFF_MAX)

    def notify_seen(self) -> None:
        """We heard from the device: reset the ping backoff to the base cadence.

        This pulls the next steady ping *earlier* for a device that had backed
        off (so a reconnected device recovers promptly), but never later. It
        deliberately does NOT reset ``ping_attempts`` and does NOT schedule the
        next ping at "now": a healthy device replies to every ping, so doing
        either turns this into a tight re-converge/re-ping loop that starves the
        rest of the network.
        """
        if self.ping_backoff > PING_BACKOFF_BASE:
            LOGGER.debug(
                "Resetting ping backoff for %s (heard from device)", self.device.nwk
            )
        self.ping_backoff = PING_BACKOFF_BASE
        soonest = datetime.now(UTC) + PING_BACKOFF_BASE
        if soonest < self.next_ping_at:
            self.next_ping_at = soonest

    def believed_reachable(self) -> bool:
        """Whether we've heard from the device recently enough to justify route
        discovery. Used to avoid spending expensive discovery on a device that
        appears to be offline (see ``DISCOVERY_OFFLINE_AFTER``)."""
        last_seen = self.device.last_seen
        if last_seen is None:
            return False
        return (
            datetime.now(UTC).timestamp() - last_seen
            < DISCOVERY_OFFLINE_AFTER.total_seconds()
        )

    def is_converged(self) -> bool:
        """Whether the device has concluded its initial convergence.

        Until then we leave routing to the coordinator (automatic) and do not
        penalize delivery failures. A reconnect resets ``ping_attempts`` (via
        :meth:`notify_seen`), so a returning device drops back to automatic
        until it has re-converged.
        """
        return self.ping_attempts >= PING_CONVERGENCE_ATTEMPTS

    def notify_delivery_failure(self, tsn: int) -> None:
        """A send to the device failed to be delivered (no-ack / send error).

        Ignored during convergence. Once converged, the route used for ``tsn``
        accrues a failure; after ``DELIVERY_FAILURE_THRESHOLD`` consecutive
        failures (without a success) the route is demoted so ``build_route``
        stops choosing it.
        """
        if not self.is_converged():
            return

        route = self.tsn_route.get(tsn)
        if route is None:
            return

        route.consecutive_delivery_failures += 1
        if route.consecutive_delivery_failures >= DELIVERY_FAILURE_THRESHOLD:
            LOGGER.debug(
                "Demoting %s route for %s after %s delivery failures",
                route.name,
                self.device.nwk,
                route.consecutive_delivery_failures,
            )
            route.notify_route_error(tsn)

    def _build_route_ping(self, tsn: int, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        if attempt == 1 or self.last_ping_route is None:
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
                LOGGER.debug("Using topology route as ping route for %s", self.device.nwk)
                self.last_ping_route = self.topology_route

            elif self.last_ping_route == self.topology_route and self.topology_route.is_usable():
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
        return self.last_ping_route.build_route(tsn, True, attempt, max_attempts)

    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        LOGGER.debug("build_route called with parameters: tsn=%s, ping=%s, attempt=%s, max_attempts=%s for %s", tsn, ping, attempt, max_attempts, self.device.nwk)
        # if we are pinging the device, we can try routes. on requests we want the highest success rate
        if ping:
            return self._build_route_ping(tsn, attempt, max_attempts)

        # Sleepy end devices (and not-yet-interviewed devices) are always
        # reached via their parent router using the coordinator's own routing --
        # we never ping them, so they never converge and source routing does not
        # apply. Mains-powered rx-on-when-idle devices are excluded here (they
        # have a route maintained like routers); see Device.should_maintain_route.
        if not self.device.should_maintain_route:
            LOGGER.debug("Using automatic route for %s (sleepy end device / not interviewed)", self.device.nwk)
            self.tsn_route[tsn] = self.automatic_route
            return self.automatic_route.build_route(tsn, ping, attempt, max_attempts)

        # During the initial convergence phase, leave routing to the coordinator
        # (automatic / source_route=None). We don't yet have a reliable picture
        # of the network, so we neither pick source routes nor penalize failures.
        if not self.is_converged():
            LOGGER.debug("Using automatic route for %s (convergence phase)", self.device.nwk)
            self.tsn_route[tsn] = self.automatic_route
            return self.automatic_route.build_route(tsn, ping, attempt, max_attempts)

        route = None

        # Only trust the direct route once a direct probe has actually been
        # credited to it (i.e. a direct ping got a reply). The device's inbound
        # LQI is not a safe proxy: that traffic may have been relayed, and
        # Zigbee links are asymmetric, so good device->coordinator RX does not
        # imply a working coordinator->device direct send.
        if self.direct_route.average_lqi >= 80 and self.direct_route.last_was_successful:
            LOGGER.debug("Using direct route for %s because lqi is good", self.device.nwk)
            self.tsn_route[tsn] = self.direct_route
            return []
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

        # remember route we took for tsn
        self.tsn_route[tsn] = route

        ret = route.build_route(tsn, ping, attempt, max_attempts)
        LOGGER.debug("Using route %s for %s", ret, self.device.nwk)
        return ret

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