from __future__ import annotations

import logging
import random
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

# Minimum time between targeted topology scans of a single device. A scan is
# several ZDO round-trips, and Topology.scan() preempts whatever scan is
# already running -- including the global periodic one -- so targeted scans
# must stay rare or they starve the neighbor graph they exist to feed.
TOPOLOGY_SCAN_MIN_INTERVAL = timedelta(minutes=15)

# Background route audit cadence for *healthy* devices. The point of the ping
# loop is to explore route quality where failures have no consequences, so
# that real sends pick from fresh knowledge -- but continuous probing is what
# melted the network originally. One ladder ping per device per interval
# advances one rung (direct -> topology -> reported -> automatic), giving full
# route-type coverage roughly every four intervals at negligible airtime.
# Scheduling is jittered so audits de-synchronize instead of bursting.
ROUTE_AUDIT_INTERVAL = timedelta(minutes=30)

# Automatic parent-link healing (see
# ControllerApplication._maybe_heal_parent_link). The check probes the
# automatic route of a source-route-reachable end device at most this often:
AUTO_REJOIN_CHECK_INTERVAL = timedelta(minutes=15)
# Once a device has a strike, its confirmation probe comes sooner -- two
# independent observations are what matters, not a long wait between them:
AUTO_REJOIN_CONFIRM_INTERVAL = timedelta(minutes=5)
# The signature must be confirmed this many checks in a row before acting --
# a single failed probe can be congestion, not a lost parent:
AUTO_REJOIN_STRIKES = 2
# No strikes are counted this soon after startup: the NCP's routing state is
# cold and the network is congested with startup traffic, so automatic-route
# probes fail spuriously and a leave sent to a healthy device is the one
# failure mode this feature must never have:
AUTO_REJOIN_STARTUP_QUIET = timedelta(minutes=5)
# And after sending a leave-with-rejoin, leave the device alone for this long:
AUTO_REJOIN_MIN_INTERVAL = timedelta(hours=24)

# Direct (zero-hop) delivery is the preferred path. While a maintained device
# is being reached some other way, its direct link is re-probed on this
# backoff so it returns to direct soon after the link recovers. A direct probe
# is a single frame whose outcome the radio reports at MAC level.
DIRECT_REPROBE_BASE = timedelta(minutes=1)
DIRECT_REPROBE_MAX = timedelta(minutes=30)


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
        # route that has stopped delivering (see notify_timeout / notify_route_error).
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
        """The path itself failed: the radio or the mesh reported that the
        frame could not be delivered over it (no MAC acknowledgement from the
        first hop, or a route error from a relay). Demoted immediately."""
        LOGGER.debug("Received route error for %s (%s) tsn %s", self.device.nwk, self.name, tsn)
        self.last_was_successful = False
        self.packages_lost += 1
        self.consecutive_delivery_failures = max(
            self.consecutive_delivery_failures + 1, DELIVERY_FAILURE_THRESHOLD
        )

    def notify_timeout(self, tsn):
        """The frame was sent but no reply arrived in time.

        This is weaker evidence than a reported failure -- the device may
        simply not have answered, or its reply got lost on the way back, which
        the outbound route has no part in. It therefore only demotes the route
        after ``DELIVERY_FAILURE_THRESHOLD`` timeouts in a row. It still has to
        count: a relay that drops frames silently is only ever seen this way.
        """
        LOGGER.debug("Received timeout for %s (%s) tsn %s", self.device.nwk, self.name, tsn)
        self.packages_lost += 1
        self.consecutive_delivery_failures += 1
        if self.consecutive_delivery_failures >= DELIVERY_FAILURE_THRESHOLD:
            self.last_was_successful = False

    def notify_sent(self, tsn):
        """The radio confirmed the frame left for its first hop."""

    def is_usable(self):
        """Whether this route delivered the last packet we sent over it.

        Deliberately not gated on LQI: the LQI of a reply only measures its
        final hop into the coordinator, and a working route with a weak last
        hop is still a working route. Gating usability on LQI put every
        healthy-but-sub-threshold device on a permanent fast ping cadence.
        LQI remains a *preference* signal when choosing among usable routes
        (see DeviceRouting.build_route).
        """
        return self.last_was_successful


class DirectRoute(RouteBase):
    """This route sends the packet directly to the device without any relays"""
    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        return []

    def notify_sent(self, tsn):
        # The first hop of a direct send is the device itself, so the radio's
        # send confirmation means the device acknowledged the frame at MAC
        # level: the link is proven without waiting for a reply.
        self.last_was_successful = True
        self.consecutive_delivery_failures = 0

    def notify_timeout(self, tsn):
        # The device acknowledged the frame (otherwise the send itself would
        # have failed), so a missing reply says nothing about this link.
        LOGGER.debug(
            "No reply from %s for tsn %s after an acknowledged direct send",
            self.device.nwk,
            tsn,
        )

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
        return self.last_was_successful and self.has_good_route()

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
        super().notify_timeout(tsn)
        LOGGER.debug(
            "Timeout on relayed route for %s (%s) tsn %s (route %s)",
            self.device.nwk,
            self.name,
            tsn,
            self.last_route,
        )
        if not self.last_was_successful:
            self._mark_bad_route(self.last_route)

    def notify_route_error(self, tsn):
        super().notify_route_error(tsn)
        self._mark_bad_route(self.last_successful_route)
        self.last_successful_route = None

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
        # The route most recently chosen for any transmission; used to
        # attribute asynchronous SOURCE_ROUTE_FAILURE network statuses, which
        # carry no TSN (see notify_source_route_failure)
        self.last_used_route: RouteBase | None = None
        self.packages_received: int = 0
        self.next_hop: zigpy.device.Device | None = None
        # Routes already used by the attempts of the request with a given TSN,
        # so that a retry takes a different path (see build_route)
        self.tried_routes: dict[int, list[RouteBase]] = {}

        # Re-probing of the direct link while it is not in use
        self.direct_probe_backoff: timedelta = DIRECT_REPROBE_BASE
        self.next_direct_probe_at: datetime = datetime.now(UTC)

        # Background ping bookkeeping (see ControllerApplication._ping_loop)
        self.ping_attempts: int = 0
        self.ping_in_flight: bool = False

        # Steady-phase ping backoff for an unreachable device. ``next_ping_at``
        # is the earliest time the device is eligible for another steady ping;
        # ``ping_backoff`` is the current interval, grown on each failed probe.
        self.ping_backoff: timedelta = PING_BACKOFF_BASE
        self.next_ping_at: datetime = datetime.now(UTC)

        # Targeted topology scan rate limiting (see
        # ControllerApplication._scan_device_topology)
        self.last_topology_scan: datetime | None = None
        self.topology_scan_in_flight: bool = False

        # Automatic parent-link healing state (see
        # ControllerApplication._maybe_heal_parent_link)
        self.last_auto_rejoin_check: datetime | None = None
        self.auto_rejoin_strikes: int = 0
        self.last_auto_rejoin: datetime | None = None

        # Background route audit for healthy devices: jittered so a fleet of
        # devices does not audit in lockstep bursts
        self.next_audit_at: datetime = datetime.now(UTC) + ROUTE_AUDIT_INTERVAL * (
            random.uniform(0.5, 1.5)
        )

    def audit_due(self, now: datetime | None = None) -> bool:
        """Whether a healthy device is due for a background route audit."""
        return (now or datetime.now(UTC)) >= self.next_audit_at

    def schedule_next_audit(self, now: datetime | None = None) -> None:
        self.next_audit_at = (now or datetime.now(UTC)) + ROUTE_AUDIT_INTERVAL * (
            random.uniform(0.75, 1.25)
        )

    @property
    def explicit_routes_supported(self) -> bool:
        """Whether the radio sends a frame over the route zigpy names."""
        return bool(
            getattr(self.device.application, "explicit_routes_supported", True)
        )

    def direct_probe_due(self, now: datetime | None = None) -> bool:
        """Whether the unused direct link should be probed again."""
        return (
            self.explicit_routes_supported
            and not self.direct_route.last_was_successful
            and (now or datetime.now(UTC)) >= self.next_direct_probe_at
        )

    def schedule_next_direct_probe(self, now: datetime | None = None) -> None:
        self.next_direct_probe_at = (now or datetime.now(UTC)) + self.direct_probe_backoff
        self.direct_probe_backoff = min(
            self.direct_probe_backoff * 2, DIRECT_REPROBE_MAX
        )

    def notify_sent(self, tsn: int) -> None:
        """The radio confirmed that the frame with ``tsn`` was sent."""
        route = self.tsn_route.get(tsn)
        if route is None:
            return
        route.notify_sent(tsn)
        if route is self.direct_route:
            self.direct_probe_backoff = DIRECT_REPROBE_BASE

    def notify_no_fault(self, tsn: int) -> None:
        """The send with ``tsn`` ended without saying anything about its route
        (the radio was busy, or the destination itself was unavailable)."""
        self.tsn_route.pop(tsn, None)

    def request_candidates(self) -> list[RouteBase]:
        """Proven paths for real requests, best first: direct, then the
        device's own reported path, then a verified topology route."""
        candidates: list[RouteBase] = []
        if self.direct_route.last_was_successful:
            candidates.append(self.direct_route)
        if (
            self.device.relays
            and self.reported_route.consecutive_delivery_failures
            < DELIVERY_FAILURE_THRESHOLD
        ):
            candidates.append(self.reported_route)
        if self.topology_route.has_good_route():
            candidates.append(self.topology_route)
        return candidates

    def usable_source_route(self) -> RouteBase | None:
        """The best currently-working source route, or None.

        Excludes the automatic route by definition: this answers "can we reach
        the device without the mesh's own routing", which is the half of the
        lost-parent signature that must hold for a rejoin request to be
        deliverable at all.
        """
        for route in (self.direct_route, self.topology_route, self.reported_route):
            if route.is_usable():
                return route
        return None

    def auto_rejoin_check_due(self, now: datetime | None = None) -> bool:
        """Whether the device is eligible for another parent-link check.

        A device with a pending strike is re-checked on the shorter
        confirmation interval so a real lost-parent case is acted on quickly;
        strike-free devices are probed on the relaxed baseline cadence.
        """
        now = now or datetime.now(UTC)
        if (
            self.last_auto_rejoin is not None
            and now - self.last_auto_rejoin < AUTO_REJOIN_MIN_INTERVAL
        ):
            return False
        interval = (
            AUTO_REJOIN_CONFIRM_INTERVAL
            if self.auto_rejoin_strikes > 0
            else AUTO_REJOIN_CHECK_INTERVAL
        )
        if (
            self.last_auto_rejoin_check is not None
            and now - self.last_auto_rejoin_check < interval
        ):
            return False
        return True

    def topology_scan_due(self, now: datetime | None = None) -> bool:
        """Whether this device is eligible for another targeted topology scan."""
        if self.topology_scan_in_flight:
            return False
        if self.last_topology_scan is None:
            return True
        return (
            (now or datetime.now(UTC)) - self.last_topology_scan
            >= TOPOLOGY_SCAN_MIN_INTERVAL
        )

    def any_route_usable(self) -> bool:
        """Whether any transmission path to the device currently works.

        Used by the background ping loop to decide whether a device still
        needs probing. This must include the automatic (NCP-routed) route:
        a device that is perfectly reachable via the coordinator's own
        routing is healthy, and keeping it on a fast ping cadence just
        because no *source* route qualifies burns airtime for nothing.
        """
        return (
            self.direct_route.is_usable()
            or self.topology_route.is_usable()
            or self.reported_route.is_usable()
            or self.automatic_route.is_usable()
        )

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
        penalize delivery failures. Convergence ends when we have learned
        something definitive, whichever comes first: enough probe attempts, or
        any route proving itself. The latter matters because the ping loop
        deliberately stops probing a device once a route works -- counting
        attempts alone would leave every healthy device "converging" forever,
        so its real traffic would never use the proven route.
        """
        return (
            self.ping_attempts >= PING_CONVERGENCE_ATTEMPTS
            or self.any_route_usable()
        )

    def _next_ping_route(self) -> RouteBase:
        """Advance the probe ladder: direct -> topology -> reported -> automatic.

        The ladder must genuinely cycle through every route type so a broken
        device gets each transmission path tried -- an earlier version only
        advanced past the topology step when it was already usable, which
        meant the reported and automatic routes were never probed at all for
        devices where topology routes cannot exist (end devices have no
        neighbor tables). Steps with no data behind them are skipped entirely:
        probing a topology route without a candidate (or a reported route
        without known relays) would transmit with source_route=None, i.e.
        actually test the automatic route while crediting the failure to the
        wrong route object.
        """
        order = [
            self.direct_route,
            self.topology_route,
            self.reported_route,
            self.automatic_route,
        ]
        start = (
            order.index(self.last_ping_route) + 1
            if self.last_ping_route in order
            else 0
        )
        for offset in range(len(order)):
            candidate = order[(start + offset) % len(order)]
            if (
                candidate is self.topology_route
                and self.topology_route.last_successful_route is None
            ):
                continue
            if candidate is self.reported_route and not self.device.relays:
                continue
            if (
                candidate is not self.automatic_route
                and not self.explicit_routes_supported
            ):
                continue
            return candidate
        return self.automatic_route

    def _build_route_ping(self, tsn: int, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        if attempt == 1 or self.last_ping_route is None:
            # only change once per ping
            self.last_ping_route = self._next_ping_route()
            LOGGER.debug(
                "Probing %s route for %s",
                self.last_ping_route.name,
                self.device.nwk,
            )
        else:
            LOGGER.debug("Using last ping route for %s", self.device.nwk)

        self.last_ping_tsn = tsn
        self.tsn_route[tsn] = self.last_ping_route
        self.last_used_route = self.last_ping_route
        return self.last_ping_route.build_route(tsn, True, attempt, max_attempts)

    def build_route(self, tsn: int, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        LOGGER.debug("build_route called with parameters: tsn=%s, ping=%s, attempt=%s, max_attempts=%s for %s", tsn, ping, attempt, max_attempts, self.device.nwk)
        # if we are pinging the device, we can try routes. on requests we want the highest success rate
        if ping:
            return self._build_route_ping(tsn, attempt, max_attempts)

        # Sleepy end devices (and not-yet-interviewed devices) are always
        # reached via their parent router using the coordinator's own routing.
        # The same goes for a radio that cannot apply a route we name: there
        # every send is "automatic", whatever we would ask for.
        if not self.device.should_maintain_route or not self.explicit_routes_supported:
            return self._use(tsn, self.automatic_route, ping, attempt, max_attempts)

        if attempt <= 1:
            self.tried_routes.pop(tsn, None)
        tried = self.tried_routes.setdefault(tsn, [])

        # Each attempt of a request takes the best proven path that this
        # request has not used yet: direct first (no extra hop), then the path
        # the device itself reported, then a verified topology route. A failed
        # path has usually been demoted already, but a silent timeout has not,
        # and the retry should still not repeat it.
        route = next(
            (c for c in self.request_candidates() if c not in tried), None
        )

        # Nothing proven is left, or this is the final retry: hand the frame
        # to the coordinator's own routing, the one path we do not choose.
        if route is None or (attempt > 1 and attempt >= max_attempts):
            route = self.automatic_route

        tried.append(route)
        return self._use(tsn, route, ping, attempt, max_attempts)

    def _use(
        self, tsn: int, route: RouteBase, ping: bool, attempt: int, max_attempts: int
    ) -> list[t.NWK] | None:
        """Record ``route`` as the one carrying ``tsn`` and build it."""
        self.tsn_route[tsn] = route
        self.last_used_route = route
        ret = route.build_route(tsn, ping, attempt, max_attempts)
        LOGGER.debug(
            "Using %s route %s for %s (attempt %s of %s)",
            route.name,
            ret,
            self.device.nwk,
            attempt,
            max_attempts,
        )
        return ret

    def notify_source_route_failure(self) -> None:
        """The mesh reported SOURCE_ROUTE_FAILURE toward this device.

        A relay on the source route we embedded could not forward the frame.
        These network statuses arrive asynchronously and carry no TSN, so the
        failure is attributed to the most recently used route. Only our own
        relayed routes are demoted: a direct send cannot cause a source-route
        failure, and failures of the automatic route's NCP-internal source
        routes are the NCP's to heal.
        """
        route = self.last_used_route
        if route is None or route in (self.direct_route, self.automatic_route):
            return

        LOGGER.debug(
            "Source route failure for %s: demoting %s route",
            self.device.nwk,
            route.name,
        )
        route.notify_route_error(None)

    def notify_relays_updated(self) -> None:
        """The device advertised a fresh path via a NWK route record.

        New relay data supersedes the demotion earned by the previous, stale
        relay list -- the reported route becomes eligible again and has to
        prove itself anew.
        """
        if self.reported_route.consecutive_delivery_failures > 0:
            LOGGER.debug(
                "Fresh relays for %s: re-enabling reported route", self.device.nwk
            )
        self.reported_route.consecutive_delivery_failures = 0

        # A route record without relays means the device reaches us directly,
        # which makes the direct link worth probing right away
        if not self.device.relays and not self.direct_route.last_was_successful:
            self.direct_probe_backoff = DIRECT_REPROBE_BASE
            self.next_direct_probe_at = datetime.now(UTC)

    def notify_route_error(self, tsn: int) -> None:
        """The path used for ``tsn`` failed (see RouteBase.notify_route_error)."""
        LOGGER.debug("Received route error for %s tsn %s", self.device.nwk, tsn)

        # Pop: the send is over, and a retry re-records its route. A leftover
        # entry would be credited or blamed by a later send that reuses this
        # (wrapped, 8-bit) TSN without recording a route of its own.
        packet_route = self.tsn_route.pop(tsn, None)
        if packet_route is not None:
            packet_route.notify_route_error(tsn)

    def notify_timeout(self, tsn: int) -> None:
        LOGGER.debug("Received timeout for %s tsn %s", self.device.nwk, tsn)

        # Pop, as in notify_route_error
        packet_route = self.tsn_route.pop(tsn, None)
        if packet_route is not None:
            packet_route.notify_timeout(tsn)

    def notify_reply(self, tsn: int, packet: t.ZigbeePacket) -> None:
        """A reply to a request we sent arrived; credit the route used for ``tsn``.

        ``tsn`` is the application-layer (ZCL/ZDO) sequence of *our* request,
        the same numbering space ``tsn_route`` is keyed with. This must only be
        called for packets positively matched to an in-flight request: the
        packet's own APS-level ``tsn`` is the device's independent counter and
        must never be used for this lookup. Ping replies are credited through
        here too -- ``tsn_route`` holds ``last_ping_route`` for ping TSNs.
        """
        if tsn == self.last_ping_tsn:
            LOGGER.debug("Received ping reply from %s tsn %s", self.device.nwk, tsn)

        # Pop the entry so a later, device-chosen TSN that happens to collide
        # with this value cannot spuriously re-credit the route.
        packet_route = self.tsn_route.pop(tsn, None)
        if packet_route is not None:
            packet_route.packet_received(packet)