"""Tests for route quality attribution in zigpy.routing."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import make_app, make_node_desc
from zigpy.profiles import zha
import zigpy.routing
import zigpy.types as t
from zigpy.zcl.clusters.general import OnOff
from zigpy.zdo import types as zdo_t


@pytest.fixture
def dev(app_mock):
    ieee = t.EUI64(map(t.uint8_t, [0, 1, 2, 3, 4, 5, 6, 7]))
    dev = app_mock.add_device(nwk=t.NWK(0x1234), ieee=ieee)
    dev.node_desc = make_node_desc()
    ep = dev.add_endpoint(3)
    ep.profile_id = zha.PROFILE_ID
    ep.add_input_cluster(OnOff.cluster_id)
    return dev


async def test_matched_reply_credits_route(dev):
    """A reply matched to our request credits the route stored for its TSN,
    regardless of the reply's APS-level sequence number."""
    dev.relays = [t.NWK(0xABCD)]
    routing = dev._routing
    reported = routing.reported_route

    def fake_send(packet):
        assert packet.source_route == [t.NWK(0xABCD)]
        # Default_Response (server_to_client) to our toggle command, carrying
        # an APS-level tsn deliberately different from the request's ZCL tsn
        dev.packet_received(
            t.ZigbeePacket(
                src=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=dev.nwk),
                src_ep=3,
                dst=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=0x0000),
                dst_ep=1,
                tsn=0xAA,
                profile_id=zha.PROFILE_ID,
                cluster_id=OnOff.cluster_id,
                data=t.SerializableBytes(b"\x18\x14\x0b\x02\x00"),
                lqi=200,
            )
        )

    dev.application.send_packet.side_effect = fake_send

    rsp = await dev.request(
        zha.PROFILE_ID, OnOff.cluster_id, 1, 3, 0x14, b"\x01\x14\x02"
    )

    assert rsp is not None
    assert reported.packages_received == 1
    assert reported.last_was_successful is True
    assert reported.last_lqi == 200
    assert reported.consecutive_delivery_failures == 0
    # The entry is consumed so later TSN collisions cannot re-credit it
    assert 0x14 not in routing.tsn_route


async def test_unsolicited_packet_does_not_credit_route(dev):
    """A device-initiated packet whose APS and ZCL sequence numbers collide
    with an in-flight request TSN must not credit that request's route."""
    routing = dev._routing
    reported = routing.reported_route
    routing.tsn_route[0x42] = reported

    # Unsolicited attribute report; both its APS tsn and its ZCL tsn are the
    # device's own counters and happen to collide with our stored entry
    dev.packet_received(
        t.ZigbeePacket(
            src=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=dev.nwk),
            src_ep=3,
            dst=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=0x0000),
            dst_ep=1,
            tsn=0x42,
            profile_id=zha.PROFILE_ID,
            cluster_id=OnOff.cluster_id,
            data=t.SerializableBytes(b"\x18\x42\x0a\x00\x00\x10\x01"),
            lqi=200,
        )
    )

    assert reported.packages_received == 0
    assert reported.last_was_successful is False
    assert 0x42 in routing.tsn_route


def test_route_usability_not_gated_on_lqi(dev):
    """A route that delivered its last packet is usable even with a weak
    final hop; usability must not depend on the reply's LQI."""
    routing = dev._routing

    routing.automatic_route.last_was_successful = True
    routing.automatic_route.last_lqi = 50

    assert routing.automatic_route.is_usable()
    assert routing.any_route_usable()


def test_any_route_usable_includes_automatic(dev):
    """A device reachable only via NCP (automatic) routing is healthy and
    must not be treated as needing continued probing."""
    routing = dev._routing

    assert not routing.any_route_usable()

    routing.automatic_route.last_was_successful = True
    assert routing.any_route_usable()

    routing.automatic_route.last_was_successful = False
    routing.direct_route.last_was_successful = True
    assert routing.any_route_usable()


def test_usable_source_route(dev):
    """Automatic never counts as a source route; best working one is picked."""
    routing = dev._routing

    routing.automatic_route.last_was_successful = True
    assert routing.usable_source_route() is None

    routing.reported_route.last_was_successful = True
    assert routing.usable_source_route() is routing.reported_route

    routing.direct_route.last_was_successful = True
    assert routing.usable_source_route() is routing.direct_route


@pytest.fixture
def rejoin_dev():
    app = make_app({"routing_auto_rejoin": True})
    # Startup quiet window has passed
    app._auto_rejoin_quiet_until = datetime.now(UTC)
    ieee = t.EUI64(map(t.uint8_t, [8, 9, 10, 11, 12, 13, 14, 15]))
    dev = app.add_device(nwk=t.NWK(0x05B3), ieee=ieee)
    dev.node_desc = make_node_desc(logical_type=zdo_t.LogicalType.EndDevice)
    # Reachable via a working direct source route
    dev._routing.direct_route.last_was_successful = True
    return dev


async def test_auto_rejoin_fires_after_strikes(rejoin_dev):
    """Source route works + automatic broken, confirmed twice -> rejoin."""
    dev = rejoin_dev
    app = dev.application
    routing = dev._routing

    # The automatic probe fails; the live source-route verification passes
    probe = AsyncMock(side_effect=lambda route: route is not routing.automatic_route)

    with (
        patch.object(dev, "ping_using_route_works", probe),
        patch.object(dev.zdo, "leave", AsyncMock()) as leave,
    ):
        await app._maybe_heal_parent_link(dev)
        assert routing.auto_rejoin_strikes == 1
        assert leave.call_count == 0

        # Second check within the interval is skipped entirely
        await app._maybe_heal_parent_link(dev)
        assert routing.auto_rejoin_strikes == 1

        # With a strike pending, the shorter confirmation interval applies:
        # aging the last check by just that much makes the device due again,
        # and the second strike fires the rejoin
        routing.last_auto_rejoin_check = (
            datetime.now(UTC) - zigpy.routing.AUTO_REJOIN_CONFIRM_INTERVAL
        )
        await app._maybe_heal_parent_link(dev)

        assert leave.call_count == 1
        assert leave.mock_calls[0].kwargs == {
            "remove_children": False,
            "rejoin": True,
            "route": routing.direct_route,
        }
        assert routing.auto_rejoin_strikes == 0
        assert routing.last_auto_rejoin is not None

        # And then the device is left alone for AUTO_REJOIN_MIN_INTERVAL
        routing.last_auto_rejoin_check = (
            datetime.now(UTC) - zigpy.routing.AUTO_REJOIN_CHECK_INTERVAL
        )
        await app._maybe_heal_parent_link(dev)
        assert leave.call_count == 1


async def test_auto_rejoin_startup_quiet_window(rejoin_dev):
    """No strikes are counted during the cold-start quiet window."""
    dev = rejoin_dev
    app = dev.application
    app._auto_rejoin_quiet_until = (
        datetime.now(UTC) + zigpy.routing.AUTO_REJOIN_STARTUP_QUIET
    )

    with patch.object(dev, "ping_using_route_works", AsyncMock()) as probe:
        await app._maybe_heal_parent_link(dev)

    assert probe.call_count == 0
    assert dev._routing.auto_rejoin_strikes == 0


def test_auto_rejoin_confirm_interval(rejoin_dev):
    """Strike-free devices use the baseline cadence; a pending strike
    shortens the re-check to the confirmation interval."""
    routing = rejoin_dev._routing

    routing.last_auto_rejoin_check = datetime.now(UTC) - (
        zigpy.routing.AUTO_REJOIN_CONFIRM_INTERVAL + timedelta(seconds=1)
    )
    assert not routing.auto_rejoin_check_due()

    routing.auto_rejoin_strikes = 1
    assert routing.auto_rejoin_check_due()


async def test_auto_rejoin_verifies_source_route_before_firing(rejoin_dev):
    """A stale 'usable' source route must not receive the leave: verification
    failure demotes it, resets strikes, and spares the 24h cooldown."""
    dev = rejoin_dev
    app = dev.application
    routing = dev._routing

    with (
        # Everything fails live: automatic probe AND source-route verification
        patch.object(dev, "ping_using_route_works", AsyncMock(return_value=False)),
        patch.object(dev.zdo, "leave", AsyncMock()) as leave,
    ):
        await app._maybe_heal_parent_link(dev)
        routing.last_auto_rejoin_check = (
            datetime.now(UTC) - zigpy.routing.AUTO_REJOIN_CONFIRM_INTERVAL
        )
        await app._maybe_heal_parent_link(dev)

    assert leave.call_count == 0
    assert routing.last_auto_rejoin is None  # cooldown not consumed
    assert routing.auto_rejoin_strikes == 0
    # The stale route was demoted, so the ping loop resumes probing the device
    assert not routing.direct_route.is_usable()
    assert not routing.any_route_usable()


async def test_auto_rejoin_healthy_device_resets_strikes(rejoin_dev):
    """A working automatic route clears the signature -- no rejoin."""
    dev = rejoin_dev
    dev._routing.auto_rejoin_strikes = 1

    with (
        patch.object(dev, "ping_using_route_works", AsyncMock(return_value=True)),
        patch.object(dev.zdo, "leave", AsyncMock()) as leave,
    ):
        await dev.application._maybe_heal_parent_link(dev)

    assert dev._routing.auto_rejoin_strikes == 0
    assert leave.call_count == 0


async def test_auto_rejoin_excludes_routers_and_respects_config(rejoin_dev):
    dev = rejoin_dev

    # Routers are never asked to leave
    dev.node_desc = make_node_desc(logical_type=zdo_t.LogicalType.Router)
    with patch.object(dev, "ping_using_route_works", AsyncMock()) as probe:
        await dev.application._maybe_heal_parent_link(dev)
    assert probe.call_count == 0

    # Disabled (default) config: no probing at all, even for end devices
    app = make_app({})
    ieee = t.EUI64(map(t.uint8_t, [1, 2, 3, 4, 5, 6, 7, 8]))
    dev2 = app.add_device(nwk=t.NWK(0x1111), ieee=ieee)
    dev2.node_desc = make_node_desc(logical_type=zdo_t.LogicalType.EndDevice)
    dev2._routing.direct_route.last_was_successful = True
    with patch.object(dev2, "ping_using_route_works", AsyncMock()) as probe:
        await app._maybe_heal_parent_link(dev2)
    assert probe.call_count == 0


def test_ping_ladder_cycles_through_all_routes(dev):
    """The probe ladder must reach reported and automatic, skipping data-less
    steps (topology without a candidate, reported without relays)."""
    routing = dev._routing

    def probe():
        routing.build_route(tsn=1, ping=True, attempt=1, max_attempts=3)
        return routing.last_ping_route

    # No topology candidate, no relays: alternates direct <-> automatic
    assert probe() is routing.direct_route
    assert probe() is routing.automatic_route
    assert probe() is routing.direct_route

    # With relays known, the reported route joins the cycle
    dev.relays = [t.NWK(0xABCD)]
    assert probe() is routing.reported_route
    assert probe() is routing.automatic_route
    assert probe() is routing.direct_route

    # With a topology candidate, all four are probed in order
    routing.topology_route.last_successful_route = [t.NWK(0x1234)]
    assert probe() is routing.topology_route
    assert probe() is routing.reported_route
    assert probe() is routing.automatic_route
    assert probe() is routing.direct_route

    # Retries within one ping keep the chosen route
    routing.build_route(tsn=2, ping=True, attempt=1, max_attempts=3)
    chosen = routing.last_ping_route
    routing.build_route(tsn=2, ping=True, attempt=2, max_attempts=3)
    assert routing.last_ping_route is chosen


async def test_ping_discovery_forced_only_with_recent_contact(dev):
    """FORCE_ROUTE_DISCOVERY costs a network-wide broadcast per send, so it is
    reserved for the stale-route case: the last ping attempt to a device we
    have recently heard from. Silent devices rely on the always-on
    ENABLE_ROUTE_DISCOVERY option instead (on-demand, no extra broadcast)."""

    async def last_attempt_discovery():
        with patch.object(
            dev.application, "request", AsyncMock()
        ) as app_request:
            await dev.request(
                zha.PROFILE_ID,
                OnOff.cluster_id,
                1,
                3,
                0x21,
                b"\x01\x21\x02",
                expect_reply=False,
                ping=True,
                retries=0,
            )
        return app_request.mock_calls[0].kwargs["force_route_discovery"]

    dev.node_desc = make_node_desc()

    assert dev.last_seen is None
    assert await last_attempt_discovery() is False  # silent: never force

    dev.last_seen = datetime.now(UTC)
    assert await last_attempt_discovery() is True  # recently heard: stale route


def test_timeout_demotes_reported_route(dev):
    """A reported route that keeps timing out silently must stop being chosen
    for real requests -- stale source-route failures arrive as asynchronous
    network statuses, not send errors, so timeouts count toward demotion."""
    dev.node_desc = make_node_desc()
    dev.relays = [t.NWK(0xFE9E), t.NWK(0x7F1A)]
    routing = dev._routing

    assert (
        routing.build_route(tsn=1, ping=False, attempt=1, max_attempts=3)
        == [t.NWK(0x7F1A), t.NWK(0xFE9E)]
    )

    for _ in range(zigpy.routing.DELIVERY_FAILURE_THRESHOLD):
        routing.reported_route.notify_timeout(1)

    routing.build_route(tsn=2, ping=False, attempt=1, max_attempts=3)
    assert routing.tsn_route[2] is not routing.reported_route

    # A fresh route record re-enables it
    dev.relays = [t.NWK(0x1234)]
    routing.build_route(tsn=3, ping=False, attempt=1, max_attempts=3)
    assert routing.tsn_route[3] is routing.reported_route


def test_source_route_failure_demotes_last_used_route(dev):
    """A SOURCE_ROUTE_FAILURE network status demotes the relayed route that
    was in use; direct/automatic sends cannot be implicated."""
    dev.node_desc = make_node_desc()
    app = dev.application
    routing = dev._routing

    # Unknown device: no-op
    app.handle_source_route_failure(t.NWK(0xDEAD))

    # Last send was via the reported route
    dev.relays = [t.NWK(0xFE9E), t.NWK(0x7F1A)]
    routing.build_route(tsn=1, ping=False, attempt=1, max_attempts=3)
    assert routing.last_used_route is routing.reported_route

    app.handle_source_route_failure(dev.nwk)
    assert (
        routing.reported_route.consecutive_delivery_failures
        >= zigpy.routing.DELIVERY_FAILURE_THRESHOLD
    )
    assert not routing.reported_route.last_was_successful

    # Last send was direct: a source-route failure cannot implicate it
    routing.last_used_route = routing.direct_route
    routing.direct_route.last_was_successful = True
    app.handle_source_route_failure(dev.nwk)
    assert routing.direct_route.last_was_successful
    assert routing.direct_route.consecutive_delivery_failures == 0


def test_proven_route_ends_convergence(dev):
    """A healthy adjacent router: one successful direct ping, then probing
    stops. That proven route must end convergence so real traffic uses it,
    instead of staying on 'automatic (convergence phase)' forever."""
    dev.node_desc = make_node_desc()
    routing = dev._routing

    # Ping loop probed once, direct route worked, probing stopped
    routing.ping_attempts = 1
    routing.direct_route.last_was_successful = True
    routing.direct_route.average_lqi = 200
    routing.direct_route.last_lqi = 200

    assert routing.is_converged()
    assert routing.build_route(tsn=1, ping=False, attempt=1, max_attempts=3) == []
    assert routing.tsn_route[1] is routing.direct_route


def test_route_audit_scheduling(dev):
    """Healthy devices are audited on a jittered interval, one ladder rung
    per audit, so route knowledge stays fresh without continuous probing."""
    routing = dev._routing

    # Initial audit is scheduled in the future (jittered around the interval)
    assert not routing.audit_due()
    assert routing.next_audit_at > datetime.now(UTC)

    routing.next_audit_at = datetime.now(UTC)
    assert routing.audit_due()

    routing.schedule_next_audit()
    assert not routing.audit_due()
    # Jitter stays within 0.75x..1.25x of the interval
    delta = routing.next_audit_at - datetime.now(UTC)
    assert (
        zigpy.routing.ROUTE_AUDIT_INTERVAL * 0.7
        < delta
        < zigpy.routing.ROUTE_AUDIT_INTERVAL * 1.3
    )


def test_topology_scan_rate_limiting(dev):
    routing = dev._routing

    # Never scanned: due immediately, but not while one is in flight
    assert routing.topology_scan_due()
    routing.topology_scan_in_flight = True
    assert not routing.topology_scan_due()
    routing.topology_scan_in_flight = False

    # Freshly scanned: not due again until the minimum interval has passed
    routing.last_topology_scan = datetime.now(UTC)
    assert not routing.topology_scan_due()

    routing.last_topology_scan = (
        datetime.now(UTC)
        - zigpy.routing.TOPOLOGY_SCAN_MIN_INTERVAL
        - timedelta(seconds=1)
    )
    assert routing.topology_scan_due()
