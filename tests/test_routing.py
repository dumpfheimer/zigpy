"""Tests for route quality attribution in zigpy.routing."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest

from tests.conftest import make_app, make_node_desc
import zigpy.exceptions
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


async def test_explicit_relay_list_reply_credits_no_stale_route(dev):
    """A send over a raw relay list (topology candidate verification) has no
    route object. Its reply must not credit whatever route last used the same
    (wrapped) TSN."""
    routing = dev._routing
    reported = routing.reported_route

    # Leftover from an earlier send whose request never got a reply
    routing.tsn_route[0x14] = reported

    def fake_send(packet):
        assert packet.source_route == [t.NWK(0xABCD)]
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

    await dev.request(
        zha.PROFILE_ID,
        OnOff.cluster_id,
        1,
        3,
        0x14,
        b"\x01\x14\x02",
        route=[t.NWK(0xABCD)],
    )

    assert reported.packages_received == 0
    assert reported.last_was_successful is False
    assert 0x14 not in routing.tsn_route


async def test_explicit_relay_list_timeout_blames_no_stale_route(dev):
    """Likewise, a timed-out relay-list send must not demote or blacklist the
    route left behind under the same TSN."""
    routing = dev._routing
    topology = routing.topology_route
    topology.last_route = [t.NWK(0x5555)]
    routing.tsn_route[0x14] = topology

    with patch("zigpy.device.APS_REPLY_TIMEOUT_EXTENDED", 0.01), patch(
        "zigpy.device.APS_REPLY_TIMEOUT", 0.01
    ):
        with pytest.raises(TimeoutError):
            await dev.request(
                zha.PROFILE_ID,
                OnOff.cluster_id,
                1,
                3,
                0x14,
                b"\x01\x14\x02",
                route=[t.NWK(0xABCD)],
                retries=0,
            )

    assert topology.consecutive_delivery_failures == 0
    assert topology.bad_routes == {}


def test_failure_notifications_consume_tsn_entry(dev):
    """Timeouts, route errors and delivery failures end a send; its TSN entry
    is removed so a later send reusing the TSN cannot be misattributed."""
    routing = dev._routing
    dev.relays = [t.NWK(0xFE9E)]

    routing.build_route(tsn=1, ping=False, attempt=1, max_attempts=3)
    routing.notify_timeout(1)
    assert 1 not in routing.tsn_route

    routing.build_route(tsn=2, ping=False, attempt=1, max_attempts=3)
    routing.notify_route_error(2)
    assert 2 not in routing.tsn_route

    routing.build_route(tsn=3, ping=False, attempt=1, max_attempts=3)
    routing.notify_no_fault(3)
    assert 3 not in routing.tsn_route


def test_direct_preferred_over_reported(dev):
    """A proven direct link beats the relayed path the device reported: an
    extra hop is a cost when the device is in range."""
    routing = dev._routing
    dev.relays = [t.NWK(0xABCD)]

    # Direct not proven yet: the reported path is used
    assert routing.build_route(tsn=1, ping=False, attempt=1, max_attempts=3) == [
        t.NWK(0xABCD)
    ]

    routing.direct_route.last_was_successful = True
    assert routing.build_route(tsn=2, ping=False, attempt=1, max_attempts=3) == []
    assert routing.tsn_route[2] is routing.direct_route


def test_retry_takes_a_different_path(dev):
    """Each attempt of one request uses the next proven path, and the final
    attempt is left to the coordinator's own routing."""
    routing = dev._routing
    dev.relays = [t.NWK(0xABCD)]
    routing.direct_route.last_was_successful = True

    assert routing.build_route(tsn=7, ping=False, attempt=1, max_attempts=3) == []
    # A silent timeout does not demote the direct route, the retry still moves on
    assert routing.build_route(tsn=7, ping=False, attempt=2, max_attempts=3) == [
        t.NWK(0xABCD)
    ]
    assert routing.build_route(tsn=7, ping=False, attempt=3, max_attempts=3) is None
    assert routing.tsn_route[7] is routing.automatic_route

    # A new request with the same TSN starts from the best path again
    assert routing.build_route(tsn=7, ping=False, attempt=1, max_attempts=3) == []

    # A single-attempt request still gets the best proven path
    assert routing.build_route(tsn=8, ping=False, attempt=1, max_attempts=1) == []


def test_no_proven_path_uses_automatic(dev):
    routing = dev._routing

    assert routing.build_route(tsn=1, ping=False, attempt=1, max_attempts=3) is None
    assert routing.tsn_route[1] is routing.automatic_route


def test_radio_without_explicit_routes_is_always_automatic(dev):
    """If the radio cannot apply a route we name, every send is automatic and
    the probe ladder does not pretend to test anything else."""
    routing = dev._routing
    dev.relays = [t.NWK(0xABCD)]
    routing.direct_route.last_was_successful = True

    with patch.object(
        type(dev.application),
        "explicit_routes_supported",
        new_callable=PropertyMock,
        return_value=False,
    ):
        assert (
            routing.build_route(tsn=1, ping=False, attempt=1, max_attempts=3) is None
        )
        assert routing.tsn_route[1] is routing.automatic_route

        routing.build_route(tsn=2, ping=True, attempt=1, max_attempts=3)
        assert routing.last_ping_route is routing.automatic_route
        routing.build_route(tsn=3, ping=True, attempt=1, max_attempts=3)
        assert routing.last_ping_route is routing.automatic_route
        assert not routing.direct_probe_due()


async def test_acknowledged_direct_send_proves_direct(dev):
    """The radio's send confirmation for a direct send is the device's own MAC
    acknowledgement: it proves the link, and a missing reply does not undo it."""
    routing = dev._routing
    direct = routing.direct_route

    with patch("zigpy.device.APS_REPLY_TIMEOUT_EXTENDED", 0.01), patch(
        "zigpy.device.APS_REPLY_TIMEOUT", 0.01
    ):
        with pytest.raises(TimeoutError):
            await dev.request(
                zha.PROFILE_ID,
                OnOff.cluster_id,
                1,
                3,
                0x31,
                b"\x01\x31\x02",
                route=direct,
                retries=0,
            )

    assert dev.application.send_packet.mock_calls[0].args[0].source_route == []
    assert direct.last_was_successful is True
    assert direct.consecutive_delivery_failures == 0
    assert 0x31 not in routing.tsn_route


@pytest.mark.parametrize(
    ("exc", "demoted"),
    [
        (zigpy.exceptions.DeliveryError("no mac ack"), True),
        (zigpy.exceptions.RouteError("source route failure"), True),
        (zigpy.exceptions.SendError("busy"), False),
    ],
)
async def test_send_failure_classification(dev, exc, demoted):
    """An unacknowledged or mis-routed send demotes its path at once; a busy
    radio says nothing about the path."""
    routing = dev._routing
    direct = routing.direct_route
    direct.last_was_successful = True
    dev.application.send_packet.side_effect = exc

    with patch.dict(
        dev.application.config, {"scheduling_timeout": 0}
    ), pytest.raises(type(exc)):
        await dev.request(
            zha.PROFILE_ID,
            OnOff.cluster_id,
            1,
            3,
            0x32,
            b"\x01\x32\x02",
            retries=0,
        )

    assert direct.last_was_successful is (not demoted)
    assert 0x32 not in routing.tsn_route
    if demoted:
        # The direct link is re-probed instead of being forgotten
        assert routing.direct_probe_due()


async def test_source_route_failure_blames_the_request_route(dev):
    """A reported source route failure fails the waiting request and demotes
    the route that request used -- not whichever route was used last."""
    routing = dev._routing
    dev.relays = [t.NWK(0xABCD)]
    reported = routing.reported_route
    direct = routing.direct_route

    def fake_send(packet):
        # A later, unrelated send makes the direct route the most recent one
        routing.last_used_route = direct
        direct.last_was_successful = True
        asyncio.get_running_loop().call_later(
            0.01, dev.application.handle_source_route_failure, dev.nwk, "status"
        )

    dev.application.send_packet.side_effect = fake_send

    with pytest.raises(zigpy.exceptions.RouteError):
        await dev.request(
            zha.PROFILE_ID,
            OnOff.cluster_id,
            1,
            3,
            0x33,
            b"\x01\x33\x02",
            timeout=5,
            retries=0,
        )

    assert not reported.last_was_successful
    assert (
        reported.consecutive_delivery_failures
        >= zigpy.routing.DELIVERY_FAILURE_THRESHOLD
    )
    assert direct.last_was_successful
    assert direct.consecutive_delivery_failures == 0


async def test_source_route_failure_spares_direct_requests(dev):
    """A request sent directly was acknowledged by the device itself; a relay's
    failure report cannot be about it."""
    routing = dev._routing
    routing.direct_route.last_was_successful = True

    def fake_send(packet):
        assert packet.source_route == []
        asyncio.get_running_loop().call_later(
            0.01, dev.application.handle_source_route_failure, dev.nwk, "status"
        )

    dev.application.send_packet.side_effect = fake_send

    with pytest.raises(TimeoutError):
        await dev.request(
            zha.PROFILE_ID,
            OnOff.cluster_id,
            1,
            3,
            0x34,
            b"\x01\x34\x02",
            timeout=0.05,
            retries=0,
        )

    assert routing.direct_route.last_was_successful


def test_single_timeout_does_not_demote(dev):
    routing = dev._routing
    automatic = routing.automatic_route
    automatic.last_was_successful = True

    automatic.notify_timeout(1)
    assert automatic.last_was_successful

    automatic.notify_timeout(2)
    assert not automatic.last_was_successful


def test_direct_reprobe_schedule(dev):
    """An unused direct link is re-probed on a growing backoff, immediately
    when the device reports that it reaches us without relays."""
    routing = dev._routing
    now = datetime.now(UTC)

    assert routing.direct_probe_due(now)
    routing.schedule_next_direct_probe(now)
    assert not routing.direct_probe_due(now)
    assert routing.direct_probe_due(now + zigpy.routing.DIRECT_REPROBE_BASE)

    for _ in range(10):
        routing.schedule_next_direct_probe(now)
    assert routing.direct_probe_backoff == zigpy.routing.DIRECT_REPROBE_MAX
    assert not routing.direct_probe_due(now)

    # An empty route record: the device says it is one hop away
    dev.relays = []
    assert routing.direct_probe_backoff == zigpy.routing.DIRECT_REPROBE_BASE
    assert routing.direct_probe_due()

    # Once direct works there is nothing to re-probe
    routing.direct_route.last_was_successful = True
    assert not routing.direct_probe_due()


async def test_forced_discovery_not_used_on_source_routed_ping(dev):
    """Route discovery cannot help a frame that carries its own route."""
    dev.last_seen = datetime.now(UTC)
    routing = dev._routing

    for route, expected in (
        (routing.direct_route, False),
        (routing.automatic_route, True),
    ):
        dev.application.send_packet.reset_mock()
        await dev.request(
            0x0000,
            zdo_t.ZDOCmd.IEEE_addr_req,
            0,
            0,
            0x40,
            b"\x40\x34\x12\x00\x00",
            expect_reply=False,
            ping=True,
            route=route,
            retries=0,
        )
        packet = dev.application.send_packet.mock_calls[0].args[0]
        assert (
            t.TransmitOptions.FORCE_ROUTE_DISCOVERY in packet.tx_options
        ) is expected


def add_router(app, nwk: int, *, direct: bool = True, lqi: int = 100):
    ieee = t.EUI64(map(t.uint8_t, [nwk & 0xFF, nwk >> 8, 9, 9, 9, 9, 9, 9]))
    router = app.add_device(nwk=t.NWK(nwk), ieee=ieee)
    router.node_desc = make_node_desc()
    router.lqi = lqi
    router._routing.direct_route.last_was_successful = direct
    return router


def test_relay_candidates_come_from_proven_direct_routers(dev):
    """Any router with a proven direct link can be a relay, whether or not a
    neighbour table mentions it; tables and the route record only order them."""
    app = dev.application
    app.topology.neighbors = {}
    add_router(app, 0x1111, lqi=90)
    add_router(app, 0x2222, lqi=200)
    add_router(app, 0x3333, direct=False, lqi=250)
    hinted = add_router(app, 0x4444, lqi=50)
    reported = add_router(app, 0x5555, lqi=10)

    relay = dev._routing.relay_route
    # No hints: every direct router, best own link first; never the others
    assert relay.candidates() == [0x2222, 0x1111, 0x4444, 0x5555]

    # A neighbour table entry moves a router up, the route record to the top
    app.topology.neighbors = {
        hinted.ieee: [zdo_t.Neighbor(nwk=dev.nwk, lqi=120)],
    }
    dev.relays = [reported.nwk, t.NWK(0x2222)]
    assert relay.candidates() == [0x5555, 0x4444, 0x2222, 0x1111]

    # A relay that failed for this device is left out for a while
    relay.bad_relays[t.NWK(0x5555)] = datetime.now(UTC)
    assert relay.candidates() == [0x4444, 0x2222, 0x1111]
    relay.bad_relays[t.NWK(0x5555)] -= zigpy.routing.BAD_RELAY_TTL * 2
    assert relay.candidates()[0] == 0x5555


async def test_relay_scan_keeps_the_first_relay_that_works(dev):
    app = dev.application
    app.topology.neighbors = {}
    add_router(app, 0x1111, lqi=200)
    add_router(app, 0x2222, lqi=100)
    routing = dev._routing
    relay = routing.relay_route

    tried = []

    async def ping(route):
        tried.append(route)
        return route == [0x2222]

    with patch.object(dev, "ping_using_route_works", side_effect=ping):
        assert relay.scan_due()
        assert await relay.scan() is True

    assert tried == [[0x1111], [0x2222]]
    assert relay.relay == 0x2222
    assert relay.is_usable()
    assert not relay.scan_due()
    assert 0x1111 in relay.bad_relays

    # Order for requests: direct, then the tested relay, then the reported path
    routing.direct_route.last_was_successful = True
    dev.relays = [t.NWK(0x7777), t.NWK(0x8888)]
    assert routing.request_candidates() == [
        routing.direct_route,
        routing.relay_route,
        routing.reported_route,
    ]
    assert routing.relay_route.build_route(1, False, 1, 3) == [0x2222]


async def test_relay_scan_without_candidates_retries_soon(dev):
    dev.application.topology.neighbors = {}
    relay = dev._routing.relay_route

    assert await relay.scan() is False
    assert relay.relay is None
    wait = relay.next_scan_at - datetime.now(UTC)
    assert timedelta(0) < wait <= zigpy.routing.RELAY_SCAN_RETRY


def test_failed_relay_is_dropped_for_this_device_only(dev):
    app = dev.application
    app.topology.neighbors = {}
    router = add_router(app, 0x1111)
    routing = dev._routing
    relay = routing.relay_route
    relay.relay = router.nwk
    relay.last_was_successful = True

    routing.tsn_route[5] = relay
    routing.notify_route_error(5)

    assert relay.relay is None
    assert not relay.is_usable()
    assert router.nwk in relay.bad_relays
    assert relay.scan_due()
    # The router's own standing is untouched
    assert router._routing.direct_route.last_was_successful

    # A relay whose own direct link is gone is not used either
    relay.relay = router.nwk
    relay.last_was_successful = True
    assert relay.is_usable()
    router._routing.direct_route.last_was_successful = False
    assert not relay.is_usable()


def test_silent_device_is_reported(dev, caplog):
    """A device that acknowledges direct pings but never answers is stuck."""
    routing = dev._routing

    for tsn in (1, 2):
        routing.tsn_route[tsn] = routing.direct_route
        routing.last_ping_tsn = tsn
        routing.notify_timeout(tsn)

    assert routing.silent_pings == 2
    assert "may be stuck" in caplog.text

    # A ping reply clears it
    routing.tsn_route[3] = routing.direct_route
    routing.last_ping_tsn = 3
    routing.notify_reply(3, t.ZigbeePacket(lqi=100, data=t.SerializableBytes(b"")))
    assert routing.silent_pings == 0


async def test_failed_probe_is_not_repeated(dev):
    """A ping whose path was reported broken is not sent again over it."""
    dev.application.send_packet.side_effect = zigpy.exceptions.DeliveryError("no ack")

    with pytest.raises(zigpy.exceptions.DeliveryError):
        await dev.ping(route=dev._routing.direct_route)

    assert len(dev.application.send_packet.mock_calls) == 1
