"""Tests for holding requests to sleepy end devices until they are awake."""

import asyncio
from unittest.mock import patch

import pytest

from tests.conftest import make_node_desc
import zigpy.config as conf
from zigpy.profiles import zha
import zigpy.types as t
from zigpy.zcl.clusters.general import OnOff
from zigpy.zdo import types as zdo_t


def make_packet(dev, data: bytes) -> t.ZigbeePacket:
    return t.ZigbeePacket(
        src=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=dev.nwk),
        src_ep=3,
        dst=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=0x0000),
        dst_ep=1,
        tsn=0xAA,
        profile_id=zha.PROFILE_ID,
        cluster_id=OnOff.cluster_id,
        data=t.SerializableBytes(data),
        lqi=200,
    )


def reply_to(dev, tsn: int) -> t.ZigbeePacket:
    """Default response to a toggle command with the given ZCL sequence."""
    return make_packet(dev, bytes([0x18, tsn, 0x0B, 0x02, 0x00]))


def unrelated(dev) -> t.ZigbeePacket:
    """Something the device says of its own accord (an attribute report)."""
    return make_packet(dev, b"\x18\x77\x0a\x00\x00\x10\x01")


@pytest.fixture
def sleepy(app_mock):
    ieee = t.EUI64(map(t.uint8_t, [0, 1, 2, 3, 4, 5, 6, 7]))
    dev = app_mock.add_device(nwk=t.NWK(0x1234), ieee=ieee)
    dev.node_desc = make_node_desc(logical_type=zdo_t.LogicalType.EndDevice)
    ep = dev.add_endpoint(3)
    ep.profile_id = zha.PROFILE_ID
    ep.add_input_cluster(OnOff.cluster_id)

    def answer(packet):
        tsn = packet.data.serialize()[1]
        asyncio.get_running_loop().call_soon(dev.packet_received, reply_to(dev, tsn))

    dev.application.send_packet.side_effect = answer
    return dev


def toggle(dev, tsn: int, **kwargs):
    return dev.request(
        zha.PROFILE_ID,
        OnOff.cluster_id,
        1,
        3,
        tsn,
        bytes([0x01, tsn, 0x02]),
        **kwargs,
    )


async def test_sleepy_classification(sleepy, app_mock):
    assert sleepy.is_sleepy

    sleepy.node_desc = make_node_desc()  # a router
    assert not sleepy.is_sleepy

    sleepy.node_desc = None  # not interviewed: awake while joining
    assert not sleepy.is_sleepy


async def test_waiting_is_off_by_default(sleepy):
    """Without a wake timeout a request goes out immediately, as before."""
    await toggle(sleepy, 0x21)
    assert len(sleepy.application.send_packet.mock_calls) == 1


async def test_request_held_until_device_is_heard(sleepy):
    task = asyncio.create_task(toggle(sleepy, 0x22, wake_timeout=5))

    await asyncio.sleep(0.02)
    assert not sleepy.application.send_packet.mock_calls
    assert len(sleepy._wake_waiters) == 1

    # The device speaks: it is awake now, the request goes out
    sleepy.packet_received(unrelated(sleepy))
    async with asyncio.timeout(1):
        assert await task is not None

    assert len(sleepy.application.send_packet.mock_calls) == 1
    assert not sleepy._wake_waiters


async def test_request_fails_unsent_if_device_stays_silent(sleepy):
    with pytest.raises(asyncio.TimeoutError, match="not heard in time"):
        await toggle(sleepy, 0x23, wake_timeout=0.02)

    assert not sleepy.application.send_packet.mock_calls
    assert not sleepy._wake_waiters


async def test_recently_heard_device_is_not_waited_for(sleepy):
    sleepy.packet_received(unrelated(sleepy))

    async with asyncio.timeout(1):
        await toggle(sleepy, 0x24, wake_timeout=5)

    assert len(sleepy.application.send_packet.mock_calls) == 1


async def test_config_sets_the_default_wait(sleepy):
    with patch.dict(
        sleepy.application.config, {conf.CONF_NWK_SLEEPY_WAKE_TIMEOUT: 0.02}
    ):
        with pytest.raises(asyncio.TimeoutError):
            await toggle(sleepy, 0x25)

        # The caller's choice wins over the setting
        await toggle(sleepy, 0x26, wake_timeout=0)

    assert len(sleepy.application.send_packet.mock_calls) == 1


async def test_always_on_devices_and_replies_are_never_held(sleepy):
    # A reply to the device (nothing expected back) goes out at once
    await toggle(sleepy, 0x27, wake_timeout=5, expect_reply=False)
    assert len(sleepy.application.send_packet.mock_calls) == 1

    # So does a request to a device that keeps its receiver on
    sleepy.node_desc = make_node_desc()
    async with asyncio.timeout(1):
        await toggle(sleepy, 0x28, wake_timeout=5)
    assert len(sleepy.application.send_packet.mock_calls) == 2


async def test_cluster_request_passes_the_wait_on(sleepy):
    cluster = sleepy.endpoints[3].on_off
    task = asyncio.create_task(cluster.toggle(wake_timeout=5))

    await asyncio.sleep(0.02)
    assert not sleepy.application.send_packet.mock_calls

    sleepy.packet_received(unrelated(sleepy))
    async with asyncio.timeout(1):
        await task

    assert len(sleepy.application.send_packet.mock_calls) == 1


async def test_waiting_requests_go_out_together(sleepy):
    """Everything waiting is sent when the device is heard, without waiting
    for each other's replies: the device may collect only during one short
    window. The per-device concurrency limit does not apply to them."""
    sleepy.application.send_packet.side_effect = None
    tasks = [
        asyncio.create_task(toggle(sleepy, 0x30 + i, wake_timeout=5)) for i in range(8)
    ]
    await asyncio.sleep(0.02)
    assert len(sleepy._wake_waiters) == 8
    assert len(sleepy.application.send_packet.mock_calls) == 0

    sleepy.packet_received(unrelated(sleepy))
    await asyncio.sleep(0.02)
    assert len(sleepy.application.send_packet.mock_calls) == 8

    for i in range(8):
        sleepy.packet_received(reply_to(sleepy, 0x30 + i))
    async with asyncio.timeout(1):
        await asyncio.gather(*tasks)


async def test_held_request_uses_short_reply_timeout_and_retries_on_next_wake(sleepy):
    """A held request does not wait the long end-device timeout for a reply;
    its retry goes out when the device is heard again."""
    sent = []
    sleepy.application.send_packet.side_effect = sent.append

    with patch("zigpy.device.SLEEPY_HELD_REPLY_TIMEOUT", 0.05), patch(
        "zigpy.device.SLEEPY_AWAKE_WINDOW", 0.01
    ):
        task = asyncio.create_task(toggle(sleepy, 0x41, wake_timeout=5, retries=1))
        await asyncio.sleep(0.01)

        sleepy.packet_received(unrelated(sleepy))
        await asyncio.sleep(0.01)
        assert len(sent) == 1

        # No reply within the short timeout; the retry waits for the device
        await asyncio.sleep(0.3)
        assert len(sent) == 1

        sleepy.packet_received(unrelated(sleepy))
        await asyncio.sleep(0.01)
        assert len(sent) == 2

        sleepy.packet_received(reply_to(sleepy, 0x41))
        async with asyncio.timeout(1):
            assert await task is not None


def expiring_parent(dev, expire_times: int):
    """A parent that holds each frame briefly and reports it expired
    ``expire_times`` times before the device finally collects one."""
    import zigpy.exceptions

    calls = []

    async def send(packet):
        calls.append(packet)
        if len(calls) <= expire_times:
            await asyncio.sleep(0.03)
            raise zigpy.exceptions.DestinationUnreachable("expired")
        tsn = packet.data.serialize()[1]
        asyncio.get_running_loop().call_soon(dev.packet_received, reply_to(dev, tsn))

    dev.application.send_packet.side_effect = send
    return calls


async def test_request_is_reparked_when_the_parent_reports_expiry(sleepy):
    """An expired frame is parked again at once and does not use up an
    attempt, so the request is there whenever the device polls."""
    calls = expiring_parent(sleepy, expire_times=5)

    with patch("zigpy.device.MIN_PARKED_SECONDS", 0.01):
        # Device heard once, so the first copy goes out
        sleepy.packet_received(unrelated(sleepy))

        async with asyncio.timeout(2):
            assert await toggle(sleepy, 0x51, wake_timeout=5, retries=0) is not None

    assert len(calls) == 6
    assert sleepy._parent_reports_expiry

    # From now on requests are parked straight away, without waiting to hear
    # the device first
    sleepy._last_heard = None
    calls = expiring_parent(sleepy, expire_times=0)
    async with asyncio.timeout(1):
        await toggle(sleepy, 0x52, wake_timeout=5)
    assert len(calls) == 1


async def test_parked_request_gives_up_when_the_wait_runs_out(sleepy):
    calls = expiring_parent(sleepy, expire_times=1000)
    sleepy.packet_received(unrelated(sleepy))

    with patch("zigpy.device.MIN_PARKED_SECONDS", 0.01):
        with pytest.raises(asyncio.TimeoutError, match="did not collect"):
            await toggle(sleepy, 0x53, wake_timeout=0.2, retries=0)

    assert 3 <= len(calls) <= 8


async def test_instant_unreachable_is_an_ordinary_failed_attempt(sleepy):
    """A destination failure that comes back at once is not a parked frame;
    it must not be retried in a tight loop."""
    import zigpy.exceptions

    sleepy.application.send_packet.side_effect = (
        zigpy.exceptions.DestinationUnreachable("gone")
    )
    sleepy.packet_received(unrelated(sleepy))

    with pytest.raises(zigpy.exceptions.DestinationUnreachable):
        await toggle(sleepy, 0x54, wake_timeout=5, retries=1)

    assert len(sleepy.application.send_packet.mock_calls) == 2
    assert not sleepy._parent_reports_expiry


def router_parent(dev, sent, expire_times: int):
    """A router parent: the send succeeds, and for the first ``expire_times``
    copies it later reports that the child never collected the frame."""

    def send(packet):
        sent.append(packet)
        if len(sent) <= expire_times:
            asyncio.get_running_loop().call_later(
                0.05,
                dev.application.handle_source_route_failure,
                dev.nwk,
                "SOURCE_ROUTE_FAILURE",
            )
        else:
            tsn = packet.data.serialize()[1]
            asyncio.get_running_loop().call_soon(dev.packet_received, reply_to(dev, tsn))

    dev.application.send_packet.side_effect = send


async def test_router_parent_expiry_resends_when_device_is_heard(sleepy):
    """Routers that report an uncollected frame as a source route failure:
    arriving seconds after the send, it means the child was asleep. It is no
    route error and no failed attempt; the request goes out again the next
    time the device is heard."""
    sent = []
    router_parent(sleepy, sent, expire_times=2)

    with patch("zigpy.device.ROUTE_ERROR_PARKED_SECONDS", 0.03), patch(
        "zigpy.device.SLEEPY_AWAKE_WINDOW", 0.01
    ):
        task = asyncio.create_task(toggle(sleepy, 0x61, wake_timeout=5, retries=0))

        for expected in (1, 2):
            await asyncio.sleep(0.02)
            sleepy.packet_received(unrelated(sleepy))
            await asyncio.sleep(0.2)
            # Sent once per wake-up, not re-sent on its own after the expiry
            assert len(sent) == expected

        sleepy.packet_received(unrelated(sleepy))
        async with asyncio.timeout(1):
            assert await task is not None

    assert len(sent) == 3
    assert not sleepy._parent_reports_expiry
    assert sleepy._routing.automatic_route.consecutive_delivery_failures == 0


async def test_keep_parked_resends_at_once_behind_a_router(sleepy):
    """The caller can ask for a copy to be kept at the parent continuously."""
    sent = []
    router_parent(sleepy, sent, expire_times=3)

    with patch("zigpy.device.ROUTE_ERROR_PARKED_SECONDS", 0.03):
        async with asyncio.timeout(2):
            # Not heard at all: parking does not wait for that
            assert (
                await toggle(
                    sleepy, 0x63, wake_timeout=5, retries=0, keep_parked=True
                )
                is not None
            )

    assert len(sent) == 4


async def test_quick_route_error_is_a_real_route_problem(sleepy):
    import zigpy.exceptions

    sent = []

    def fail_fast(packet):
        sent.append(packet)
        asyncio.get_running_loop().call_later(
            0.01,
            sleepy.application.handle_source_route_failure,
            sleepy.nwk,
            "SOURCE_ROUTE_FAILURE",
        )

    sleepy.application.send_packet.side_effect = fail_fast
    sleepy.packet_received(unrelated(sleepy))

    with pytest.raises(zigpy.exceptions.RouteError):
        await toggle(sleepy, 0x62, wake_timeout=5, retries=0)

    assert len(sent) == 1
