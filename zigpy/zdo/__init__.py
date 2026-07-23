from __future__ import annotations

from collections.abc import Coroutine
import functools
import logging

from zigpy.const import APS_REPLY_TIMEOUT
import zigpy.profiles
import zigpy.types as t
import zigpy.util

from . import types

LOGGER = logging.getLogger(__name__)

ZDO_ENDPOINT = 0


class ZDO(zigpy.util.CatchingTaskMixin, zigpy.util.ListenableMixin):
    """The ZDO endpoint of a device"""

    class LeaveOptions(t.bitmap8):
        """ZDO Mgmt_Leave_req Options."""

        NONE = 0
        RemoveChildren = 1 << 6
        Rejoin = 1 << 7

    def __init__(self, device):
        self._device = device
        self._listeners = {}

    def _serialize(self, command, *args, **kwargs):
        keys, schema = types.CLUSTERS[command]
        # TODO: expose this in a future PR
        assert not kwargs
        return t.serialize(args, schema)

    def deserialize(self, cluster_id, data):
        if cluster_id not in types.CLUSTERS:
            raise ValueError(f"Invalid ZDO cluster ID: 0x{cluster_id:04X}")

        _, param_types = types.CLUSTERS[cluster_id]
        hdr, data = types.ZDOHeader.deserialize(cluster_id, data)
        args, data = t.deserialize(data, param_types)

        if data:
            # TODO: Seems sane to check, but what should we do?
            self.warning("Data remains after deserializing ZDO frame: %r", data)

        return hdr, args

    async def request(
        self,
        command,
        *args,
        timeout=APS_REPLY_TIMEOUT,
        expect_reply: bool = True,
        use_ieee: bool = False,
        ask_for_ack: bool | None = None,
        priority: int | None = None,
        retries: int | None = None,
        retry_delay: float | None = None,
        route=None,
        **kwargs,
    ):
        data = self._serialize(command, *args, **kwargs)
        tsn = self.device.get_sequence()
        return await self._device.request(
            profile=0x0000,
            cluster=command,
            src_ep=ZDO_ENDPOINT,
            dst_ep=ZDO_ENDPOINT,
            sequence=tsn,
            data=t.uint8_t(tsn).serialize() + data,
            timeout=timeout,
            expect_reply=expect_reply,
            use_ieee=use_ieee,
            ask_for_ack=ask_for_ack,
            priority=priority,
            retries=retries,
            retry_delay=retry_delay,
            route=route,
        )

    async def reply(
        self,
        command,
        *args,
        tsn: int | t.uint8_t | None = None,
        timeout=APS_REPLY_TIMEOUT,
        expect_reply: bool = False,
        use_ieee: bool = False,
        ask_for_ack: bool | None = None,
        priority: int | None = None,
        retries: int | None = None,
        retry_delay: float | None = None,
        **kwargs,
    ):
        data = self._serialize(command, *args, **kwargs)
        if tsn is None:
            tsn = self.device.get_sequence()
        return await self._device.reply(
            profile=0x0000,
            cluster=command,
            src_ep=ZDO_ENDPOINT,
            dst_ep=ZDO_ENDPOINT,
            sequence=tsn,
            data=t.uint8_t(tsn).serialize() + data,
            timeout=timeout,
            expect_reply=expect_reply,
            use_ieee=use_ieee,
            ask_for_ack=ask_for_ack,
            priority=priority,
            retries=retries,
            retry_delay=retry_delay,
        )

    def handle_message(
        self,
        profile: int,
        cluster: int,
        hdr: types.ZDOHeader,
        args: list,
    ) -> None:
        self.debug("ZDO request %s: %s", hdr.command_id, args)

        handler = getattr(self, f"handle_{hdr.command_id.name.lower()}", None)
        if handler is not None:
            handler(hdr, *args)
        else:
            self.debug("No handler for ZDO request:%s(%s)", hdr.command_id, args)

        self.listener_event(
            f"zdo_{hdr.command_id.name.lower()}",
            self._device,
            None,  # was `dst_addressing` but this was never set
            hdr,
            args,
        )

    def handle_nwk_addr_req(
        self,
        hdr: types.ZDOHeader,
        ieee: t.EUI64,
        request_type: int,
        start_index: int | None = None,
    ):
        """Handle ZDO NWK Address request."""

        app = self._device.application
        if ieee == app.state.node_info.ieee:
            self.create_catching_task(
                self.NWK_addr_rsp(
                    0,
                    app.state.node_info.ieee,
                    app.state.node_info.nwk,
                    0,
                    0,
                    [],
                    tsn=hdr.tsn,
                    priority=t.PacketPriority.LOW,
                )
            )

    def handle_ieee_addr_req(
        self,
        hdr: types.ZDOHeader,
        nwk: t.NWK,
        request_type: int,
        start_index: int | None = None,
    ):
        """Handle ZDO IEEE Address request."""

        app = self._device.application
        if nwk in (
            t.BroadcastAddress.ALL_DEVICES,
            t.BroadcastAddress.RX_ON_WHEN_IDLE,
            t.BroadcastAddress.ALL_ROUTERS_AND_COORDINATOR,
            app.state.node_info.nwk,
        ):
            self.create_catching_task(
                self.IEEE_addr_rsp(
                    0,
                    app.state.node_info.ieee,
                    app.state.node_info.nwk,
                    0,
                    0,
                    [],
                    tsn=hdr.tsn,
                    priority=t.PacketPriority.LOW,
                )
            )

    def handle_device_annce(
        self,
        hdr: types.ZDOHeader,
        nwk: t.NWK,
        ieee: t.EUI64,
        capability: int,
    ):
        """Handle ZDO device announcement request."""
        self.listener_event("device_announce", self._device)

    def handle_mgmt_permit_joining_req(
        self,
        hdr: types.ZDOHeader,
        permit_duration: int,
        tc_significance: int,
    ):
        """Handle ZDO permit joining request."""

        self.listener_event("permit_duration", permit_duration)

    def _coordinator_device(self, nwk_addr_of_interest: t.NWK):
        """The coordinator's own device, if the request targets it.

        Some devices -- notably Aqara -- periodically interrogate the
        coordinator with descriptor requests as an "is my network still
        alive?" check and may drop off the network when it never answers,
        so these requests must be handled.
        """
        app = self._device.application
        if nwk_addr_of_interest != app.state.node_info.nwk:
            return None
        try:
            return app._device
        except (KeyError, AttributeError):
            # Coordinator device not registered (yet)
            return None

    def handle_active_ep_req(
        self,
        hdr: types.ZDOHeader,
        nwk_addr_of_interest: t.NWK,
    ):
        """Handle ZDO Active endpoint request for the coordinator."""
        coordinator = self._coordinator_device(nwk_addr_of_interest)
        if coordinator is None:
            return

        self.create_catching_task(
            self.Active_EP_rsp(
                types.Status.SUCCESS,
                nwk_addr_of_interest,
                [t.uint8_t(ep) for ep in coordinator.endpoints if ep != 0],
                tsn=hdr.tsn,
                priority=t.PacketPriority.CRITICAL,
            )
        )

    def handle_node_desc_req(
        self,
        hdr: types.ZDOHeader,
        nwk_addr_of_interest: t.NWK,
    ):
        """Handle ZDO Node descriptor request for the coordinator."""
        coordinator = self._coordinator_device(nwk_addr_of_interest)
        if coordinator is None or coordinator.node_desc is None:
            return

        self.create_catching_task(
            self.Node_Desc_rsp(
                types.Status.SUCCESS,
                nwk_addr_of_interest,
                coordinator.node_desc,
                tsn=hdr.tsn,
                priority=t.PacketPriority.CRITICAL,
            )
        )

    def handle_simple_desc_req(
        self,
        hdr: types.ZDOHeader,
        nwk_addr_of_interest: t.NWK,
        endpoint: t.uint8_t,
    ):
        """Handle ZDO Simple descriptor request for the coordinator."""
        coordinator = self._coordinator_device(nwk_addr_of_interest)
        if (
            coordinator is None
            or endpoint == 0
            or endpoint not in coordinator.endpoints
        ):
            return

        ep = coordinator.endpoints[endpoint]
        if ep.profile_id is None:
            return

        self.create_catching_task(
            self.Simple_Desc_rsp(
                types.Status.SUCCESS,
                nwk_addr_of_interest,
                types.SizePrefixedSimpleDescriptor(
                    endpoint=t.uint8_t(endpoint),
                    profile=t.uint16_t(ep.profile_id),
                    device_type=t.uint16_t(ep.device_type or 0),
                    device_version=t.uint8_t(0),
                    input_clusters=[t.uint16_t(c) for c in ep.in_clusters],
                    output_clusters=[t.uint16_t(c) for c in ep.out_clusters],
                ),
                tsn=hdr.tsn,
                priority=t.PacketPriority.CRITICAL,
            )
        )

    def handle_match_desc_req(
        self,
        hdr: types.ZDOHeader,
        addr: t.NWK,
        profile: int,
        in_clusters: list,
        out_cluster: list,
    ):
        """Handle ZDO Match_desc_req request."""

        local_addr = self._device.application.state.node_info.nwk
        if profile != zigpy.profiles.zha.PROFILE_ID:
            self.create_catching_task(
                self.Match_Desc_rsp(
                    0,
                    local_addr,
                    [],
                    tsn=hdr.tsn,
                    priority=t.PacketPriority.CRITICAL,
                )
            )
            return

        self.create_catching_task(
            self.Match_Desc_rsp(
                0,
                local_addr,
                [t.uint8_t(1)],
                tsn=hdr.tsn,
                priority=t.PacketPriority.CRITICAL,
            )
        )

    async def bind(self, cluster, **kwargs):
        return await self.Bind_req(
            self._device.ieee,
            cluster.endpoint.endpoint_id,
            cluster.cluster_id,
            self.device.application.get_dst_address(cluster),
            **kwargs,
        )

    async def unbind(self, cluster):
        return await self.Unbind_req(
            self._device.ieee,
            cluster.endpoint.endpoint_id,
            cluster.cluster_id,
            self.device.application.get_dst_address(cluster),
        )

    def leave(
        self, remove_children: bool = True, rejoin: bool = False, route=None
    ) -> Coroutine:
        opts = self.LeaveOptions.NONE
        if remove_children:
            opts |= self.LeaveOptions.RemoveChildren
        if rejoin:
            opts |= self.LeaveOptions.Rejoin

        return self.Mgmt_Leave_req(self._device.ieee, opts, route=route)

    def permit(self, duration=60, tc_significance=0):
        return self.Mgmt_Permit_Joining_req(duration, tc_significance)

    def log(self, lvl, msg, *args, **kwargs):
        msg = "[0x%04x:zdo] " + msg
        args = (self._device.nwk, *args)
        return LOGGER.log(lvl, msg, *args, **kwargs)

    @property
    def device(self):
        return self._device

    def __getattr__(self, name):
        try:
            command = types.ZDOCmd[name]
        except KeyError as exc:
            raise AttributeError(f"No such '{name}' ZDO command") from exc

        if command & 0x8000:
            return functools.partial(self.reply, command)
        return functools.partial(self.request, command)


def broadcast(
    app,
    command,
    grpid,
    radius,
    *args,
    broadcast_address=t.BroadcastAddress.RX_ON_WHEN_IDLE,
    **kwargs,
):
    params, param_types = types.CLUSTERS[command]

    named_args = dict(zip(params, args, strict=False))
    named_args.update(kwargs)
    assert set(named_args.keys()) == set(params)

    sequence = app.get_sequence()
    data = bytes([sequence]) + t.serialize(named_args.values(), param_types)

    return zigpy.device.broadcast(
        app,
        0,
        command,
        0,
        0,
        grpid,
        radius,
        sequence,
        data,
        broadcast_address=broadcast_address,
    )
