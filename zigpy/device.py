from __future__ import annotations

import asyncio
from asyncio import timeout as asyncio_timeout
from collections.abc import Callable
import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import enum
import itertools
import logging
import math
import time
import typing
from typing import Any, Final, Self
import warnings

from cryptography.exceptions import InvalidTag

from zigpy import zdo
from zigpy.const import (
    APS_REPLY_TIMEOUT,
    APS_REPLY_TIMEOUT_EXTENDED,
    SIG_ENDPOINTS,
    SIG_EP_INPUT,
    SIG_EP_OUTPUT,
    SIG_EP_PROFILE,
    SIG_EP_TYPE,
    SIG_MANUFACTURER,
    SIG_MODEL,
    SIG_NODE_DESC,
)
import zigpy.config as conf
import zigpy.datastructures
import zigpy.endpoint
from zigpy.event import EventBase
import zigpy.exceptions
from zigpy.exceptions import (
    DeliveryError,
    DestinationUnreachable,
    RouteError,
    SendError,
)
import zigpy.listeners
from zigpy.ota.manager import update_firmware
from zigpy.profiles import zha, zll
import zigpy.routing
from zigpy.routing import DeviceRouting
import zigpy.types as t
import zigpy.util
from zigpy.util import CreateTaskMixin
from zigpy.zcl import Cluster, ClusterType, OtaQueryCacheClearedEvent, foundation
from zigpy.zcl.clusters.general import Ota, PollControl, QueryNextImageCommand
import zigpy.zdo.types as zdo_t
from zigpy.zgp.commands import (
    GPD_COMMAND_SCHEMAS,
    GPCommissioningPayload,
    GPGenericSwitchConfiguration,
)
from zigpy.zgp.crypto import build_gpdf_header, decrypt_payload
from zigpy.zgp.types import (
    ApplicationID,
    DeviceID,
    GPDCommandID,
    SecurityKeyType,
    SecurityLevel,
    SecurityStatus,
    SrcID,
)
from zigpy.zgp.util import derive_alias, synthetic_ieee

if typing.TYPE_CHECKING:
    from zigpy.application import ControllerApplication
    from zigpy.ota.providers import OtaImageWithMetadata


LOGGER = logging.getLogger(__name__)

PACKET_DEBOUNCE_WINDOW = 10
# Must be a multiple of the priority split below (halves): an odd value is
# rounded up by the limiter with a warning for every device
MAX_DEVICE_CONCURRENCY = 6
DEFAULT_FAST_POLL_TIMEOUT = 30

# A sleepy end device that has just been heard is assumed to keep polling its
# parent for this long, so a request can still be sent straight away
SLEEPY_AWAKE_WINDOW = 3.0
# Most requests that may wait for one sleepy device to wake up; further ones
# are sent immediately instead of queueing without bound
MAX_WAKE_WAITERS = 64
# Reply timeout for a request that was held until a sleepy device was heard.
# If the device is listening it collects the frame from its parent within the
# few seconds the parent keeps it, so waiting the full extended timeout only
# delays the retry, which itself waits for the device to be heard again.
SLEEPY_HELD_REPLY_TIMEOUT = 10
# A send that ends in "the parent's held copy expired" after at least this long
# was really parked at the parent; anything quicker is an ordinary failure
MIN_PARKED_SECONDS = 1.0
# Some routers report a frame their sleeping child never collected as a route
# error instead of an expiry. A broken route fails within about a second (MAC
# retries), a parent's hold lasts several, so a route error for a sleepy
# device that takes at least this long is read as "held and not collected"
ROUTE_ERROR_PARKED_SECONDS = 2.5
DEFAULT_REQUEST_RETRIES = 2
DEFAULT_REQUEST_RETRY_DELAY = 0.1

AFTER_OTA_ATTR_READ_DELAY = 10

# zgpDuplicateTimeout: unprotected GPDFs repeating the last MAC sequence number
# within this many seconds are duplicates
GP_DUPLICATE_TIMEOUT = timedelta(seconds=2)


@dataclass(frozen=True, slots=True)
class ResponseKey:
    """Key for request/response matching."""

    endpoint_id: int
    cluster_id: int
    direction: foundation.Direction | None
    tsn: int


@dataclass(kw_only=True, frozen=True)
class GreenPowerCommandReceived:
    """Event generated when a Green Power command is received."""

    event_type: Final[str] = "gp_command_received"

    device_ieee: str
    endpoint_id: int | None
    command_id: int
    command: t.Struct | None


class Status(enum.IntEnum):
    """The status of a Device. Maintained for backwards compatibility."""

    # No initialization done
    NEW = 0
    # ZDO endpoint discovery done
    ZDO_INIT = 1
    # Endpoints initialized
    ENDPOINTS_INIT = 2


class BaseDevice(
    zigpy.util.LocalLogMixin, zigpy.util.ListenableMixin, CreateTaskMixin, EventBase
):
    """Base class shared by all device types, keyed on `ieee` and `nwk` addresses."""

    def __init__(
        self, application: ControllerApplication, ieee: t.EUI64, nwk: t.NWK
    ) -> None:
        super().__init__()
        self._application: ControllerApplication = application
        self._ieee: t.EUI64 = ieee
        self.nwk: t.NWK = t.NWK(nwk)

        self.lqi: int | None = None
        self.rssi: int | None = None
        self._last_seen: datetime | None = None

        # Persist the original signature for the device, before quirks are applied
        self._original_signature: dict[str, Any] | None = None

        self._on_remove_callbacks: list[typing.Callable[[], None]] = []
        self._tasks: set[asyncio.Future[Any]] = set()

    def on_remove(self) -> None:
        """Call on remove callbacks."""
        for callback in self._on_remove_callbacks:
            callback()

        self._on_remove_callbacks.clear()

        for task in self._tasks:
            task.cancel()

        self._tasks.clear()

    def radio_details(self, lqi: int | None = None, rssi: int | None = None) -> None:
        """Set the link quality and signal strength of the last received packet."""
        warnings.warn(
            "`radio_details` is deprecated, assign to `lqi` and `rssi` instead",
            DeprecationWarning,
        )

        if lqi is not None:
            self.lqi = lqi

        if rssi is not None:
            self.rssi = rssi

    def update_last_seen(self) -> None:
        """Update the `last_seen` attribute to the current time and emit an event."""
        warnings.warn(
            "Calling `update_last_seen` directly is deprecated", DeprecationWarning
        )
        self.last_seen = datetime.now(UTC)

    @property
    def last_seen(self) -> float | None:
        return self._last_seen.timestamp() if self._last_seen is not None else None

    @last_seen.setter
    def last_seen(self, value: datetime | float | None):
        if isinstance(value, int | float):
            value = datetime.fromtimestamp(value, UTC)

        self._last_seen = value
        self.listener_event("device_last_seen_updated", self._last_seen)

    @property
    def application(self) -> ControllerApplication:
        return self._application

    @property
    def ieee(self) -> t.EUI64:
        return self._ieee

    @property
    def original_signature(self) -> dict[str, Any] | None:
        return self._original_signature

    @original_signature.setter
    def original_signature(self, value: dict[str, Any] | None) -> None:
        if self._original_signature is not None:
            return

        self._original_signature = value

    def get_signature(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def is_initialized(self) -> bool:
        raise NotImplementedError

    @property
    def manufacturer_id(self) -> int | None:
        raise NotImplementedError

    async def reinterview(self) -> None:
        raise NotImplementedError


class GreenPowerDevice(BaseDevice):
    """A Green Power Device (GPD)."""

    def __init__(
        self,
        application: ControllerApplication,
        *,
        application_id: ApplicationID,
        src_id: SrcID | None = None,
        ieee: t.EUI64 | None = None,
    ) -> None:
        if application_id is ApplicationID.SrcID:
            if src_id is None:
                raise ValueError("`src_id` is required with SrcID addressing")

            ieee = synthetic_ieee(src_id)
            gpd_id = src_id
        else:
            if ieee is None:
                raise ValueError("`ieee` is required with IEEE addressing")

            gpd_id = ieee

        # The alias is the NWK address the GPD occupies on the air: proxies send on
        # its behalf using it as the NWK source address (spec A.3.6.3.3). Distinct
        # GPD IDs can derive the same alias; the spec tolerates the collision.
        super().__init__(application, ieee, derive_alias(gpd_id))

        self.application_id = application_id
        self._src_id = src_id

        # Commissioning-derived signature (the quirk match input). All optional:
        # cheap GPDs commonly advertise only `device_id`.
        self.device_id: DeviceID | None = None  # GPD device type
        self.gpd_manufacturer_id: t.uint16_t | None = None
        self.gpd_model_id: t.uint16_t | None = None
        self.commands: list[GPDCommandID] = []
        self.server_cluster_ids: list[t.uint16_t] = []
        self.client_cluster_ids: list[t.uint16_t] = []

        # The only application description a `GenericSwitch` sends
        self.switch_configuration: GPGenericSwitchConfiguration | None = None

        # Whether the GPD uses an incremental (rather than random) MAC sequence number
        self.mac_seq_num_capability: bool = False

        # Whether the GPD can receive after transmitting, and whether it is stationary
        self.rx_on_capability: bool = False
        self.fixed_location: bool = False

        # Security material for decoding subsequent data frames.
        self.security_key: t.KeyData | None = None
        self.security_key_type: SecurityKeyType | None = None
        self.security_level: SecurityLevel | None = None
        self.frame_counter: t.uint32_t | None = None

    @classmethod
    def from_commissioning(
        cls,
        application: ControllerApplication,
        packet: t.ZigbeeGpPacket,
        commissioning: GPCommissioningPayload,
        *,
        security_level: SecurityLevel,
        security_key_type: SecurityKeyType,
        security_key: t.KeyData | None,
    ) -> Self:
        """Create a GPD from its Commissioning GPDF (0xE0)."""
        assert packet.src_id is not None

        device = cls(
            application, application_id=ApplicationID.SrcID, src_id=packet.src_id
        )
        device.device_id = commissioning.device_id
        device.gpd_manufacturer_id = commissioning.manufacturer_id
        device.gpd_model_id = commissioning.model_id
        device.commands = [
            GPDCommandID(cmd) for cmd in commissioning.gpd_commands or []
        ]
        device.server_cluster_ids = list(commissioning.server_clusters or [])
        device.client_cluster_ids = list(commissioning.client_clusters or [])

        if commissioning.switch_information is not None:
            device.switch_configuration = commissioning.switch_information.configuration

        device.mac_seq_num_capability = bool(
            commissioning.options.mac_seq_num_capability
        )
        device.rx_on_capability = bool(commissioning.options.rx_on_capability)
        device.fixed_location = bool(commissioning.options.fixed_location)
        device.security_key = security_key
        device.security_key_type = security_key_type
        device.security_level = security_level
        # An unprotected GPD has no security frame counter, the GPDF carries its MAC
        # sequence number instead. An outgoing counter it still sends is ignored
        # (A.4.2.1.1.3).
        device.frame_counter = (
            commissioning.outgoing_counter
            if security_level != SecurityLevel.NoSecurity
            else packet.frame_counter
        )

        # The commissioning GPDF is the latest reception from the GPD
        device.last_seen = packet.timestamp
        device.lqi = packet.lqi
        device.rssi = packet.rssi

        return device

    @property
    def src_id(self) -> SrcID | None:
        """The 32-bit GPD SrcID, or None for IEEE-addressed GPDs."""
        return self._src_id

    @property
    def manufacturer_id(self) -> int | None:
        return self.gpd_manufacturer_id

    @property
    def is_initialized(self) -> bool:
        """Commissioning fully describes a GPD, there is no separate interview."""
        return True

    async def reinterview(self) -> None:
        """A GPD cannot be queried, so there is nothing to re-interview."""

    def get_signature(self) -> dict[str, Any]:
        return {
            "application_id": self.application_id,
            "src_id": self._src_id,
            "device_id": self.device_id,
            "manufacturer_id": self.gpd_manufacturer_id,
            "model_id": self.gpd_model_id,
            "commands": list(self.commands),
            "server_cluster_ids": list(self.server_cluster_ids),
            "client_cluster_ids": list(self.client_cluster_ids),
            "switch_configuration": self.switch_configuration,
            "mac_seq_num_capability": self.mac_seq_num_capability,
        }

    @property
    def name(self) -> str:
        if self.application_id is ApplicationID.SrcID:
            return f"GreenPowerDevice 0x{self._src_id:08X}"

        return f"GreenPowerDevice {self._ieee}"

    def __repr__(self) -> str:
        return (
            f"<"
            f"{type(self).__name__}"
            f" application_id={self.application_id!r}"
            f" src_id={self.src_id!r}"
            f" ieee={self.ieee!r}"
            f" device_id={self.device_id!r}"
            f" nwk={self.nwk!r}"
            f">"
        )

    def log(self, lvl: int, msg: str, *args, **kwargs) -> None:
        msg = "[%s] " + msg
        args = (self.name, *args)
        LOGGER.log(lvl, msg, *args, **kwargs)

    def _is_packet_duplicate(self, packet: t.ZigbeeGpPacket) -> bool:
        if self.frame_counter is None or packet.frame_counter is None:
            return False

        if packet.security_level in (
            SecurityLevel.FullFrameCounterAndMIC,
            SecurityLevel.Encrypted,
        ):
            # The security frame counter increments monotonically over the GPD's
            # lifetime, so anything at or below the last value is a replay
            return packet.frame_counter <= self.frame_counter

        # Unprotected GPDFs carry the 8-bit MAC sequence number instead, which both
        # wraps and may be random: only an identical value shortly after the last
        # frame counts as the same GPDF forwarded again
        return (
            packet.frame_counter == self.frame_counter
            and self._last_seen is not None
            and ((packet.timestamp - self._last_seen) < GP_DUPLICATE_TIMEOUT)
        )

    def _process_security(self, packet: t.ZigbeeGpPacket) -> t.ZigbeeGpPacket:
        """Verify and decrypt a protected GPDF that the radio did not process."""
        assert packet.application_id is ApplicationID.SrcID
        assert packet.src_id is not None
        assert packet.frame_counter is not None
        assert packet.command_id is not None
        assert packet.mic is not None
        assert packet.individual_key is not None
        assert packet.auto_commissioning is not None
        assert packet.rx_after_tx is not None

        # Without the commissioned key, or with another level or kind of key than the
        # commissioned one, the GPDF is dropped (spec A.3.7.3.1.2, Table 12)
        if (
            self.security_key is None
            or packet.security_level != self.security_level
            or packet.individual_key
            != (
                self.security_key_type
                in (SecurityKeyType.IndividualKey, SecurityKeyType.DerivedIndividual)
            )
        ):
            return packet.replace(security_status=SecurityStatus.AuthFailure)

        header = build_gpdf_header(
            packet.src_id,
            packet.frame_counter,
            security_level=packet.security_level,
            individual_key=packet.individual_key,
            auto_commissioning=packet.auto_commissioning,
            rx_after_tx=packet.rx_after_tx,
        )

        # AES-CCM over a GPDF takes ~20 us on a CM4. Offloading it to an executor
        # would cost ~200 us for the thread handoff alone, mostly on the event loop
        # itself.
        try:
            plaintext = decrypt_payload(
                packet.src_id,
                packet.frame_counter,
                self.security_key.serialize(),
                bytes([packet.command_id]) + packet.payload.serialize(),
                packet.mic.serialize(),
                header=header,
                security_level=packet.security_level,
            )
        except InvalidTag:
            return packet.replace(security_status=SecurityStatus.AuthFailure)

        return packet.replace(
            command_id=GPDCommandID(plaintext[0]),
            payload=t.SerializableBytes(plaintext[1:]),
            security_key_type=self.security_key_type,
            security_status=SecurityStatus.SecuritySuccess,
        )

    def packet_received(self, packet: t.ZigbeeGpPacket) -> None:
        """Process a decoded GPDF, decrypting it if the radio did not."""

        if packet.security_status is SecurityStatus.Unprocessed:
            packet = self._process_security(packet)

        # Only frames with status NO_SECURITY or SECURITY_SUCCESS are delivered (spec
        # A.3.5.2.5)
        if packet.security_status not in (
            None,
            SecurityStatus.NoSecurity,
            SecurityStatus.SecuritySuccess,
        ):
            self.debug(
                "Dropping packet that failed security processing: %s",
                packet.security_status,
            )
            return

        # The frame's security level and key type must match the commissioned values
        # or it is silently dropped. Checked before anything is stored, so that an
        # unprotected frame cannot reset the anti-replay counter (spec A.3.5.2.4)
        if self.security_level is not None and (
            packet.security_level != self.security_level
            or packet.security_key_type != self.security_key_type
        ):
            self.debug("Dropping packet with mismatched security parameters")
            return

        # A GP device's packets can be relayed by multiple routers so the same GPDF can
        # be received multiple times, without indicating any new liveness from the
        # device: duplicates and replays must not refresh `last_seen`, the link
        # metrics, or the duplicate window itself
        if self._is_packet_duplicate(packet):
            self.debug("Filtering duplicate packet")
            return

        self.last_seen = packet.timestamp

        if packet.lqi is not None:
            self.lqi = packet.lqi

        if packet.rssi is not None:
            self.rssi = packet.rssi

        if packet.frame_counter is not None:
            self.frame_counter = packet.frame_counter

        # A Decommissioning command is only acted upon once it passes the freshness and
        # security checks above (spec A.3.5.2.4)
        if packet.command_id == GPDCommandID.DecommissioningRequest:
            self._application.green_power.decommission(self)
            return

        # As is a Success command, which finishes bidirectional commissioning (spec
        # A.3.9.1 step 18.b)
        if packet.command_id == GPDCommandID.SuccessReport:
            self._application.green_power.commissioning_succeeded(self)
            return

        command: t.Struct | None

        if packet.command_id in GPD_COMMAND_SCHEMAS:
            schema = GPD_COMMAND_SCHEMAS[packet.command_id]

            try:
                command, _ = schema.deserialize(packet.payload.serialize())
            except Exception:  # noqa: BLE001
                self.debug("Failed to parse GPDF payload %r", packet, exc_info=True)
                command = None
        else:
            self.debug(
                "No schema for GPD command %s, dropping payload", packet.command_id
            )
            command = None

        self.emit(
            GreenPowerCommandReceived.event_type,
            GreenPowerCommandReceived(
                device_ieee=str(self.ieee),
                endpoint_id=packet.endpoint,
                command_id=packet.command_id,
                command=command,
            ),
        )


class ZigbeeDevice(BaseDevice):
    """A conventional Zigbee device on the network."""

    manufacturer_id_override = None

    def __init__(self, application: ControllerApplication, ieee: t.EUI64, nwk: t.NWK):
        super().__init__(application, ieee, nwk)
        self.zdo: zdo.ZDO = zdo.ZDO(self)
        self.endpoints: dict[int, zdo.ZDO | zigpy.endpoint.Endpoint] = {0: self.zdo}

        self.ota_in_progress: bool = False

        self._initialize_task: asyncio.Task | None = None
        self._reinterview_in_progress: bool = False
        self._group_scan_task: asyncio.Task | None = None
        self._fast_polling_reset_task: asyncio.Task | None = None

        self._manufacturer: str | None = None
        self._model: str | None = None
        self.node_desc: zdo_t.NodeDescriptor | None = None
        self._requests: dict[ResponseKey, asyncio.Future] = {}
        # Requests that have been handed to the radio and now await a reply
        self._requests_awaiting_reply: set[ResponseKey] = set()
        self._relays: t.Relays | None = None
        self._skip_configuration: bool = False
        self._send_sequence: int = 0

        self._fast_polling = False

        self._packet_debouncer = zigpy.datastructures.Debouncer()
        self._concurrent_requests_semaphore = zigpy.datastructures.RequestLimiter(
            max_concurrency=MAX_DEVICE_CONCURRENCY,
            capacities={
                t.PacketPriority.HIGH: 0.5,
                # t.PacketPriority.NORMAL is shared with LOW
                t.PacketPriority.LOW: 0.5,
            },
        )

        # Retained for backwards compatibility, will be removed in a future release
        self.status = Status.NEW

        self._on_remove_callbacks.append(
            self._application.register_callback_listener(
                src=self,
                filters=[PollControl.ClientCommandDefs.checkin.schema()],
                callback=self.poll_control_checkin_callback,
            )
        )

        self._routing: DeviceRouting = DeviceRouting(self)

        # Requests waiting for this (sleepy) device to be heard, and when it
        # was last heard on the event loop's clock (see _wait_until_awake)
        self._wake_waiters: list[asyncio.Future] = []
        self._last_heard: float | None = None
        # Whether the coordinator itself holds frames for this device, i.e. is
        # its parent (learned when a send expires in the coordinator's own
        # buffer). A frame parked there costs no airtime until the device
        # polls, so requests are kept parked continuously instead of being
        # sent only when the device is heard.
        self._parent_reports_expiry: bool = False

    @contextlib.asynccontextmanager
    async def _limit_concurrency(self, *, priority: int | None = None):
        """Async context manager to limit device request concurrency."""
        # Defer to the current app-level priority if not specified
        if priority is None:
            priority = self._application._packet_priority_var.get()

        start_time = time.monotonic()
        manager: contextlib.AbstractAsyncContextManager

        if priority >= t.PacketPriority.CRITICAL:
            LOGGER.debug(
                "Critical priority request received (%s), skipping queue with %d requests",
                priority,
                self._concurrent_requests_semaphore.waiting_requests,
            )
            manager = contextlib.nullcontext()
            was_locked = False
        else:
            manager = self._concurrent_requests_semaphore(priority=priority)
            was_locked = self._concurrent_requests_semaphore.locked(priority=priority)

        if was_locked:
            LOGGER.debug(
                "Device concurrency (%s) reached, delaying request (%s enqueued)",
                self._concurrent_requests_semaphore.active_requests,
                self._concurrent_requests_semaphore.waiting_requests,
            )

        async with manager:
            if was_locked:
                LOGGER.debug(
                    "Previously delayed device request is now running, delayed by %0.2fs",
                    time.monotonic() - start_time,
                )

            yield

    def get_sequence(self) -> int:
        self._send_sequence = (self._send_sequence + 1) % 256
        return self._send_sequence

    @property
    def name(self) -> str:
        return f"0x{self.nwk:04X}"

    @property
    def non_zdo_endpoints(self) -> list[zigpy.endpoint.Endpoint]:
        return [
            ep for epid, ep in self.endpoints.items() if not (isinstance(ep, zdo.ZDO))
        ]

    @property
    def has_non_zdo_endpoints(self) -> bool:
        return bool(self.non_zdo_endpoints)

    @property
    def all_endpoints_init(self) -> bool:
        return self.has_non_zdo_endpoints and all(
            ep.status != zigpy.endpoint.Status.NEW for ep in self.non_zdo_endpoints
        )

    @property
    def is_initialized(self) -> bool:
        return self.node_desc is not None and self.all_endpoints_init

    @property
    def should_maintain_route(self) -> bool:
        """Whether zigpy should actively maintain a source route to this device.

        True for routers, and for any device that keeps its receiver on while
        idle (``rx_on_when_idle``). The latter covers devices that report
        ``logical_type = EndDevice`` but actually receive continuously -- e.g.
        Xiaomi/Aqara mains switches (``lumi.switch.*``) -- which benefit from a
        warm route just like a router.

        We intentionally do not require the mains-power bit: continuous RX is
        itself the strongest power draw in Zigbee, so an rx-on-when-idle device
        is effectively always-on regardless of how it reports its power source
        (Aqara devices frequently mis-report it as battery). Genuinely sleepy
        end devices have ``rx_on_when_idle = False`` and are correctly excluded.
        """
        if self.nwk == 0x0000:
            return False

        # Explicit per-IEEE override for devices that mis-report their node
        # descriptor (e.g. some Aqara mains switches claiming to be sleepy end
        # devices). Honoured even before the device is interviewed.
        if self.ieee in self._application.config[conf.CONF_NWK_ROUTING_MAINTAIN_DEVICES]:
            return True

        nd = self.node_desc
        if nd is None:
            return False
        if nd.is_router:
            return True
        return bool(nd.is_receiver_on_when_idle)

    @property
    def is_sleepy(self) -> bool:
        """Whether the device turns its receiver off between transmissions.

        Such a device only receives what its parent holds for it when it wakes
        up and polls. Devices we have not interviewed are not treated as
        sleepy: they are awake while joining.
        """
        nd = self.node_desc
        return (
            nd is not None
            and nd.is_end_device
            and not self.should_maintain_route
            and self.nwk != 0x0000
        )

    def _notify_heard(self) -> None:
        """The device sent something: release requests waiting for it."""
        self._last_heard = asyncio.get_running_loop().time()

        for waiter in self._wake_waiters:
            if not waiter.done():
                waiter.set_result(None)

    async def _wait_until_awake(self, deadline: float) -> None:
        """Wait until a sleepy device is likely listening.

        Returns at once if it was heard within ``SLEEPY_AWAKE_WINDOW``,
        otherwise when it is next heard. Raises ``TimeoutError`` when
        ``deadline`` (event loop time) passes first.
        """
        loop = asyncio.get_running_loop()

        if (
            self._last_heard is not None
            and loop.time() - self._last_heard < SLEEPY_AWAKE_WINDOW
        ):
            return

        if len(self._wake_waiters) >= MAX_WAKE_WAITERS:
            self.debug("Too many requests waiting for the device, sending now")
            return

        waiter: asyncio.Future = loop.create_future()
        self._wake_waiters.append(waiter)
        self.debug(
            "Holding request until the device is heard (%d waiting)",
            len(self._wake_waiters),
        )

        try:
            async with asyncio_timeout(max(0.0, deadline - loop.time())):
                await waiter
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"Sleepy device {self.nwk!r} was not heard in time"
            ) from None
        finally:
            self._wake_waiters.remove(waiter)

    def schedule_group_membership_scan(self) -> asyncio.Task:
        """Rescan device group's membership."""
        if self._group_scan_task and not self._group_scan_task.done():
            self.debug("Cancelling old group rescan")
            self._group_scan_task.cancel()

        self._group_scan_task = self.create_task(
            self.group_membership_scan(), name="group_membership_scan"
        )
        return self._group_scan_task

    async def group_membership_scan(self) -> None:
        """Sync up group membership."""
        for ep in self.non_zdo_endpoints:
            await ep.group_membership_scan()

    @property
    def initializing(self) -> bool:
        """Return True if device is being initialized."""
        return self._initialize_task is not None and not self._initialize_task.done()

    @property
    def reinterviewing(self) -> bool:
        """Return True if device is being re-interviewed."""
        return self._reinterview_in_progress

    def cancel_initialization(self) -> None:
        """Cancel initialization call."""
        if self.initializing:
            self.debug("Canceling old initialize call")
            self._initialize_task.cancel()  # type:ignore[union-attr]

    def schedule_initialize(self) -> asyncio.Task | None:
        # Already-initialized devices don't need to be re-initialized
        if self.is_initialized:
            self.debug("Skipping initialization, device is fully initialized")
            self._application.device_initialized(self)
            return None

        if self.reinterviewing:
            self.debug("Skipping initialization, re-interview in progress")
            return None

        self.debug("Scheduling initialization")

        self.cancel_initialization()
        self._initialize_task = self.create_task(self.initialize(), name="initialize")

        return self._initialize_task

    async def reinterview(self) -> None:
        """Re-interview this device using a shadow device for safety.

        Creates a fresh Device, discovers its endpoints/clusters/model info,
        and on success swaps the old device out. On failure, the old device
        and its DB state are preserved.
        """
        if self.reinterviewing:
            self.debug("Re-interview already in progress, skipping")
            return

        if self.initializing:
            self.debug("Initialization in progress, skipping re-interview")
            return

        if self.ota_in_progress:
            self.debug("OTA in progress, skipping re-interview")
            return

        self._reinterview_in_progress = True

        try:
            shadow = ZigbeeDevice(self._application, self._ieee, self.nwk)
            shadow._reinterview_in_progress = True  # prevent auto-initialization

            # The shadow talks to the same physical device: it keeps using the
            # paths already proven for it instead of starting from nothing
            shadow._relays = self._relays
            shadow._routing = self._routing
            shadow._routing.move_to(shadow)

            # Temporarily register the shadow in app.devices so it receives
            # ZDO responses routed by packet_received().
            self._application.devices[self._ieee] = shadow

            try:
                async with self._application.request_priority(
                    t.PacketPriority.CRITICAL
                ):
                    await shadow._discover()
            except Exception:
                # Discovery failed — restore old device, clean up shadow
                self._application.devices[self._ieee] = self
                self._routing.move_to(self)
                shadow.on_remove()
                raise

            # Discovery succeeded — swap the old device for the new one.
            # If this somehow fails, the state may be partially swapped; attempting
            # to undo would likely make things worse.
            await self._application._device_reinterviewed(self, shadow)
        except Exception:  # noqa: BLE001
            self.warning(
                "Re-interview failed, keeping existing device",
                exc_info=True,
            )
            self._application.listener_event("device_reinterview_failure", self)
        finally:
            self._reinterview_in_progress = False

    async def get_node_descriptor(self) -> zdo_t.NodeDescriptor:
        self.info("Requesting 'Node Descriptor'")

        status, _, node_desc = await self.zdo.Node_Desc_req(self.nwk)

        if status != zdo_t.Status.SUCCESS:
            raise zigpy.exceptions.InvalidResponse(
                f"Requesting Node Descriptor failed: {status}"
            )

        self.node_desc = node_desc
        self.info("Got Node Descriptor: %s", node_desc)

        return node_desc

    async def initialize(self) -> None:
        try:
            # Perform initialization with critical priority
            async with self._application.request_priority(t.PacketPriority.CRITICAL):
                await self._initialize()
        except (TimeoutError, zigpy.exceptions.ZigbeeException):
            self.application.listener_event("device_init_failure", self)
        except Exception:  # noqa: BLE001
            LOGGER.warning(
                "Device %r failed to initialize due to unexpected error",
                self,
                exc_info=True,
            )

            self.application.listener_event("device_init_failure", self)

    def find_cluster(
        self, cluster_id: int, cluster_type: ClusterType = ClusterType.Server
    ) -> Cluster:
        """Find the first cluster by its ID and type on any endpoint."""
        for ep in self.non_zdo_endpoints:
            if cluster_type == ClusterType.Server and cluster_id in ep.in_clusters:
                return ep.in_clusters[cluster_id]
            elif cluster_type == ClusterType.Client and cluster_id in ep.out_clusters:
                return ep.out_clusters[cluster_id]
        raise ValueError(
            f"Cluster {cluster_id:#06x} not found in any endpoint of device {self}"
        )

    async def poll_control_checkin_callback(
        self,
        zcl_hdr: foundation.ZCLHeader,
        command: foundation.CommandSchema,
    ) -> None:
        """Handle Poll Control check-in callback."""
        poll_control = self.find_cluster(cluster_id=PollControl.cluster_id)

        # Initiate fast polling mode if we are initializing or waiting for requests to
        # be sent
        fast_poll = (
            self.initializing
            or self.reinterviewing
            or self._concurrent_requests_semaphore.active_requests > 0
            or bool(self._wake_waiters)
            or self._fast_polling
        )

        async with self._application.request_priority(t.PacketPriority.CRITICAL):
            try:
                await poll_control.checkin_response(
                    start_fast_polling=fast_poll,
                    fast_poll_timeout=(
                        int(DEFAULT_FAST_POLL_TIMEOUT * 4) if fast_poll else 0
                    ),
                    tsn=zcl_hdr.tsn,
                    expect_reply=False,
                    disable_default_response=True,
                    # The device is free to stop listening 7.68s after checking in,
                    # any retry would arrive after it has gone back to sleep
                    retries=0,
                )
            except (TimeoutError, DeliveryError) as exc:
                # The device is asleep or out of reach, it will check in again later
                self.debug("Failed to send check-in response: %r", exc)

    async def begin_fast_polling(
        self, timeout: float = DEFAULT_FAST_POLL_TIMEOUT, *, reset_after: bool = True
    ) -> None:
        """Ask the device to enter fast polling mode."""
        try:
            poll_control = self.find_cluster(cluster_id=PollControl.cluster_id)
        except ValueError:
            LOGGER.debug("Device does not support fast polling")
            # The device doesn't have the cluster, there's nothing more we can do
            return

        # Cancel any fast poll reset tasks
        if self._fast_polling_reset_task is not None:
            self._fast_polling_reset_task.cancel()
            self._fast_polling_reset_task = None

        # The units are quarter seconds, we round up to the nearest one
        adjusted_timeout = math.ceil(timeout * 4) / 4
        LOGGER.debug("Beginning fast polling for %0.2fs", adjusted_timeout)

        # We must first bind to the cluster, otherwise the device will not send a check-
        # in command
        await poll_control.bind()
        await poll_control.write_attributes(
            {PollControl.AttributeDefs.fast_poll_timeout.id: int(4 * adjusted_timeout)}
        )

        self._fast_polling = True

        if reset_after:

            async def reset_fast_polling() -> None:
                await asyncio.sleep(adjusted_timeout)
                self._fast_polling = False

            self._fast_polling_reset_task = self.create_task(
                reset_fast_polling(), name="reset_fast_polling"
            )

    @contextlib.asynccontextmanager
    async def fast_poll_mode(
        self, initial_timeout: float = DEFAULT_FAST_POLL_TIMEOUT
    ) -> None:
        """Ask the device to enter fast polling mode."""
        await self.begin_fast_polling(timeout=initial_timeout, reset_after=False)

        try:
            yield
        finally:
            LOGGER.debug("Stopping fast polling on next device check-in")
            self._fast_polling = False

    async def _discover(self) -> None:
        """Discover all basic information about a device: its node descriptor, all
        endpoints and clusters, and the model and manufacturer attributes from any
        Basic cluster exposing those attributes. Does not signal completion.
        """

        # Some devices are improperly initialized and are missing a node descriptor
        if self.node_desc is None:
            await self.get_node_descriptor()

        # Devices should have endpoints other than ZDO
        if self.has_non_zdo_endpoints:
            self.info("Already have endpoints: %s", self.endpoints)
        else:
            self.info("Discovering endpoints")

            status, _, endpoints = await self.zdo.Active_EP_req(self.nwk)

            if status != zdo_t.Status.SUCCESS:
                raise zigpy.exceptions.InvalidResponse(
                    f"Endpoint request failed: {status}"
                )

            self.info("Discovered endpoints: %s", endpoints)

            for endpoint_id in endpoints:
                if endpoint_id != 0:
                    self.add_endpoint(endpoint_id)

        self.status = Status.ZDO_INIT

        # Initialize all of the discovered endpoints
        initiated_fast_polling = self._fast_polling

        if self.all_endpoints_init:
            self.info(
                "All endpoints are already initialized: %s", self.non_zdo_endpoints
            )

            if not initiated_fast_polling:
                # Begin fast polling if we are re-initializing
                await self.begin_fast_polling()
        else:
            self.info("Initializing endpoints %s", self.non_zdo_endpoints)

            for ep in self.non_zdo_endpoints:
                await ep.initialize()

                # Some devices list endpoints as active but have no simple descriptor
                if ep.status == zigpy.endpoint.Status.ENDPOINT_INACTIVE:
                    self.info("Removing inactive endpoint %s", ep.endpoint_id)
                    del self.endpoints[ep.endpoint_id]
                    continue

                if not initiated_fast_polling:
                    # Ask the device to enter fast polling mode as soon as we are
                    # aware of a PollControl cluster
                    try:
                        await self.begin_fast_polling()
                    except (TimeoutError, DeliveryError):
                        pass
                    else:
                        initiated_fast_polling = True

        # Query model info
        if self.model is not None and self.manufacturer is not None:
            self.info("Already have model and manufacturer info")
        else:
            for ep in self.non_zdo_endpoints:
                if self.model is None or self.manufacturer is None:
                    model, manufacturer = await ep.get_model_info()
                    self.info(
                        "Read model %r and manufacturer %r from %s",
                        model,
                        manufacturer,
                        ep,
                    )

                    if model is not None:
                        self.model = model

                    if manufacturer is not None:
                        self.manufacturer = manufacturer

        try:
            ota = self.find_cluster(
                cluster_id=Ota.cluster_id, cluster_type=ClusterType.Client
            )
        except ValueError:
            self.debug("Device does not support OTA cluster")
        else:
            await ota.read_attributes([Ota.AttributeDefs.current_file_version.name])

            # Notify the device that OTA images may be available so it queries us
            try:
                await ota.image_notify(
                    payload_type=Ota.ImageNotifyCommand.PayloadType.QueryJitter,
                    query_jitter=100,
                )
            except Exception:  # noqa: BLE001
                self.debug(
                    "OTA image_notify failed during initialization", exc_info=True
                )

        self.status = Status.ENDPOINTS_INIT

        self.info("Discovered basic device information for %s", self)

    async def _initialize(self) -> None:
        """Discover device information and signal to the application."""
        await self._discover()

        # Signal to the application that the device is ready
        self._application.device_initialized(self)

    def add_endpoint(self, endpoint_id) -> zigpy.endpoint.Endpoint:
        ep = zigpy.endpoint.Endpoint(self, endpoint_id)
        self.endpoints[endpoint_id] = ep
        return ep

    async def add_to_group(self, grp_id: int, name: str | None = None) -> None:
        for ep in self.non_zdo_endpoints:
            await ep.add_to_group(grp_id, name)

    async def remove_from_group(self, grp_id: int) -> None:
        for ep in self.non_zdo_endpoints:
            await ep.remove_from_group(grp_id)

    async def request(
        self,
        profile,
        cluster,
        src_ep,
        dst_ep,
        sequence,
        data,
        expect_reply=True,
        timeout=APS_REPLY_TIMEOUT,
        use_ieee=False,
        ask_for_ack: bool | None = None,
        priority: int | None = None,
        retries: int | None = None,
        retry_delay: float | None = None,
        ping: bool | None = False,
        route: list[t.NWK] | DeviceRouting | zigpy.routing.RouteBase | None = None,
        wake_timeout: float | None = None,
        keep_parked: bool = False,
        **kwargs,
    ):
        """Send a request to the device and wait for its reply.

        ``timeout`` is how long to wait for the reply once the request has
        been sent. ``wake_timeout`` only matters for sleepy end devices: it is
        how long the request may wait for the device to be heard before it is
        sent at all (``None`` uses the ``sleepy_wake_timeout`` setting, 0
        sends immediately). If the device stays silent that long the request
        fails with ``TimeoutError`` without having been sent.

        A sleepy device's request is sent each time the device is heard. With
        ``keep_parked`` it is instead re-sent as soon as the parent reports
        that its held copy expired, so one is always waiting there; behind a
        router that is a transmission every few seconds for the whole wait.
        (When the coordinator is the parent this is always done: it is free.)
        """
        if retries is None:
            retries = DEFAULT_REQUEST_RETRIES

        if retry_delay is None:
            retry_delay = DEFAULT_REQUEST_RETRY_DELAY

        extended_timeout = False

        if self.node_desc is None or self.node_desc.is_end_device:
            self.debug("Extending timeout for 0x%02x request", sequence)
            timeout = APS_REPLY_TIMEOUT_EXTENDED
            extended_timeout = True

        if expect_reply:
            if dst_ep == zdo.ZDO_ENDPOINT:
                rsp_key = ResponseKey(
                    endpoint_id=dst_ep,
                    # e.g. Node_Desc_req = 0x0002 corresponds to Node_Desc_rsp = 0x8002
                    cluster_id=cluster ^ 0x8000,
                    direction=None,
                    tsn=sequence,
                )
            else:
                zcl_hdr, _ = foundation.ZCLHeader.deserialize(data)
                rsp_key = ResponseKey(
                    endpoint_id=dst_ep,
                    cluster_id=cluster,
                    direction=zcl_hdr.frame_control.direction.flip(),
                    tsn=sequence,
                )

            if rsp_key in self._requests:
                self.debug(
                    "Duplicate request key %s, pending requests %s",
                    rsp_key,
                    self._requests,
                )
                raise zigpy.exceptions.ControllerException(
                    f"Duplicate request key: {rsp_key}"
                )
        else:
            rsp_key = None

        max_attempts = retries + 1

        # Requests we initiate to a sleepy device wait until it is listening.
        # Replies to the device are not held (it is awake, it just asked), nor
        # is anything during an interview, when the device stays awake.
        wake_deadline: float | None = None

        if (
            expect_reply
            and self.is_sleepy
            and not self.initializing
            and not self.reinterviewing
        ):
            if wake_timeout is None:
                wake_timeout = self._application.config[
                    conf.CONF_NWK_SLEEPY_WAKE_TIMEOUT
                ]

            if wake_timeout > 0:
                wake_deadline = asyncio.get_running_loop().time() + wake_timeout
                timeout = min(timeout, SLEEPY_HELD_REPLY_TIMEOUT)

        # Use a lambda so we don't leave the coroutine unawaited in case of an exception
        send_request = lambda attempt: self._application.request(  # noqa: E731
            device=self,
            profile=profile,
            cluster=cluster,
            src_ep=src_ep,
            dst_ep=dst_ep,
            sequence=sequence,
            data=data,
            expect_reply=expect_reply,
            use_ieee=use_ieee,
            extended_timeout=extended_timeout,
            ask_for_ack=ask_for_ack,
            priority=priority,
            # FORCED route discovery broadcasts an RREQ before every send even
            # when a route exists, so it is reserved for the narrow case where
            # the NCP likely holds a *stale* route: the last ping attempt to a
            # device we have recently heard from. When the NCP simply has no
            # route (silent/unknown devices), the ENABLE_ROUTE_DISCOVERY APS
            # option -- which bellows sets on every unicast -- already triggers
            # discovery on demand at no extra broadcast cost.
            force_route_discovery=(
                bool(ping)
                and attempt == max_attempts - 1
                and self._routing.believed_reachable()
            ),
            ping=ping,
            route=route,
            attempt=attempt+1,
            max_attempts=max_attempts,
            **kwargs,
        )

        loop = asyncio.get_running_loop()

        async def run_attempts():
            attempt = 0

            while True:
                # A sleepy device is normally only sent to once it has been heard
                # (also before a retry: by then it may be asleep again). If the
                # request is kept parked we do not wait: it is sent now and again
                # each time the parent's copy expires, so it is there whenever the
                # device polls, even at moments we never see.
                if wake_deadline is not None and not (
                    keep_parked or self._parent_reports_expiry
                ):
                    await self._wait_until_awake(wake_deadline)

                send_started = loop.time()
                handed_over = False

                # Requests held for a sleepy device all have to reach its parent
                # within the one short window in which it collects, so they do
                # not queue behind each other for a concurrency slot
                limiter: contextlib.AbstractAsyncContextManager = (
                    self._limit_concurrency(priority=priority)
                    if wake_deadline is None
                    else contextlib.nullcontext()
                )

                try:
                    async with limiter:
                        if not expect_reply:
                            await send_request(attempt=attempt)
                            handed_over = True
                            self._routing.notify_sent(sequence)
                            return None

                        assert rsp_key is not None

                        future: asyncio.Future[list[Any] | foundation.CommandSchema] = (
                            asyncio.Future()
                        )
                        self._requests[rsp_key] = future

                        try:
                            await send_request(attempt=attempt)
                            handed_over = True
                            self.debug(
                                "Request 0x%02x confirmed by the radio after %.0f ms",
                                sequence,
                                (loop.time() - send_started) * 1000,
                            )
                            self._routing.notify_sent(sequence)
                            self._requests_awaiting_reply.add(rsp_key)
                            async with asyncio_timeout(timeout):
                                return await future
                        except asyncio.TimeoutError as to:
                            self.notify_timeout(sequence)
                            # Sent, not reported as expired, and still no reply:
                            # this parent does not tell us about expired frames
                            self._parent_reports_expiry = False
                            raise to
                        finally:
                            if not future.done():
                                future.cancel()
                            self._requests.pop(rsp_key, None)
                            self._requests_awaiting_reply.discard(rsp_key)
                except zigpy.exceptions.ParsingError:
                    raise
                except Exception as exc:
                    LOGGER.debug(
                        "Failed to send request, attempt %d of %d",
                        attempt + 1,
                        max_attempts,
                        exc_info=True,
                    )

                    # Feed the failure back into routing. Only failures that are
                    # about the path demote it: a busy radio (SendError) or a
                    # destination that is itself unavailable says nothing about
                    # the route, while a route error or a send the first hop did
                    # not acknowledge means this path does not work right now.
                    held_for = loop.time() - send_started
                    expired_at_parent = wake_deadline is not None and (
                        (
                            isinstance(exc, DestinationUnreachable)
                            and held_for >= MIN_PARKED_SECONDS
                        )
                        or (
                            isinstance(exc, RouteError)
                            and held_for >= ROUTE_ERROR_PARKED_SECONDS
                        )
                    )

                    if expired_at_parent or isinstance(
                        exc, (SendError, DestinationUnreachable)
                    ):
                        self._routing.notify_no_fault(sequence)
                    elif isinstance(exc, DeliveryError):
                        self._routing.notify_route_error(sequence)

                    # The parent held the frame and the device did not collect it.
                    # This is not a failed attempt: while the caller is still
                    # willing to wait, the request goes out again -- at once if it
                    # is kept parked, otherwise when the device is next heard.
                    if expired_at_parent:
                        # The send itself failed: the copy expired in the
                        # coordinator's own buffer, so the coordinator is the
                        # parent. Otherwise a router held it and told us.
                        self._parent_reports_expiry = not handed_over
                        self.debug(
                            "Request 0x%02x was not collected from the %s after"
                            " %.1fs, %s",
                            sequence,
                            "coordinator" if not handed_over else "parent router",
                            held_for,
                            "parking it again"
                            if keep_parked or self._parent_reports_expiry
                            else "sending it again when the device is heard",
                        )

                        if loop.time() < wake_deadline:
                            continue

                        raise asyncio.TimeoutError(
                            f"Sleepy device {self.nwk!r} did not collect the request"
                            f" in time"
                        ) from exc

                    if attempt >= max_attempts - 1:
                        raise

                    attempt += 1
                    await asyncio.sleep(retry_delay)

        return await run_attempts()

    def handle_message(
        self,
        profile: int,
        cluster: int,
        src_ep: int,
        dst_ep: int,
        message: bytes,
    ):
        """Deprecated compatibility function. Use `packet_received` instead."""

        warnings.warn(
            "`handle_message` is deprecated, use `packet_received`", DeprecationWarning
        )

        self.packet_received(
            t.ZigbeePacket(
                profile_id=profile,
                cluster_id=cluster,
                src_ep=src_ep,
                dst_ep=dst_ep,
                data=t.SerializableBytes(message),
                dst=t.AddrModeAddress(
                    addr_mode=t.AddrMode.NWK,
                    address=self.nwk,
                ),
            )
        )

    def _find_zcl_cluster_strict(
        self, hdr: foundation.ZCLHeader, packet: t.ZigbeePacket
    ) -> Cluster:
        """Find the ZCL cluster for a given header and packet, strict."""
        assert packet.src_ep is not None
        ep = self.endpoints[packet.src_ep]

        if hdr.frame_control.direction == foundation.Direction.Client_to_Server:
            return ep.out_clusters[packet.cluster_id]
        else:
            return ep.in_clusters[packet.cluster_id]

    def _find_zcl_cluster(
        self, hdr: foundation.ZCLHeader, packet: t.ZigbeePacket
    ) -> Cluster:
        """Find the ZCL cluster for a given header and packet."""
        try:
            return self._find_zcl_cluster_strict(hdr, packet)
        except KeyError:
            # If the cluster is not found, try to find it with flipped direction. This
            # will be removed in 2025.9.0.
            cluster = self._find_zcl_cluster_strict(
                hdr.replace(
                    frame_control=hdr.frame_control.replace(
                        direction=hdr.frame_control.direction.flip()
                    )
                ),
                packet,
            )
            LOGGER.debug(
                (
                    "Cluster 0x%04x on %r has incorrect direction (got %r for %r cluster)."
                    " Please report this here: https://github.com/zigpy/zigpy/issues/1640"
                ),
                packet.cluster_id,
                self,
                hdr.frame_control.direction,
                cluster.cluster_type,
            )

            return cluster

    def custom_profile_packet_received(self, packet: t.ZigbeePacket) -> None:
        """Handle packets with a custom profile ID."""
        self.debug(
            "Received packet with custom profile 0x%04x, ignoring",
            packet.profile_id,
        )

    def _should_filter_packet(self, packet: t.ZigbeePacket) -> bool:
        """Check if packet should be filtered as duplicate."""
        return self._packet_debouncer.filter(
            # Be conservative with deduplication
            obj=packet.replace(timestamp=None, tsn=None, lqi=None, rssi=None),
            expire_in=PACKET_DEBOUNCE_WINDOW,
        )

    def _parse_packet_header(
        self, packet: t.ZigbeePacket
    ) -> tuple[zdo_t.ZDOHeader | foundation.ZCLHeader, ResponseKey] | tuple[None, None]:
        """Parse packet header and create response key."""
        data = packet.data.serialize()

        if packet.src_ep == zdo.ZDO_ENDPOINT:
            hdr, _ = zdo_t.ZDOHeader.deserialize(packet.cluster_id, data)
            rsp_key = ResponseKey(
                endpoint_id=packet.src_ep,
                cluster_id=packet.cluster_id,
                direction=None,
                tsn=hdr.tsn,
            )
            return hdr, rsp_key
        elif packet.profile_id in (zha.PROFILE_ID, zll.PROFILE_ID):
            hdr, _ = foundation.ZCLHeader.deserialize(data)
            rsp_key = ResponseKey(
                endpoint_id=packet.src_ep,
                cluster_id=packet.cluster_id,
                direction=hdr.frame_control.direction,
                tsn=hdr.tsn,
            )
            return hdr, rsp_key
        else:
            return None, None

    def _match_packet_endpoint_cluster(
        self, packet: t.ZigbeePacket, hdr: zdo_t.ZDOHeader | foundation.ZCLHeader
    ) -> (
        tuple[zigpy.endpoint.Endpoint, Cluster]
        | tuple[zigpy.zdo.ZDO, None]
        | tuple[None, None]
    ):
        """Validate packet routing and find target endpoint and cluster."""
        if packet.src_ep not in self.endpoints:
            self.debug(
                "Ignoring message on unknown endpoint %s (expected one of %s)",
                packet.src_ep,
                self.endpoints,
            )
            return None, None

        endpoint = self.endpoints[packet.src_ep]

        if packet.src_ep == zdo.ZDO_ENDPOINT:
            assert isinstance(endpoint, zigpy.zdo.ZDO)
            return endpoint, None
        else:
            assert isinstance(endpoint, zigpy.endpoint.Endpoint)
            try:
                zcl_cluster = self._find_zcl_cluster(hdr, packet)
            except KeyError:
                self.debug(
                    "Ignoring message on unknown cluster: 0x%04x",
                    packet.cluster_id,
                )
                return None, None
            else:
                return endpoint, zcl_cluster

    def _parse_packet_command(
        self, packet: t.ZigbeePacket, endpoint: typing.Any, zcl_cluster: Cluster | None
    ) -> typing.Any:
        """Deserialize packet data."""
        data = packet.data.serialize()

        if packet.src_ep == zdo.ZDO_ENDPOINT:
            _, cmd = endpoint.deserialize(packet.cluster_id, data)
        else:
            assert zcl_cluster is not None
            _, cmd = zcl_cluster.deserialize(data)

        return cmd

    def _maybe_match_response(
        self, rsp_key: ResponseKey, cmd: typing.Any | None, error: Exception | None
    ) -> bool:
        """Handle response matching for pending requests, returns True if packet was matched."""
        future = self._requests.get(rsp_key)
        if future is None:
            return False

        # Attribute reports often collide with command responses, they should never be
        # matched up
        if isinstance(
            cmd,
            foundation.GENERAL_COMMANDS[
                foundation.GeneralCommand.Report_Attributes
            ].schema,
        ):
            return False

        try:
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(cmd)
        except asyncio.InvalidStateError:
            self.debug(
                "Invalid state on future for %s -- probably duplicate response",
                rsp_key,
            )

        return True

    def packet_received(self, packet: t.ZigbeePacket) -> None:
        """Process received packet through the device's packet handling pipeline."""
        self.last_seen = packet.timestamp

        if packet.lqi is not None:
            self.lqi = packet.lqi

        if packet.rssi is not None:
            self.rssi = packet.rssi

        # Hearing from the device resets its ping backoff so a reconnected
        # device is re-probed and re-routed promptly.
        self._routing.notify_seen()

        # ... and releases requests held back until a sleepy device is awake
        self._notify_heard()

        # Filter duplicate packets
        if self._should_filter_packet(packet):
            self.debug("Filtering duplicate packet")
            return

        # Parse packet header and create response key
        hdr, rsp_key = self._parse_packet_header(packet)
        if hdr is None:
            self.custom_profile_packet_received(packet)
            return

        # Validate packet routing and find target endpoint/cluster
        endpoint, zcl_cluster = self._match_packet_endpoint_cluster(packet, hdr)
        if endpoint is None:
            return

        # Deserialize packet data
        try:
            cmd = self._parse_packet_command(packet, endpoint, zcl_cluster)
        except Exception as exc:  # noqa: BLE001
            cmd = None
            error = zigpy.exceptions.ParsingError()
            error.__cause__ = exc
            self.debug("Failed to parse packet %r", packet, exc_info=error)
        else:
            error = None

        # Handle response matching for pending requests
        if self._maybe_match_response(rsp_key, cmd, error):
            # The packet was positively matched (by cluster, direction and
            # application-layer TSN) to a request we sent, so it is safe to
            # credit the route that carried it. Even a reply that failed to
            # parse proves delivery worked.
            self._routing.notify_reply(rsp_key.tsn, packet)
            return

        # Skip further processing if there was a parsing error
        if error is not None:
            return

        # Pass the request off to a listener, if one is registered
        for listener in itertools.chain(
            self._application._req_listeners[zigpy.listeners.ANY_DEVICE],
            self._application._req_listeners[self],
        ):
            # Resolve only until the first future listener
            if listener.resolve(hdr, cmd) and isinstance(
                listener, zigpy.listeners.FutureListener
            ):
                break

        # Finally, pass it off to the cluster message handler. This will be removed.
        if zcl_cluster is not None:
            zcl_cluster.handle_message(hdr, cmd)
        else:
            assert isinstance(endpoint, zdo.ZDO)
            endpoint.handle_message(packet.profile_id, packet.cluster_id, hdr, cmd)

    async def reply(
        self,
        profile,
        cluster,
        src_ep,
        dst_ep,
        sequence,
        data,
        timeout=APS_REPLY_TIMEOUT,
        expect_reply: bool = False,
        use_ieee: bool = False,
        ask_for_ack: bool | None = None,
        priority: int | None = None,
        **kwargs,
    ):
        return await self.request(
            profile=profile,
            cluster=cluster,
            src_ep=src_ep,
            dst_ep=dst_ep,
            sequence=sequence,
            data=data,
            expect_reply=expect_reply,
            timeout=timeout,
            use_ieee=use_ieee,
            ask_for_ack=ask_for_ack,
            priority=priority,
            **kwargs,
        )

    async def update_firmware(
        self,
        image: OtaImageWithMetadata,
        progress_callback: Callable[[int, int, float], None] | None = None,
        force: bool = False,
    ) -> foundation.Status | None:
        """Update device firmware."""
        if self.ota_in_progress:
            self.debug("OTA already in progress")
            return None

        self.ota_in_progress = True

        try:
            result = await update_firmware(
                device=self,
                image=image,
                progress_callback=progress_callback,
                force=force,
            )
        except Exception as exc:  # noqa: BLE001
            self.debug("OTA failed!", exc_info=exc)
            raise
        finally:
            self.ota_in_progress = False

        if result != foundation.Status.SUCCESS:
            return result

        # Clear the current file version when the update succeeds
        ota = self.find_cluster(
            cluster_id=Ota.cluster_id, cluster_type=ClusterType.Client
        )
        ota.update_attribute(Ota.AttributeDefs.current_file_version.id, None)
        ota.last_query_cmd = None
        ota.emit(
            OtaQueryCacheClearedEvent.event_type,
            OtaQueryCacheClearedEvent(
                device_ieee=str(self.ieee),
                endpoint_id=ota.endpoint.endpoint_id,
            ),
        )

        await asyncio.sleep(AFTER_OTA_ATTR_READ_DELAY)
        await ota.read_attributes(
            [Ota.AttributeDefs.current_file_version.name],
            retries=10,
            retry_delay=AFTER_OTA_ATTR_READ_DELAY,
        )

        # Prompt device to send QueryNextImage with updated version for query cache
        try:
            await ota.image_notify(
                payload_type=Ota.ImageNotifyCommand.PayloadType.QueryJitter,
                query_jitter=100,
            )
        except Exception:  # noqa: BLE001
            self.debug("Post-OTA image_notify failed", exc_info=True)

        # Re-interview the device after successful OTA to pick up
        # any changes in clusters/endpoints/model and re-apply quirks.
        # reinterview() handles its own errors internally.
        await self.reinterview()

        return result

    def get_last_ota_query_cmd(self) -> QueryNextImageCommand | None:
        """Return the last cached QueryNextImageCommand, preferring client clusters."""
        for ep in self.non_zdo_endpoints:
            # Prefer client (out) clusters since runtime routing places
            # query_next_image there when both cluster types exist.
            for clusters in (ep.out_clusters, ep.in_clusters):
                cluster = clusters.get(Ota.cluster_id)
                if isinstance(cluster, Ota) and cluster.last_query_cmd is not None:
                    return cluster.last_query_cmd
        return None

    def log(self, lvl, msg, *args, **kwargs) -> None:
        msg = "[0x%04x] " + msg
        args = (self.nwk, *args)
        LOGGER.log(lvl, msg, *args, **kwargs)

    @property
    def manufacturer(self) -> str | None:
        return self._manufacturer

    @manufacturer.setter
    def manufacturer(self, value) -> None:
        if isinstance(value, str):
            self._manufacturer = value

    @property
    def manufacturer_id(self) -> int | None:
        """Return manufacturer id."""
        if self.manufacturer_id_override:
            return self.manufacturer_id_override
        elif self.node_desc is not None:
            return self.node_desc.manufacturer_code
        else:
            return None

    @property
    def model(self) -> str | None:
        return self._model

    @model.setter
    def model(self, value) -> None:
        if isinstance(value, str):
            self._model = value

    @property
    def skip_configuration(self) -> bool:
        return self._skip_configuration

    @skip_configuration.setter
    def skip_configuration(self, should_skip_configuration) -> None:
        if isinstance(should_skip_configuration, bool):
            self._skip_configuration = should_skip_configuration
        else:
            self._skip_configuration = False

    @property
    def relays(self) -> t.Relays | None:
        """Relay list."""
        return self._relays

    @relays.setter
    def relays(self, relays: t.Relays | None) -> None:
        if relays is None:
            pass
        elif not isinstance(relays, t.Relays):
            relays = t.Relays(relays)

        self._relays = relays
        # A fresh route record supersedes any demotion earned by the previous
        # (possibly stale) relay list
        self._routing.notify_relays_updated()
        self.listener_event("device_relays_updated", relays)

    def __getitem__(self, key):
        return self.endpoints[key]

    def clone(self) -> ZigbeeDevice:
        """Return a detached copy of this device for independent modification."""
        new = ZigbeeDevice(self.application, self.ieee, self.nwk)
        new.lqi = self.lqi
        new.rssi = self.rssi
        new.last_seen = self.last_seen
        new.relays = self.relays
        new.original_signature = self.original_signature
        new.status = self.status
        new.node_desc = self.node_desc
        new.manufacturer = self.manufacturer
        new.model = self.model
        new.manufacturer_id_override = self.manufacturer_id_override
        new.skip_configuration = self.skip_configuration

        for endpoint in self.non_zdo_endpoints:
            new_ep = new.add_endpoint(endpoint.endpoint_id)
            new_ep.status = endpoint.status
            new_ep.profile_id = endpoint.profile_id
            new_ep.device_type = endpoint.device_type

            for cluster in endpoint.in_clusters.values():
                new_cluster = new_ep.add_input_cluster(cluster.cluster_id)
                new_cluster._attr_cache_internal = cluster._attr_cache.clone(
                    new_cluster
                )

            for cluster in endpoint.out_clusters.values():
                new_cluster = new_ep.add_output_cluster(cluster.cluster_id)
                new_cluster._attr_cache_internal = cluster._attr_cache.clone(
                    new_cluster
                )

        return new

    def get_signature(self) -> dict[str, typing.Any]:
        # return the device signature by providing essential device information
        #    - Model Identifier ( Attribute 0x0005 of Basic Cluster 0x0000 )
        #    - Manufacturer Name ( Attribute 0x0004 of Basic Cluster 0x0000 )
        #    - Endpoint list
        #        - Profile Id, Device Id, Cluster Out, Cluster In
        signature: dict[str, typing.Any] = {}
        if self._manufacturer is not None:
            signature[SIG_MANUFACTURER] = self.manufacturer
        if self._model is not None:
            signature[SIG_MODEL] = self._model
        if self.node_desc is not None:
            signature[SIG_NODE_DESC] = self.node_desc.as_dict()

        for endpoint_id, endpoint in self.endpoints.items():
            if endpoint_id == zdo.ZDO_ENDPOINT:  # ZDO
                continue
            signature.setdefault(SIG_ENDPOINTS, {})
            in_clusters = list(endpoint.in_clusters)
            out_clusters = list(endpoint.out_clusters)
            signature[SIG_ENDPOINTS][endpoint_id] = {
                SIG_EP_PROFILE: endpoint.profile_id,
                SIG_EP_TYPE: endpoint.device_type,
                SIG_EP_INPUT: in_clusters,
                SIG_EP_OUTPUT: out_clusters,
            }
        return signature

    def build_route(self, tsn: t.uint8_t, ping: bool, attempt: int, max_attempts: int) -> list[t.NWK] | None:
        """Build a route to the device based on its relays."""

        return self._routing.build_route(tsn, ping, attempt, max_attempts)

    def notify_route_error(self, tsn: t.uint8_t) -> None:
        self._routing.notify_route_error(tsn)

    def fail_requests_awaiting_reply(
        self, exc: Exception, *, relayed_only: bool = False
    ) -> int:
        """Fail in-flight requests that were sent and are waiting for a reply.

        With ``relayed_only`` requests sent directly are left alone: the
        device acknowledged those itself, so a failure reported by a relay
        cannot be about them.

        Used when the network reports that the device cannot be reached (e.g. a
        route error), so `request` fails immediately and its retry logic can
        pick another route instead of waiting for the reply timeout. Requests
        not yet handed to the radio are left alone: they may take a different
        route. Returns the number of requests failed.
        """
        failed = 0

        for rsp_key in self._requests_awaiting_reply:
            future = self._requests.get(rsp_key)
            if future is None or future.done():
                continue

            if (
                relayed_only
                and self._routing.tsn_route.get(rsp_key.tsn)
                is self._routing.direct_route
            ):
                continue

            future.set_exception(exc)
            failed += 1

        return failed

    def notify_timeout(self, tsn: t.uint8_t) -> None:
        self._routing.notify_timeout(tsn)

    async def ping_using_route_works(
        self, route: list[t.NWK] | zigpy.routing.RouteBase
    ) -> bool:
        """Whether a ping over the given explicit route gets a reply.

        ``route`` may be a relay list or a RouteBase (e.g. ``AutomaticRoute``
        to probe the coordinator's own routing). Never raises -- callers in
        the background maintenance loop treat any failure as "no".
        """
        for n in range(2):
            try:
                params, param_types = zdo_t.CLUSTERS[zdo_t.ZDOCmd.IEEE_addr_req]
                tsn = self.get_sequence()
                zdo_payload = bytes([tsn]) + t.serialize([self.nwk, 0, 0], param_types)

                # ping the device
                ret = await self.request(
                    profile=0x0000,
                    cluster=zdo_t.ZDOCmd.IEEE_addr_req,
                    src_ep=0,
                    dst_ep=0,
                    sequence=tsn,
                    data=zdo_payload,
                    expect_reply=True,
                    ask_for_ack=True,  # Ensure we get a transport acknowledgment
                    priority=t.PacketPriority.LOW,
                    ping=True,
                    route=route,
                    retries=0,
                )
                # Any reply proves the route, whatever status it carries
                LOGGER.debug("Ping to %s succeeded using route %s: %s", self.nwk, route, ret)
                return True
            except asyncio.TimeoutError:
                LOGGER.debug("Ping to %s failed with timeout using route %s", self.nwk, route)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug(
                    "Ping to %s failed using route %s: %r", self.nwk, route, exc
                )
        return False

    async def request_rejoin(self) -> None:
        """Ask the device to leave the network and immediately rejoin.

        Sends ``Mgmt_Leave_req`` with ``Rejoin=1`` / ``RemoveChildren=0``: the
        device performs a secured NWK rejoin and gets a fresh parent -- the
        remote fix for an end device whose parent aged it out of its child
        table. The request is sent over the best working source route when one
        is known, since a parentless device is typically unreachable via
        normal routing. The device keeps its address, bindings, and database
        entry.

        Caution: some legacy stacks (older Xiaomi/Aqara) may leave without
        rejoining and then need a power cycle or re-pairing.
        """
        route = self._routing.usable_source_route()
        self.info(
            "Requesting leave-with-rejoin (via %s route) to restore the parent link",
            route.name if route is not None else "default",
        )
        await self.zdo.leave(remove_children=False, rejoin=True, route=route)

    async def ping(
        self,
        route: zigpy.routing.RouteBase | None = None,
        retries: int | None = None,
    ):
        """Ping the device with a ZDO IEEE address request.

        Without ``route`` the routing ladder picks which route to probe.
        """

        if not self.should_maintain_route:
            # only ping routers and mains-powered rx-on-when-idle devices; not
            # the coordinator or sleepy end devices
            return None

        params, param_types = zdo_t.CLUSTERS[zdo_t.ZDOCmd.IEEE_addr_req]
        tsn = self.get_sequence()
        zdo_payload = bytes([tsn]) + t.serialize([self.nwk, 0, 0], param_types)

        # Failure accounting (notify_timeout / notify_route_error) is done
        # inside Device.request; notifying here as well double-counted every
        # failed ping against its route.
        return await self.request(
            profile=0x0000,
            cluster=zdo_t.ZDOCmd.IEEE_addr_req,
            src_ep=0,
            dst_ep=0,
            sequence=tsn,
            data=zdo_payload,
            expect_reply=True,
            ask_for_ack=True,  # Ensure we get a transport acknowledgment
            priority=t.PacketPriority.LOW,
            ping=True,
            route=route,
            retries=retries,
        )

    def __repr__(self) -> str:
        return (
            f"<"
            f"{type(self).__name__}"
            f" model={self.model!r}"
            f" manuf={self.manufacturer!r}"
            f" nwk={t.NWK(self.nwk)}"
            f" ieee={self.ieee}"
            f" is_initialized={self.is_initialized}"
            f">"
        )


# Backwards-compatible alias
Device = ZigbeeDevice


async def broadcast(
    app,
    profile,
    cluster,
    src_ep,
    dst_ep,
    grpid,
    radius,
    sequence,
    data,
    broadcast_address=t.BroadcastAddress.RX_ON_WHEN_IDLE,
):
    return await app.broadcast(
        profile,
        cluster,
        src_ep,
        dst_ep,
        grpid,
        radius,
        sequence,
        data,
        broadcast_address=broadcast_address,
    )
