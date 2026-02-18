from __future__ import annotations

import asyncio
from asyncio import timeout as asyncio_timeout
import logging
import pathlib
import socket
import typing
from typing import Literal
import urllib.parse

try:
    # serialx is API-compatible with pyserial
    import serialx as pyserial
    import serialx as pyserial_asyncio
except ImportError:
    import serial as pyserial
    import serial_asyncio_fast as pyserial_asyncio

from zigpy.typing import UNDEFINED, UndefinedType

LOGGER = logging.getLogger(__name__)
DEFAULT_SOCKET_PORT = 6638
SOCKET_CONNECT_TIMEOUT = 5


class SerialProtocol(asyncio.Protocol):
    """Base class for packet-parsing serial protocol implementations."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._transport: pyserial_asyncio.SerialTransport | None = None

        self._connected_event = asyncio.Event()
        self._disconnected_event = asyncio.Event()
        self._disconnected_event.set()

    async def wait_until_connected(self) -> None:
        """Wait for the protocol's transport to be connected."""
        await self._connected_event.wait()

    def connection_made(self, transport: pyserial_asyncio.SerialTransport) -> None:
        LOGGER.debug("Connection made: %s", transport)

        self._transport = transport
        self._disconnected_event.clear()
        self._connected_event.set()

    def connection_lost(self, exc: BaseException | None) -> None:
        LOGGER.debug("Connection lost: %r", exc)
        self._connected_event.clear()
        self._disconnected_event.set()
        self._transport = None

    def data_received(self, data: bytes) -> None:
        self._buffer += data

    def close(self) -> None:
        self._buffer.clear()

        if self._transport is not None:
            self._transport.close()

    async def wait_until_closed(self) -> None:
        LOGGER.debug("Waiting for serial port to close")
        await self._disconnected_event.wait()

    async def disconnect(self) -> None:
        self.close()
        await self.wait_until_closed()


async def create_serial_connection(
    loop: asyncio.BaseEventLoop,
    protocol_factory: typing.Callable[[], asyncio.Protocol],
    url: pathlib.Path | str,
    *,
    baudrate: int = 115200,  # We default to 115200 instead of 9600
    exclusive: bool | None = True,
    xonxoff: bool | UndefinedType = UNDEFINED,
    rtscts: bool | UndefinedType = UNDEFINED,
    flow_control: Literal["hardware", "software"] | None | UndefinedType = UNDEFINED,
    **kwargs: typing.Any,
) -> tuple[asyncio.Transport, asyncio.Protocol]:
    """Wrapper around pyserial-asyncio that transparently substitutes a normal TCP
    transport and protocol when a `socket` connection URI is provided.
    """

    if flow_control is not UNDEFINED:
        xonxoff = flow_control == "software"
        rtscts = flow_control == "hardware"

    if xonxoff is UNDEFINED:
        xonxoff = False

    if rtscts is UNDEFINED:
        rtscts = False

    LOGGER.debug(
        "Opening a serial connection to %r (baudrate=%s, xonxoff=%s, rtscts=%s)",
        url,
        baudrate,
        xonxoff,
        rtscts,
    )

    url = str(url)
    parsed_url = urllib.parse.urlparse(url)

    if parsed_url.scheme in ("socket", "tcp"):
        async with asyncio_timeout(SOCKET_CONNECT_TIMEOUT):
            transport, protocol = await loop.create_connection(
                protocol_factory=lambda: QuickAckProtocolProxy(protocol_factory),
                host=parsed_url.hostname,
                port=parsed_url.port or DEFAULT_SOCKET_PORT,
            )
            try:
                sock = transport.get_extra_info('socket')
                if sock is None:
                    LOGGER.debug("TCP socket: Transport has no socket, cannot optimize TCP socket buffer sizes")
                else:
                    # we want a large read buffer to prevent backpressure on the coordinator or its bridge
                    min_recv_buffer_size = 1024 * 128
                    current_recv_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                    if min_recv_buffer_size > current_recv_buffer_size:
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, min_recv_buffer_size)
                        current_recv_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                        if current_recv_buffer_size == min_recv_buffer_size:
                            LOGGER.debug("TCP socket: changed receive buffer size to %s bytes", current_recv_buffer_size)

                    # this is disputable
                    # it comes down to two different factors.
                    # a smaller write buffer could block writes, letting zigpy know that there is congestion
                    # that could be used within zigpy to reorder packets or cancel out conflicting requests (e.g a "on" request after an "off" request do the same device/endpoint)
                    # a larger write buffer can reduce backpressure on zigpy and calls between userspace and kernel.
                    #
                    # I think as zigpy/ZHA architecture is right now, it will not profit from blocking at writes.
                    # if the decision is made to use the small buffer it should be slightly higher than the largest expected zigbee packet (I believe without TCP overhead, but that should be verified)
                    min_send_buffer_size = 1024 * 128
                    current_send_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
                    if min_send_buffer_size != current_send_buffer_size:
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, min_send_buffer_size)
                        current_send_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
                        if current_send_buffer_size == min_send_buffer_size:
                            LOGGER.debug("TCP socket: changed send buffer size to %s bytes", current_send_buffer_size)
            except Exception:
                LOGGER.exception("Failed to configure TCP socket buffer sizes")

    else:
        try:
            try:
                transport, protocol = await pyserial_asyncio.create_serial_connection(
                    loop,
                    protocol_factory,
                    url=url,
                    baudrate=baudrate,
                    exclusive=exclusive,
                    xonxoff=xonxoff,
                    rtscts=rtscts,
                    **kwargs,
                )
            except pyserial.SerialException as exc:
                # Unwrap unnecessarily wrapped PySerial exceptions
                if exc.__context__ is not None:
                    raise exc.__context__ from None

                raise
        except BlockingIOError as exc:
            # Re-raise a more useful exception
            raise PermissionError(
                "The serial port is locked by another application"
            ) from exc

    return transport, protocol


TCP_QUICKACK = getattr(socket, 'TCP_QUICKACK', 12)

class QuickAckProtocolProxy(asyncio.Protocol):
    """
    Wraps an existing asyncio.Protocol to enforce TCP_QUICKACK
    on every read operation.
    """

    def __init__(self, original_protocol):
        if isinstance(original_protocol, asyncio.Protocol):
            self.protocol = original_protocol
        elif callable(original_protocol):
            self.protocol = original_protocol()
        else:
            raise TypeError("original_protocol must be asyncio.Protocol or callable returning one")
        self.sock = None

    def _set_quickack(self):
        """Helper to re-enable QUICKACK safely."""
        if self.sock:
            try:
                self.sock.setsockopt(socket.IPPROTO_TCP, TCP_QUICKACK, 1)
            except OSError:
                # Socket might be closed or not TCP
                pass

    def connection_made(self, transport):
        # 1. Capture the socket
        self.sock = transport.get_extra_info('socket')

        if self.sock:
            # Enable TCP_NODELAY (disable Nagle) permanently
            try:
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass

            # Enable initial QUICKACK
            self._set_quickack()

        # 2. Delegate to the wrapped protocol
        self.protocol.connection_made(transport)

    def data_received(self, data):
        # 1. Re-enable QUICKACK immediately!
        # The kernel just disabled it because it processed this packet.
        self._set_quickack()

        # 2. Delegate payload to business logic
        self.protocol.data_received(data)

    def connection_lost(self, exc):
        self.protocol.connection_lost(exc)

    # Proxy other standard Protocol methods just in case your inner protocol needs them
    def eof_received(self):
        if hasattr(self.protocol, 'eof_received'):
            return self.protocol.eof_received()
        else:
            return None

    def pause_writing(self):
        if hasattr(self.protocol, 'pause_writing'):
            self.protocol.pause_writing()

    def resume_writing(self):
        if hasattr(self.protocol, 'resume_writing'):
            self.protocol.resume_writing()