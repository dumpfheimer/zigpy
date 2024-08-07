from __future__ import annotations

import typing

import zigpy.types as t

if typing.TYPE_CHECKING:
    from zigpy.config import CONF_OTA_PROVIDER_TYPE

CONF_OTA_PROVIDER_TYPE = "type"

CONF_DEVICE_BAUDRATE_DEFAULT = 115200
CONF_DEVICE_FLOW_CONTROL_DEFAULT = None
CONF_STARTUP_ENERGY_SCAN_DEFAULT = True
CONF_MAX_CONCURRENT_REQUESTS_DEFAULT = 8
CONF_NWK_BACKUP_ENABLED_DEFAULT = True
CONF_NWK_BACKUP_PERIOD_DEFAULT = 24 * 60  # 24 hours
CONF_NWK_CHANNEL_DEFAULT = None
CONF_NWK_CHANNELS_DEFAULT = [11, 15, 20, 25]
CONF_NWK_EXTENDED_PAN_ID_DEFAULT = None
CONF_NWK_PAN_ID_DEFAULT = None
CONF_NWK_KEY_DEFAULT = None
CONF_NWK_KEY_SEQ_DEFAULT = 0x00
CONF_NWK_TC_ADDRESS_DEFAULT = None
CONF_NWK_TC_LINK_KEY_DEFAULT = t.KeyData(b"ZigBeeAlliance09")
CONF_NWK_UPDATE_ID_DEFAULT = 0x00
CONF_NWK_VALIDATE_SETTINGS_DEFAULT = False
CONF_OTA_ENABLED_DEFAULT = True
CONF_OTA_DISABLE_DEFAULT_PROVIDERS_DEFAULT: list[str] = []
CONF_OTA_BROADCAST_ENABLED_DEFAULT = True
CONF_OTA_BROADCAST_INITIAL_DELAY_DEFAULT = 3.9 * 60 * 60  # 3.9 hours
CONF_OTA_BROADCAST_INTERVAL_DEFAULT = 3.9 * 60 * 60  # 3.9 hours
CONF_OTA_PROVIDERS_DEFAULT = [
    {
        CONF_OTA_PROVIDER_TYPE: "ledvance",
    },
    {
        CONF_OTA_PROVIDER_TYPE: "sonoff",
    },
    {
        CONF_OTA_PROVIDER_TYPE: "inovelli",
    },
    {
        CONF_OTA_PROVIDER_TYPE: "thirdreality",
    },
]
CONF_OTA_EXTRA_PROVIDERS_DEFAULT: list[dict[str, typing.Any]] = []
CONF_SOURCE_ROUTING_DEFAULT = False
CONF_TOPO_SCAN_PERIOD_DEFAULT = 4 * 60  # 4 hours
CONF_TOPO_SCAN_ENABLED_DEFAULT = True
CONF_TOPO_SKIP_COORDINATOR_DEFAULT = False
CONF_WATCHDOG_ENABLED_DEFAULT = True
