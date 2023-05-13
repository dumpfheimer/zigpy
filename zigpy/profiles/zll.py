from __future__ import annotations

import zigpy.types as t

PROFILE_ID = 49246


class DeviceType(t.enum16):
    ON_OFF_LIGHT = 0x0000
    ON_OFF_PLUGIN_UNIT = 0x0010
    DIMMABLE_LIGHT = 0x0100
    DIMMABLE_PLUGIN_UNIT = 0x0110
    COLOR_LIGHT = 0x0200
    EXTENDED_COLOR_LIGHT = 0x0210
    COLOR_TEMPERATURE_LIGHT = 0x0220

    COLOR_CONTROLLER = 0x0800
    COLOR_SCENE_CONTROLLER = 0x0810
    CONTROLLER = 0x0820
    SCENE_CONTROLLER = 0x0830
    CONTROL_BRIDGE = 0x0840
    ON_OFF_SENSOR = 0x0850


CLUSTERS = {
    DeviceType.ON_OFF_LIGHT: ([0x0004, 0x0005, 0x0006, 0x0008, 0x1000], []),
    DeviceType.ON_OFF_PLUGIN_UNIT: ([0x0004, 0x0005, 0x0006, 0x0008, 0x1000], []),
    DeviceType.DIMMABLE_LIGHT: ([0x0004, 0x0005, 0x0006, 0x0008, 0x1000], []),
    DeviceType.DIMMABLE_PLUGIN_UNIT: ([0x0004, 0x0005, 0x0006, 0x0008, 0x1000], []),
    DeviceType.COLOR_LIGHT: ([0x0004, 0x0005, 0x0006, 0x0008, 0x0300, 0x1000], []),
    DeviceType.EXTENDED_COLOR_LIGHT: (
        [0x0004, 0x0005, 0x0006, 0x0008, 0x0300, 0x1000],
        [],
    ),
    DeviceType.COLOR_TEMPERATURE_LIGHT: (
        [0x0004, 0x0005, 0x0006, 0x0008, 0x0300, 0x1000],
        [],
    ),
    DeviceType.COLOR_CONTROLLER: ([], [0x0004, 0x0006, 0x0008, 0x0300]),
    DeviceType.COLOR_SCENE_CONTROLLER: ([], [0x0004, 0x0005, 0x0006, 0x0008, 0x0300]),
    DeviceType.CONTROLLER: ([], [0x0004, 0x0006, 0x0008]),
    DeviceType.SCENE_CONTROLLER: ([], [0x0004, 0x0005, 0x0006, 0x0008]),
    DeviceType.CONTROL_BRIDGE: ([], [0x0004, 0x0005, 0x0006, 0x0008, 0x0300]),
    DeviceType.ON_OFF_SENSOR: ([], [0x0004, 0x0005, 0x0006, 0x0008, 0x0300]),
}
