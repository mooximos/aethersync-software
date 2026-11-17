"""INA226 battery voltage and current sensor on I2C (bus 1, address 0x40, 0.01 ohm shunt)."""
from __future__ import annotations

import time
from typing import Callable

REG_CONFIG, REG_BUS_V, REG_CURRENT, REG_CALIBRATION = 0x00, 0x02, 0x04, 0x05
REG_MANUFACTURER, REG_DIE = 0xFE, 0xFF
# Average 16 samples, 1.1 ms conversions for bus and shunt, measure both continuously.
CONFIG = 0x4127 | (0b010 << 9)
BUS_V_LSB = 0.00125   # volts per bit


class INA226:
    def __init__(self, bus: int = 1, address: int = 0x40, shunt_ohm: float = 0.01, max_a: float = 5.0,
                 smbus_factory: Callable | None = None) -> None:
        if smbus_factory is None:
            from smbus2 import SMBus
            smbus_factory = SMBus
        self.bus = smbus_factory(bus)
        self.address = address
        self.current_lsb = max_a / 32768                          # amps per bit
        self.calibration = int(0.00512 / (self.current_lsb * shunt_ohm))
        self._write(REG_CONFIG, CONFIG)
        self._write(REG_CALIBRATION, self.calibration)

    def _write(self, register: int, value: int) -> None:
        self.bus.write_i2c_block_data(self.address, register, [(value >> 8) & 0xFF, value & 0xFF])

    def _read(self, register: int) -> int:
        high, low = self.bus.read_i2c_block_data(self.address, register, 2)
        return (high << 8) | low

    def ids(self) -> tuple[int, int]:
        """(manufacturer ID, die ID); a real INA226 gives (0x5449, 0x2260)."""
        return self._read(REG_MANUFACTURER), self._read(REG_DIE)

    def read(self) -> tuple[float | None, float | None]:
        """(bus volts, amps). Retries once on an I2C error, then gives (None, None)."""
        for attempt in range(2):
            try:
                volts = self._read(REG_BUS_V) * BUS_V_LSB
                raw = self._read(REG_CURRENT)
                amps = (raw - 65536 if raw & 0x8000 else raw) * self.current_lsb
                return round(volts, 3), round(amps, 3)
            except OSError:
                if attempt == 0:
                    time.sleep(0.01)
        return None, None


class LowBatteryWatch:
    """True once the voltage has stayed below the limit for the whole hold time."""

    def __init__(self, below_v: float, hold_s: float) -> None:
        self.below_v, self.hold_s = below_v, hold_s
        self.low_since: float | None = None

    def update(self, now: float, volts: float | None) -> bool:
        if volts is None or volts >= self.below_v:
            self.low_since = None
            return False
        if self.low_since is None:
            self.low_since = now
        return now - self.low_since >= self.hold_s
