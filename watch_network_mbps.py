import time
from pathlib import Path

def read_net():
    data = {}
    for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
        iface, rest = line.split(":", 1)
        iface = iface.strip()
        cols = rest.split()
        rx = int(cols[0])
        tx = int(cols[8])
        if iface != "lo":
            data[iface] = (rx, tx)
    return data

prev = read_net()
prev_t = time.time()

while True:
    time.sleep(1)
    now = read_net()
    now_t = time.time()
    dt = now_t - prev_t

    print("\033c", end="")
    print("Network throughput inside container")
    print("=" * 42)

    total_rx = 0
    total_tx = 0

    for iface, (rx, tx) in now.items():
        old_rx, old_tx = prev.get(iface, (rx, tx))
        rx_mbps = (rx - old_rx) * 8 / dt / 1_000_000
        tx_mbps = (tx - old_tx) * 8 / dt / 1_000_000
        total_rx += max(0, rx - old_rx)
        total_tx += max(0, tx - old_tx)
        print(f"{iface}: download={rx_mbps:7.2f} Mbps upload={tx_mbps:7.2f} Mbps")

    print("-" * 42)
    print(f"TOTAL: download={total_rx*8/dt/1_000_000:7.2f} Mbps upload={total_tx*8/dt/1_000_000:7.2f} Mbps")

    prev = now
    prev_t = now_t
