# Two machines over Thunderbolt: first-time setup

This tutorial takes two computers that have never been connected and ends with a verified IP link between them over a Thunderbolt cable. After it, the framework pages show how to run a collective across the link ([PyTorch](https://torch-tbccl.tensorsofthewall.com/en/latest/getting-started/quickstart.html), [vLLM](https://vllm-tbccl.tensorsofthewall.com/en/latest/), [Exo](https://exo-tbccl.tensorsofthewall.com/en/latest/)).

It sets up the network only. It does not need TBCCL, PyTorch or a GPU, and nothing in it requires an administrator password except where a step says so.

```{note}
**What has been tested.** The procedure was written from, and checked against, one pair: an Apple silicon Mac mini (macOS 26) and an x86-64 Linux desktop (kernel 6.18, NetworkManager 1.56) with an Intel Thunderbolt 4 controller, joined by a Thunderbolt cable. That pair was already configured when the tutorial was written, so the *configuration* steps below (the new `192.168.77.x` addresses, the new NetworkManager profile, running with the default MTU) are written from the tools' documented behaviour and offline syntax checks and have **not yet been run from scratch on real hardware**. Mac to Mac and Linux to Linux are not covered. Windows is not supported. Where a step is unverified, it says so. If something here does not match what you see, please [report it](../project/getting-help.md).
```

## 0. How to read this page

Every command block is labelled with the machine it runs on: **Mac** or **Linux**. Run it in a terminal on that machine, in your home directory unless the text says otherwise. Text in `<angle brackets>` is a value you replace; names such as `192.168.77.1` are examples chosen in section 2, and you can use others as long as you use them consistently. A line starting with `#` is a comment, not part of the command. Each section ends with what success looks like, and the [troubleshooting guide](../guides/thunderbolt-troubleshooting.md) says what to do when a step fails.

## 1. What you need

- **Two computers with Thunderbolt or USB4 ports.** A port is Thunderbolt/USB4 only if it has the lightning-bolt symbol, or the machine's specification says Thunderbolt 3/4/5 or USB4. A plain USB-C port (USB 3.x, DisplayPort-only, charging-only) will not carry a computer-to-computer link even though the plug is the same shape.
- **One cable that supports Thunderbolt or USB4 data**, connected **directly** from a Thunderbolt port on one computer to a Thunderbolt port on the other. Use a cable marked Thunderbolt (or a certified USB4 cable). A charge-only or USB 2 cable has the right plug and carries no link. Use a short cable (one metre or less is the safe choice).
- **No switch, router, dock or Ethernet adapter in between.** This is a point-to-point cable between the two machines.
- **Operating systems.** macOS (Apple silicon) and Linux (x86-64) are the tested systems. On Linux you need a kernel with USB4/Thunderbolt and Thunderbolt IP networking (`thunderbolt` and `thunderbolt_net` modules; most current distribution kernels have them). Windows is out of scope.
- **Tools.** Python 3 on both machines to run the checker script. On Linux, `ip` (iproute2) and NetworkManager's `nmcli` if your system uses NetworkManager. Both machines need you to be able to change network settings; on Linux that means `sudo` for the profile step.

### What this link is, and what it is not

The link here is **Thunderbolt IP networking**: the cable carries ordinary Ethernet-style frames and each computer gets a network interface with an IP address. TBCCL's collectives travel as normal TCP over it.

It is not PCIe tunnelling (an external GPU or storage enclosure), it is not a Thunderbolt dock, and it is not RDMA. TBCCL does not use hardware RDMA over this link. Expect an ordinary TCP link whose round trip is a few tenths of a millisecond, and bandwidth well below the cable's headline figure.

### The checker you will use

`tools/thunderbolt_link_doctor.py` is a single standard-library Python file that only reads state. It never changes addresses, routes, MTU, firewall rules or device authorization, never loads modules and never resets anything. You can use it without installing TBCCL:

```sh
curl -fsSLO https://raw.githubusercontent.com/tensorsofthewall/tbccl/main/tools/thunderbolt_link_doctor.py
python3 thunderbolt_link_doctor.py            # local checks only
```

Run it on each machine after each section below. It ends with a **Summary** (detected link interface, local address, peer, selected route, ping and TCP result, concerns, checks not performed) and a verdict: `READY` only when the interface, route, ping and TCP checks all ran and passed, `INCOMPLETE` when nothing failed but something was not checked (for example no `--peer` or no TCP test), `NOT READY` when something failed. Its exit status is 0, 3 or 1 respectively, so a plain run without `--peer` never reports success by itself. The summary is also where it tells you the real name of the Thunderbolt interface.

## 2. Pick the addresses

You choose a private subnet that **nothing else on either computer already uses**. This tutorial uses:

| | Address | Subnet mask |
|---|---|---|
| Mac | `192.168.77.1` | `255.255.255.0` (`/24`) |
| Linux | `192.168.77.2` | `255.255.255.0` (`/24`) |

Before you commit to it, check that `192.168.77.0/24` is not in use. A subnet that matches your Wi-Fi, a VPN, a VM network or an overlay network (ZeroTier, Tailscale, Docker) makes the system send link traffic to the wrong place.

```sh
# Linux
ip -4 route
# macOS
netstat -rn -f inet
```

If any line mentions `192.168.77.`, choose another subnet and substitute it **everywhere** below (both machines, every command, and the PyTorch commands later). For example, with `10.77.0.0/24`:

| In this page | Becomes |
|---|---|
| `192.168.77.1/24` (Mac) | `10.77.0.1/24` |
| `192.168.77.2/24` (Linux) | `10.77.0.2/24` |
| `192.168.77.1` / `192.168.77.2` in commands | `10.77.0.1` / `10.77.0.2` |
| netmask `255.255.255.0` | unchanged |

The two addresses must share the first three numbers (for a `/24`) and differ in the last. Avoid ranges your Wi-Fi, router, VPN, Docker or VM software already uses. Leave **gateway and DNS empty** on this link: it is a private cable, not a path to the internet, and giving it a default route would redirect your normal traffic.

## 3. Connect the cable and look at both machines

Plug the cable into a Thunderbolt port on each machine. Wait about ten seconds. On Linux the interface does not exist until the cable is connected and the peer is ready, so you cannot configure it before this.

### macOS: what the Mac sees

On macOS the link is a **Thunderbolt Bridge** network service. It can combine all of the Mac's Thunderbolt ports into one bridge interface, so it does not matter which port you use, and the address belongs on the bridge, not on an individual port. The bridge is usually named `bridge0`, but read the actual name from your machine rather than assuming it:

```sh
# Mac
networksetup -listallnetworkservices
networksetup -listallhardwareports     # look for "Hardware Port: Thunderbolt Bridge" and its Device
system_profiler SPThunderboltDataType  # ports that say "Status: Device connected"
```

Note the **Device** printed for *Thunderbolt Bridge* (on the tested Mac it was `bridge0`; yours may differ, for example `bridge1`). Store it in a variable so the rest of this page works unchanged:

```sh
# Mac
BR=$(networksetup -listallhardwareports | awk '/Hardware Port: Thunderbolt Bridge/{f=1} f&&/^Device:/{print $2; exit}')
echo "Thunderbolt Bridge device: ${BR:-NOT FOUND}"
ifconfig "$BR"
```

If `NOT FOUND` is printed, the service does not exist yet: create it in section 4, then repeat.

In `ifconfig`, a `status: active` line and `member: enN` entries mean a cable is carrying a link; `status: inactive` means no Thunderbolt port has a working link yet.

### Linux: what the Linux machine sees

```sh
# Linux
ls /sys/bus/thunderbolt/devices/        # a "domain0", the host "0-0", and the peer "0-1" once the cable is detected
lsmod | grep thunderbolt                # expect: thunderbolt and thunderbolt_net
ip -br link                             # a new interface appears when the cable and peer are ready
nmcli device status                     # if you use NetworkManager
cat /sys/bus/thunderbolt/devices/domain0/security
```

The interface is usually called `thunderbolt0`, but distributions and udev rules can rename it. Find the real name by its driver, and keep it in a variable for the rest of this page:

```sh
# Linux
TB_IF=$(for d in /sys/class/net/*; do [ "$(basename "$(readlink "$d/device/driver" 2>/dev/null)")" = thunderbolt-net ] && basename "$d"; done | head -n1)
echo "Thunderbolt interface: ${TB_IF:-NOT FOUND}"
```

If several interfaces use the driver, pick the one that goes with the cable you just connected (`ip -br link` shows which has a carrier). `NOT FOUND` means no Thunderbolt network interface exists yet; see the checks below.

- If `/sys/bus/thunderbolt/devices/` has no peer entry: the cable is not detected. Check the cable and ports first (see the [troubleshooting guide](../guides/thunderbolt-troubleshooting.md)).
- If the peer appears but no interface does and `thunderbolt_net` is not listed, your kernel lacks the driver. You can try `sudo modprobe thunderbolt_net` once, yourself, and look again. Do not script module reloads, and do not reset the controller.
- If `domain0/security` is anything other than `none` (for example `user` or `secure`), the peer must be **authorized** before the link works. Use the mechanism your distribution provides, normally `boltctl list` and then `boltctl enroll <uuid>` for the device you recognise (or your desktop's Thunderbolt settings panel). Do not enable blanket auto-authorization. This is described, not tested on a `user`-security machine.

## 4. Configure the Mac

Use the graphical settings; they give the same result on every recent macOS.

1. **System Settings → Network.** Look for **Thunderbolt Bridge** in the list.
2. If it is missing: click the **…** (or **+**) button below the list, choose **Add Service…**, set the interface to **Thunderbolt Bridge**, and click Create. (On older macOS, **System Preferences → Network → + → Thunderbolt Bridge**.)
3. Click **Thunderbolt Bridge → Details… → TCP/IP**.
4. Set **Configure IPv4** to **Manually**, **IP Address** `192.168.77.1`, **Subnet Mask** `255.255.255.0`. **Leave Router empty.**
5. Click **DNS** and leave it empty. Click **OK**. Leave IPv6 on its default (Automatic): Exo uses IPv6 link-local addresses on this link.

Check it:

```sh
# Mac
ifconfig "$BR" | grep -E 'inet |status|mtu'
route -n get 192.168.77.2 | grep interface
```

Expect `inet 192.168.77.1 netmask 0xffffff00`, `status: active` (cable connected), and `interface:` showing your Thunderbolt Bridge device (the value of `$BR`) for the route (before the Linux side is configured the route may still show another interface; recheck at the end).

If a Mac has a VPN or a firewall product that filters local traffic, read §8.

## 5. Configure Linux

Do not change anything yet: first find out what already manages this interface.

```sh
# Linux
nmcli -f NAME,UUID,TYPE,DEVICE connection show      # all existing profiles
nmcli -f DEVICE,TYPE,STATE,CONNECTION device status  # is your $TB_IF listed, and "managed" or "unmanaged"?
```

- **A profile already uses `$TB_IF`** (for example one you made earlier): stop here and use it. Note its name and addresses (`nmcli connection show "<name>"`) and, if it has other addresses, use those instead of the example ones everywhere, or edit that profile yourself. Do not create a second profile for the same interface and do not overwrite the existing one.
- **A profile named `thunderbolt-direct` exists but is something else**: pick another `con-name`.
- **The interface shows `unmanaged`, or NetworkManager is not installed**: use the alternative at the end of this section.
- **Otherwise** create a new profile.

**Graphical alternative:** Settings → Network → + (add a wired connection), choose the Thunderbolt interface, under IPv4 choose **Manual**, address `192.168.77.2`, netmask `255.255.255.0`, **leave gateway and DNS empty**, tick "Use this connection only for resources on its network" (this is `never-default`), keep IPv6 on **Link-local only**, save.

**Command line** (uses your `$TB_IF`; the profile binds to that interface name):

```sh
# Linux
sudo nmcli connection add type ethernet ifname "$TB_IF" con-name thunderbolt-direct \
    ipv4.method manual ipv4.addresses 192.168.77.2/24 ipv4.never-default yes \
    ipv6.method link-local ipv6.never-default yes
sudo nmcli connection up thunderbolt-direct
```

If `thunderbolt0` is indeed your interface name, this is the same as writing `ifname thunderbolt0`.

Verify:

```sh
# Linux
ip -br addr show "$TB_IF"          # expect 192.168.77.2/24 and an fe80:: address
ip route get 192.168.77.1          # expect: dev <your Thunderbolt interface>, no "via"
```

**Undo it later:** `sudo nmcli connection delete thunderbolt-direct` removes only the profile you created.

### Alternative when NetworkManager does not manage the interface

This is a fallback, not the main path, and has not been tested on a real link. With `systemd-networkd`, a file such as `/etc/systemd/network/50-thunderbolt.network` containing

```ini
[Match]
Driver=thunderbolt-net

[Network]
Address=192.168.77.2/24
LinkLocalAddressing=ipv6
IPv6AcceptRA=no
```

and `sudo networkctl reload` gives the same result (no gateway is configured; check with `networkctl status "$TB_IF"`). For a **temporary** test on any Linux (gone after the cable is removed or on reboot): `sudo ip addr add 192.168.77.2/24 dev "$TB_IF"` and `sudo ip link set "$TB_IF" up`. Do not add a default route.

```{note}
The `nmcli connection add` form was checked for syntax and for the settings it produces on NetworkManager 1.56 without activating it. Activating it from scratch has not been tested on a clean machine.
```

## 6. MTU: leave it alone

Both operating systems come up with a working default. **Do not change the MTU to get started.** Larger frames ("jumbo frames", MTU 9000) can raise throughput on some setups, but they only work if **both** ends use the same value, and a mismatch makes large transfers stall while ping still works. Treat jumbo frames as an optional tuning step after everything here works. The doctor prints each side's MTU so you can compare.

## 7. Check the link

Do these checks from both machines. Ping alone is not enough.

### 7.1 Local state

```sh
# Linux
python3 thunderbolt_link_doctor.py --peer 192.168.77.1
# Mac
python3 thunderbolt_link_doctor.py --peer 192.168.77.2
```

At this point the verdict will say `INCOMPLETE` (no TCP test yet); that is expected until 7.3. Every line should be `PASS` or `INFO`. `SKIP` means a check could not run (for example, a missing privilege) and the report tells you why. `FAIL` lines say what to do next.

### 7.2 Ping, and which interface was used

Linux:

```sh
# Linux
ping -c 3 192.168.77.1
ip route get 192.168.77.1        # "dev" must be your Thunderbolt interface ($TB_IF)
```

macOS:

```sh
# Mac
ping -c 3 192.168.77.2
route -n get 192.168.77.2        # "interface:" must be your Thunderbolt Bridge device ($BR)
```

Replies take a few tenths of a millisecond (about 0.2 to 0.9 ms on the tested pair). If the route names another interface (Wi-Fi, a VPN, `zt…`), the link traffic would not use the cable: pick a different subnet (§2).

### 7.3 A real TCP connection

TBCCL uses TCP, not ping, so test TCP in **both** directions. The doctor includes a harmless echo listener that only accepts connections on the address you give it and stops by itself after two minutes.

On the Mac, start the listener, then on Linux connect to it:

```sh
# Mac (leave running)
python3 thunderbolt_link_doctor.py --listen 192.168.77.1 --tcp-port 50555
# Linux (in another terminal)
python3 thunderbolt_link_doctor.py --peer 192.168.77.1 --tcp-port 50555
```

Then swap: listener on Linux (`--listen 192.168.77.2`), check from the Mac (`--peer 192.168.77.2`). Both runs must report `PASS tcp` and the verdict `READY`. Pick another port if 50555 is taken.

The first time a program listens or connects on the local network, macOS can show a permission prompt (see §8). Approve it for the program (Python); if you are connected over SSH the prompt may appear only on the Mac's own screen.

### 7.4 Leave it running and try something real

Once ping and TCP pass in both directions, the link is ready. The next step is a collective: install `torch-tbccl` on both machines and run the [two-host all-reduce](https://torch-tbccl.tensorsofthewall.com/en/latest/getting-started/quickstart.html). Use the addresses from this page.

## 8. Permissions and firewalls

### macOS

- **Local Network** (System Settings → Privacy & Security → Local Network): the first program that talks on the local network asks. If a connection stalls or reports `No route to host` although ping works, check that the program (Terminal, your SSH daemon wrapper, or Python) is allowed. Over SSH the prompt may not appear; approve it from a local session or from Screen Sharing.
- **Application firewall** (System Settings → Network → Firewall): if it is on, macOS may ask whether Python may accept incoming connections. Choose **Allow** for that program. Never turn the whole firewall off to make this work, and do not script firewall changes.
- **VPN and overlay software** can capture all traffic, including the link subnet. If `route -n get` shows a `utun` interface, change the link subnet or exclude it in the VPN's settings.
- A process left running after its SSH session closes loses local network access; run Mac processes in a live session or `tmux`.

### Linux

A host firewall that blocks incoming TCP on the Thunderbolt interface makes ping work and TCP fail. TBCCL and PyTorch's rendezvous pick ports at run time, so rather than opening one port, allow traffic **from the peer's address on the Thunderbolt interface only**, using your firewall's own tool. For example, with `firewalld` the interface can be placed in a trusted zone, or with `nftables` a rule can accept `iifname "<thunderbolt interface>" ip saddr 192.168.77.1`. Pick the narrowest form your tool offers. These are pointers, not a tested recipe; see your distribution's firewall documentation. Do not disable the firewall.

## 9. Disconnect cleanly

Stop any programs using the link, then unplug the cable. On Linux the interface and its connection go away by themselves; on the Mac the bridge returns to `status: inactive`. To remove what you created: delete the `thunderbolt-direct` profile on Linux (`sudo nmcli connection delete thunderbolt-direct`) and the Thunderbolt Bridge address on the Mac (set Configure IPv4 back to **Using DHCP**). If you reuse your existing configuration, leave it as it is.

## 10. When something is wrong

Work through the [Thunderbolt troubleshooting guide](../guides/thunderbolt-troubleshooting.md). It lists, for each symptom, a read-only diagnostic, the likely causes and a safe next step. If the doctor reports PCIe errors or the link drops repeatedly, stop sending traffic and replace the cable before anything else; do not try to reset the controller.

An existing link that was configured some other way, with different addresses or jumbo frames, is described in [the reference for a configured link](../guides/thunderbolt-link.md).
