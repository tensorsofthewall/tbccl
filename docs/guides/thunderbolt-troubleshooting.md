# Thunderbolt link troubleshooting

Use this page when the link from [Two machines over Thunderbolt](../getting-started/two-machine-thunderbolt.md) does not come up or misbehaves. Every entry has the same shape: the symptom, a **read-only** check, the likely causes, a **safe next step**, and when to ask for help.

**Rules for every entry.** Never reset the controller, rescan or remove PCI devices, unload and reload drivers in a loop, disable PCIe error reporting (AER), bypass the IOMMU, change power management, or unplug and replug repeatedly to see if it helps. Repeated reconnecting can hide a bad cable and, on a marginal link, make it worse. If a step below asks you to reconnect, do it **once**.

Start with the checker on both machines. It only reads state:

```sh
python3 thunderbolt_link_doctor.py --peer <peer-link-address>
```

Its report names the failing area; jump to the matching entry. The final **Summary** lists the detected interface, the selected route, ping and TCP results and the checks that were **not performed**; a verdict of `INCOMPLETE` (exit status 3) means nothing failed but the link was not verified end to end, and only `READY` (exit status 0) means interface, route, ping and TCP all ran and passed.

## Which entry do I need?

| What you see | Entry |
|---|---|
| No Thunderbolt device listed, no `0-1` entry, Mac port says "No device connected" | [1](#1-the-cable-is-not-detected) |
| Device listed but no network interface (Linux) or no Thunderbolt Bridge (Mac) | [2](#2-controller-present-no-network-interface) |
| Interface exists, no carrier / `inactive` / `NO-CARRIER` | [3](#3-interface-exists-but-no-carrier) |
| Interface up, no IPv4 address | [4](#4-no-ipv4-address) |
| Addresses on different subnets | [5](#5-wrong-or-mismatched-subnet) |
| Ping goes out another interface, VPN or Wi-Fi | [6](#6-conflicting-routes) |
| Ping works one way only, or neither way | [7](#7-ping-fails-in-one-or-both-directions) |
| Ping works, TCP or the framework hangs | [8](#8-ping-works-but-tcp-fails-or-hangs) |
| macOS prompts or blocks | [9](#9-macos-permissions) |
| Linux firewall | [10](#10-linux-firewall) |
| IPv4 fine, IPv6 link-local missing (Exo discovery) | [11](#11-ipv4-works-ipv6-link-local-does-not) |
| `AER` / PCIe errors in the report or kernel log | [12](#12-pcie-or-aer-errors) |
| Link drops or the interface disappears | [13](#13-unexpected-disconnects) |

## 1. The cable is not detected

**Symptom.** Linux: `/sys/bus/thunderbolt/devices/` has the host (`0-0`) but no peer (`0-1`). macOS: `system_profiler SPThunderboltDataType` says `Status: No device connected` for every port.

**Check (read-only).**

```sh
ls /sys/bus/thunderbolt/devices/                              # Linux
system_profiler SPThunderboltDataType | grep -E 'Status|Speed' # macOS
```

**Likely causes.** The cable is not Thunderbolt/USB4 data-capable (charge-only, USB 2, or too long or damaged); the port is plain USB-C; the cable goes through a hub, dock or adapter; the other machine's port is disabled in firmware or the machine is asleep.

**Safe next step.** Use a cable marked Thunderbolt or USB4, connected directly between two Thunderbolt ports. Try the other port on one machine, once. Confirm in the machine's specification that the port is Thunderbolt/USB4.

**Escalate.** If a known-good cable and port still show nothing on either side, report the machine models and the doctor output.

## 2. Controller present, no network interface

**Symptom.** The peer is listed but there is no `thunderbolt0` (Linux), or the Mac has no Thunderbolt Bridge service.

**Check.**

```sh
lsmod | grep thunderbolt ; modinfo thunderbolt_net | head -3    # Linux
cat /sys/bus/thunderbolt/devices/domain0/security
nmcli device status
networksetup -listallhardwareports | grep -A1 'Thunderbolt Bridge'   # macOS
```

**Likely causes.** Linux: the `thunderbolt_net` driver is missing or not loaded, or the device is not authorized (security level `user` or `secure`). macOS: the Thunderbolt Bridge service was deleted.

**Safe next step.** Linux: if the module exists but is not loaded, run `sudo modprobe thunderbolt_net` once yourself and check again. If the security level is not `none`, authorize the peer with `boltctl` or your desktop's Thunderbolt panel (not a blanket auto-authorize). macOS: add the service again (System Settings → Network → Add Service → Thunderbolt Bridge).

**Escalate.** A kernel without `thunderbolt_net` needs a different kernel; report the distribution and kernel version.

## 3. Interface exists but no carrier

**Symptom.** Linux `ip -br link` shows `NO-CARRIER` or `DOWN`; macOS `ifconfig` shows `status: inactive`.

**Check.**

```sh
ip -br link show thunderbolt0 ; cat /sys/class/net/thunderbolt0/carrier
ifconfig bridge0 | grep -E 'status|member'
```

**Likely causes.** The other machine's interface is down or not configured; the peer is unauthorized; the cable was disturbed; one machine is asleep.

**Safe next step.** Run the checker on the **other** machine and fix its first failure; the carrier appears once both ends are up. Do not repeatedly replug.

## 4. No IPv4 address

**Symptom.** The doctor reports `address: ... has no IPv4 address`.

**Check.** `ip -br addr show thunderbolt0` or `ifconfig bridge0`.

**Likely causes.** The profile was never created or activated; the NetworkManager profile is bound to another interface name; the macOS service is set to DHCP and there is no DHCP server on a direct cable.

**Safe next step.** Re-do section 4 or 5 of the setup guide: a manual IPv4 address on each end, same subnet. On Linux run `nmcli -f NAME,DEVICE,STATE connection show --active` to see whether your profile is active and on the right device.

## 5. Wrong or mismatched subnet

**Symptom.** The doctor reports `subnet: peer ... is not in the same subnet`. Or the two addresses differ in more than the last number (with a `/24` mask).

**Check.** Compare `ip -br addr` (Linux) and `ifconfig bridge0 | grep inet` (macOS).

**Likely causes.** Typo; different masks; one side kept an old address (for example `192.168.3.x`).

**Safe next step.** Put both addresses in one subnet with the same mask (`192.168.77.1/24` and `192.168.77.2/24`). Do not guess; read each machine's actual address.

## 6. Conflicting routes

**Symptom.** The doctor reports `route: traffic ... would leave through <another interface>`.

**Check.**

```sh
ip route get <peer>               # Linux
route -n get <peer>               # macOS
ip -4 route ; netstat -rn -f inet # all routes
```

**Likely causes.** Your link subnet overlaps Wi-Fi, a VPN, a VM or Docker network, ZeroTier or Tailscale. The more specific or earlier route wins, and it is not the cable.

**Safe next step.** Choose a link subnet that nothing else uses and set it on both machines. Do not delete the other network's routes.

## 7. Ping fails in one or both directions

**Symptom.** `ping` from A to B fails, from B to A works (or both fail).

**Check.** Run the doctor on both sides. Check the exact address and the route first (entries 5 and 6).

**Likely causes.** One side has no address or the wrong one (entries 4 to 5); the interface is not up (entry 3); a firewall drops ICMP only on one machine (entries 9 and 10); one side's routing sends replies elsewhere.

**Safe next step.** Fix the first FAIL on the failing side. If both sides look right and both directions fail, go to entry 1 to 3 for the cable. ICMP being blocked does not mean TCP is blocked, so continue with the TCP check in entry 8.

## 8. Ping works but TCP fails or hangs

**Symptom.** Ping passes. The doctor's TCP check fails, or a framework run hangs at startup.

**Check.** Use the bundled listener in each direction (setup guide section 7.3). Try the other direction too: a firewall on only one machine is common.

**Likely causes.** A firewall on the receiving machine (entries 9 and 10); the program was not approved to use the local network (macOS); an application bound to `127.0.0.1` or to another interface; a mismatched MTU after somebody enabled jumbo frames on one side only; the hostname resolves to another address (PyTorch rendezvous).

**Safe next step.** Compare `MTU` on both sides in the doctor report; if they differ, set both to the default (1500) or both to the same larger value. Pin the address in the framework (for example PyTorch's `--local-addr` and `--master-addr`, see the framework tutorial). Check entries 9 and 10.

## 9. macOS permissions

**Symptom.** ICMP works; TCP connects hang or fail with `No route to host`; a dialog appeared once and was dismissed.

**Check.** System Settings → Privacy & Security → Local Network (is the program listed and on?). System Settings → Network → Firewall (is it on, and is the program blocked?). `/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate` shows the firewall state and changes nothing.

**Safe next step.** Allow the specific program. Over SSH, approve from a local session. Do not turn the firewall off and do not script changes to it.

## 10. Linux firewall

**Symptom.** Same as entry 8, and the machine runs `firewalld`, `ufw`, `nftables` or `iptables` rules.

**Check.** `sudo firewall-cmd --get-active-zones`, `sudo ufw status`, or `sudo nft list ruleset` (these read, they do not change). Look for a rule dropping input on the Thunderbolt interface.

**Safe next step.** Allow traffic from the peer's link address on the Thunderbolt interface only, with your firewall's own tool. The rule's form depends on your tool; this is not scripted and not tested here. Do not disable the firewall.

## 11. IPv4 works, IPv6 link-local does not

**Symptom.** TCP over IPv4 passes. Exo does not discover the other machine. The doctor reports `ipv6-link-local` as WARN.

**Check.** `ip -6 addr show thunderbolt0` (needs an `fe80::` address); `ifconfig bridge0 | grep inet6`. Try `ping -6 fe80::<peer>%<interface>`.

**Likely causes.** IPv6 is disabled on the interface (`ipv6.method disabled` in NetworkManager, or "Link-local only"/"Off" on macOS); a sysctl turns IPv6 off.

**Safe next step.** Set Linux `ipv6.method link-local` and macOS IPv6 to Automatic on the link, then re-run the doctor. PyTorch and vLLM do not need IPv6; Exo does.

## 12. PCIe or AER errors

**Symptom.** The doctor reports `pcie-errors` WARN (correctable) or FAIL (nonfatal/fatal), or the kernel log mentions `AER`, `pcieport` errors, or Thunderbolt timeouts.

**Check.** `cat /sys/bus/pci/devices/<controller>/aer_dev_correctable` (and `_nonfatal`, `_fatal`). The doctor locates the controller. Compare the counts before and after a short test.

**Likely causes.** A marginal cable, a damaged port, or an over-long cable. A handful of correctable errors can be normal; counts that rise while you run traffic are not.

**Safe next step.** If **nonfatal or fatal** errors appear, or correctable errors grow quickly, stop all traffic over the link. Swap to a short certified Thunderbolt 4/USB4 cable and measure again. Do not turn AER reporting off and do not bypass the IOMMU to hide the messages.

**Escalate.** Report the doctor's JSON, the kernel version and the controller model. The only validated controller is Intel Thunderbolt 4 (Maple Ridge); others are not tested.

## 13. Unexpected disconnects

**Symptom.** The interface vanishes and returns, or `NO-CARRIER` appears during a run.

**Check.** `journalctl -k --since "10 min ago" | grep -i -E 'thunderbolt|usb4|aer'` (needs permission to read the log); the doctor's `kernel-log` line.

**Likely causes.** A loose or poor cable; the machine's power management put the controller or the port to sleep; a sleeping or rebooting peer; heat.

**Safe next step.** Reseat the cable once, firmly, or use a different cable. Stop the run first. Keep power management settings as they are while diagnosing, and do not change them as a fix.

**Escalate.** If a known-good cable still drops, file an issue with the doctor report and the log excerpt ([getting help](../project/getting-help.md)). Remove IP addresses and hostnames you do not want to share; the doctor's JSON already masks MAC addresses and IPv6 suffixes.
