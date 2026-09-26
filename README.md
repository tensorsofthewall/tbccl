# tbccl


### macOS Local Network permission

On macOS, `tb_pingpong` requires Local Network access.

If the client connects from Linux to macOS but stalls, or macOS returns
`No route to host` despite working ping/routing, check:

System Settings → Privacy & Security → Local Network

and allow `tb_pingpong`.

When developing over Remote SSH, macOS may not present the permission
dialog in the remote session. A local GUI session may be required to
approve it.