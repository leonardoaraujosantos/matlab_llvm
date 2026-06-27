//===----------------------------------------------------------------------===//
// Instrument Control / base-MATLAB networking — `tcpclient` / `tcpserver` /
// `udpport` runtime.
//
// Backs the MATLAB-side handle classes (runtime/toolbox/instrument/
// instrument_classdefs.m): thin `handle` wrappers whose one-line method bodies
// forward the receiver `obj` (+ matrix/string/scalar args) into the
// `matlab_*` entries below — the same System-Object convention as sim3d
// (runtime/toolbox/sim3d/runtime_sim3d.cpp).
//
// All socket state lives here, keyed by the handle object pointer (stable for a
// given handle). Design (see openspec/changes/network-io-tcp-udp/design.md):
//   * Non-blocking sockets + bounded poll() timeout — a stalled or absent peer
//     never blocks the program indefinitely.
//   * tcpserver does a LAZY non-blocking accept on first I/O, so constructing a
//     server does not block waiting for a client (a client connecting via the
//     listen backlog completes its handshake before accept()).
//   * Tier-1 payloads are raw little-endian float64 (one matrix element per 8
//     bytes); this is self-consistent for sim<->sim. Byte-exact third-party
//     interop is a documented Non-Goal of this tier.
// Not thread-safe: one handle per thread.
//===----------------------------------------------------------------------===//

#include "matlab_runtime.h"
#include "runtime_internal.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// matlab_string* return helper (defined in matlab_runtime.cpp).
extern "C" matlab_string *matlab_string_from_literal(const char *src, int64_t n);

namespace {

// Mirror of the runtime string descriptor (char* + length, not zero-terminated).
struct InstrString { char *data; int64_t len; };
std::string toStr(const void *s) {
  if (!s) return {};
  const InstrString *p = reinterpret_cast<const InstrString *>(s);
  if (!p->data || p->len <= 0) return {};
  return std::string(p->data, p->data + static_cast<size_t>(p->len));
}

enum class Role { TcpClient, TcpServer, Udp };

struct Conn {
  Role role = Role::TcpClient;
  int fd = -1;          // active data socket (TCP) or UDP socket
  int listenFd = -1;    // tcpserver only: the listening socket
  int timeoutMs = 1000; // bounded poll() timeout for reads
  // Leftover bytes from a previous read (so a partial float64 / unterminated
  // line is not lost between calls).
  std::vector<uint8_t> inbuf;
  // UDP default destination (set by the most recent write-with-dest); lets
  // writeline/write reuse a peer without re-specifying it.
  sockaddr_in udpDest{};
  bool haveDest = false;
};

std::map<matlab_obj *, Conn> g_conns;

Conn &connOf(matlab_obj *o) { return g_conns[o]; }

void setNonBlocking(int fd) {
#if defined(SOCK_NONBLOCK)
  // Best-effort; also set via fcntl below for portability.
#endif
  int fl = ::fcntl(fd, F_GETFL, 0);
  if (fl >= 0) ::fcntl(fd, F_SETFL, fl | O_NONBLOCK);
}

bool fillAddr(sockaddr_in &a, const std::string &host, int port) {
  std::memset(&a, 0, sizeof(a));
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<uint16_t>(port));
  std::string h = host.empty() ? "127.0.0.1" : host;
  if (h == "localhost") h = "127.0.0.1";
  if (h == "0.0.0.0") { a.sin_addr.s_addr = INADDR_ANY; return true; }
  return ::inet_pton(AF_INET, h.c_str(), &a.sin_addr) == 1;
}

// Block (up to timeoutMs) until fd is readable; return true if data is ready.
bool waitReadable(int fd, int timeoutMs) {
  if (fd < 0) return false;
  pollfd p{fd, POLLIN, 0};
  int r = ::poll(&p, 1, timeoutMs);
  return r > 0 && (p.revents & POLLIN);
}

// tcpserver: accept a pending client (non-blocking) if not yet connected.
void tryAccept(Conn &c) {
  if (c.role != Role::TcpServer || c.fd >= 0 || c.listenFd < 0) return;
  if (!waitReadable(c.listenFd, c.timeoutMs)) return;
  int fd = ::accept(c.listenFd, nullptr, nullptr);
  if (fd >= 0) { setNonBlocking(fd); c.fd = fd; }
}

// Drain up to `want` more bytes into c.inbuf (non-blocking, one poll wait).
void drainInto(Conn &c, size_t want) {
  int fd = c.fd;
  if (fd < 0) return;
  if (!waitReadable(fd, c.timeoutMs)) return;
  uint8_t tmp[4096];
  while (c.inbuf.size() < want) {
    sockaddr_in from{};
    socklen_t fl = sizeof(from);
    ssize_t n;
    if (c.role == Role::Udp)
      n = ::recvfrom(fd, tmp, sizeof(tmp), 0,
                     reinterpret_cast<sockaddr *>(&from), &fl);
    else
      n = ::recv(fd, tmp, sizeof(tmp), 0);
    if (n > 0) {
      c.inbuf.insert(c.inbuf.end(), tmp, tmp + n);
      if (c.role == Role::Udp) { c.udpDest = from; c.haveDest = true; }
      if (static_cast<size_t>(n) < sizeof(tmp)) break; // drained what was ready
    } else {
      break; // would block / closed
    }
  }
}

void sendBytes(Conn &c, const uint8_t *data, size_t n,
               const sockaddr_in *dest) {
  if (c.fd < 0) return;
  if (c.role == Role::Udp && dest)
    ::sendto(c.fd, data, n, 0, reinterpret_cast<const sockaddr *>(dest),
             sizeof(*dest));
  else if (c.role == Role::Udp && c.haveDest)
    ::sendto(c.fd, data, n, 0, reinterpret_cast<const sockaddr *>(&c.udpDest),
             sizeof(c.udpDest));
  else
    ::send(c.fd, data, n, 0);
}

} // namespace

extern "C" {

// ---- constructors ---------------------------------------------------------

// tcpclient(address, port): connect to a TCP server. Non-blocking after the
// connect handshake. Returns the handle (null fd on failure — surfaced as an
// empty read/zero write downstream; the classdef can check status).
void *matlab_tcpclient_new(void *obj_v, void *host_v, double port) {
  if (!obj_v) return obj_v;
  Conn &c = connOf(reinterpret_cast<matlab_obj *>(obj_v));
  c.role = Role::TcpClient;
  sockaddr_in a;
  if (!fillAddr(a, toStr(host_v), static_cast<int>(port))) return obj_v;
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return obj_v;
  if (::connect(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a)) == 0) {
    setNonBlocking(fd);
    c.fd = fd;
  } else {
    ::close(fd);
  }
  return obj_v;
}

// tcpserver(address, port): bind + listen. Accept is lazy (on first I/O).
void *matlab_tcpserver_new(void *obj_v, void *host_v, double port) {
  if (!obj_v) return obj_v;
  Conn &c = connOf(reinterpret_cast<matlab_obj *>(obj_v));
  c.role = Role::TcpServer;
  sockaddr_in a;
  if (!fillAddr(a, toStr(host_v), static_cast<int>(port))) return obj_v;
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return obj_v;
  int yes = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  if (::bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a)) != 0 ||
      ::listen(fd, 1) != 0) {
    ::close(fd);
    return obj_v;
  }
  setNonBlocking(fd);
  c.listenFd = fd;
  return obj_v;
}

// udpport("LocalPort", p) / udpport(): a connectionless datagram socket bound
// to localPort (0 = ephemeral).
void *matlab_udpport_new(void *obj_v, double localPort) {
  if (!obj_v) return obj_v;
  Conn &c = connOf(reinterpret_cast<matlab_obj *>(obj_v));
  c.role = Role::Udp;
  int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) return obj_v;
  int yes = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  sockaddr_in a;
  fillAddr(a, "0.0.0.0", static_cast<int>(localPort));
  if (::bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a)) != 0) {
    ::close(fd);
    return obj_v;
  }
  setNonBlocking(fd);
  c.fd = fd;
  return obj_v;
}

// ---- write / read ---------------------------------------------------------

// write(obj, data): send the matrix elements as little-endian float64. Returns
// the count of elements sent.
double matlab_net_write(void *obj_v, matlab_mat *data) {
  if (!obj_v || !data || !data->data) return 0.0;
  Conn &c = connOf(reinterpret_cast<matlab_obj *>(obj_v));
  tryAccept(c);
  int64_t n = data->rows * data->cols;
  if (n <= 0) return 0.0;
  sendBytes(c, reinterpret_cast<const uint8_t *>(data->data),
            static_cast<size_t>(n) * sizeof(double), nullptr);
  return static_cast<double>(n);
}

// write(udp, data, address, port): send a datagram to an explicit destination.
double matlab_udp_write_to(void *obj_v, matlab_mat *data, void *host_v,
                           double port) {
  if (!obj_v || !data || !data->data) return 0.0;
  Conn &c = connOf(reinterpret_cast<matlab_obj *>(obj_v));
  sockaddr_in dest;
  if (!fillAddr(dest, toStr(host_v), static_cast<int>(port))) return 0.0;
  c.udpDest = dest;
  c.haveDest = true;
  int64_t n = data->rows * data->cols;
  if (n <= 0) return 0.0;
  sendBytes(c, reinterpret_cast<const uint8_t *>(data->data),
            static_cast<size_t>(n) * sizeof(double), &dest);
  return static_cast<double>(n);
}

// read(obj, count): return a 1xcount row of float64 elements (fewer if the
// peer sent less within the timeout). count<=0 returns whatever is buffered.
matlab_mat *matlab_net_read(void *obj_v, double count) {
  if (!obj_v) return mat_alloc(1, 0);
  Conn &c = connOf(reinterpret_cast<matlab_obj *>(obj_v));
  tryAccept(c);
  int64_t want = static_cast<int64_t>(count);
  size_t wantBytes = want > 0 ? static_cast<size_t>(want) * sizeof(double)
                              : c.inbuf.size() + sizeof(double);
  drainInto(c, wantBytes);
  int64_t have = static_cast<int64_t>(c.inbuf.size() / sizeof(double));
  int64_t take = (want > 0 && want < have) ? want : have;
  matlab_mat *m = mat_alloc(1, take);
  if (take > 0) {
    std::memcpy(m->data, c.inbuf.data(), static_cast<size_t>(take) * sizeof(double));
    c.inbuf.erase(c.inbuf.begin(),
                  c.inbuf.begin() + static_cast<long>(take) * static_cast<long>(sizeof(double)));
  }
  return m;
}

// ---- line-oriented text ---------------------------------------------------

// writeline(obj, str): send the string plus a newline terminator.
double matlab_net_writeline(void *obj_v, void *str_v) {
  if (!obj_v) return 0.0;
  Conn &c = connOf(reinterpret_cast<matlab_obj *>(obj_v));
  tryAccept(c);
  std::string s = toStr(str_v);
  s.push_back('\n');
  sendBytes(c, reinterpret_cast<const uint8_t *>(s.data()), s.size(), nullptr);
  return static_cast<double>(s.size());
}

// readline(obj): return the next newline-terminated line (without the
// terminator), or "" if none arrives within the timeout.
matlab_string *matlab_net_readline(void *obj_v) {
  if (!obj_v) return matlab_string_from_literal("", 0);
  Conn &c = connOf(reinterpret_cast<matlab_obj *>(obj_v));
  tryAccept(c);
  // Pull until a newline is buffered or nothing more is ready.
  for (int guard = 0; guard < 64; ++guard) {
    auto it = std::find(c.inbuf.begin(), c.inbuf.end(), '\n');
    if (it != c.inbuf.end()) {
      std::string line(c.inbuf.begin(), it);
      c.inbuf.erase(c.inbuf.begin(), it + 1);
      return matlab_string_from_literal(line.c_str(),
                                        static_cast<int64_t>(line.size()));
    }
    size_t before = c.inbuf.size();
    drainInto(c, before + 1);
    if (c.inbuf.size() == before) break; // no more data
  }
  return matlab_string_from_literal("", 0);
}

// flush(obj): discard buffered input.
double matlab_net_flush(void *obj_v) {
  if (!obj_v) return 0.0;
  Conn &c = connOf(reinterpret_cast<matlab_obj *>(obj_v));
  c.inbuf.clear();
  // Drain anything sitting in the socket buffer too.
  uint8_t tmp[4096];
  while (c.fd >= 0 && waitReadable(c.fd, 0)) {
    ssize_t n = (c.role == Role::Udp)
                    ? ::recvfrom(c.fd, tmp, sizeof(tmp), 0, nullptr, nullptr)
                    : ::recv(c.fd, tmp, sizeof(tmp), 0);
    if (n <= 0) break;
  }
  return 0.0;
}

} // extern "C"
