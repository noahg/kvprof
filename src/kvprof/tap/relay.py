"""The byte relay: every byte goes through unchanged; the recorder only watches.

A client disconnect closes the upstream socket, so llama-server cancels the task exactly as it would
without the tap. Recording errors never affect the relay.
"""
import itertools
import socket
import threading
import time

from ..util import log, now_hms
from .recorder import Recorder


def pump(src, dst, tap, done):
    try:
        while True:
            data = src.recv(262144)
            if not data:
                break
            t = time.time()
            tap(data, t)
            dst.sendall(data)
    except OSError:
        pass
    finally:
        done()


def handle(client, upstream_addr, conn_id, ctx):
    try:
        up = socket.create_connection(upstream_addr, timeout=10)
    except OSError as e:
        log(f"{now_hms()}  tap: cannot reach upstream {upstream_addr[0]}:{upstream_addr[1]}: {e}")
        client.close()
        return
    up.settimeout(None)
    for s in (client, up):
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
    rec = Recorder(conn_id, ctx)
    state = {"closed_by": None}
    lock = threading.Lock()

    def client_done():  # client stopped sending: close upstream fully so the server cancels
        with lock:
            if state["closed_by"] is None:
                state["closed_by"] = "client"
        rec.on_close(state["closed_by"], time.time())
        try:
            up.close()
        except OSError:
            pass

    def upstream_done():  # upstream finished: let the client see EOF, keep reading until it closes
        with lock:
            if state["closed_by"] is None:
                state["closed_by"] = "upstream"
        rec.on_close(state["closed_by"], time.time())
        try:
            client.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        try:
            client.close()
        except OSError:
            pass

    t1 = threading.Thread(target=pump, args=(client, up, rec.on_request_bytes, client_done), daemon=True)
    t2 = threading.Thread(target=pump, args=(up, client, rec.on_response_bytes, upstream_done), daemon=True)
    t1.start(); t2.start()
    t1.join(); t2.join()
    for s in (client, up):
        try:
            s.close()
        except OSError:
            pass


def serve_relay(listen, upstream, ctx, ready_event=None, stop_event=None, on_listen=None):
    """Accept connections on listen=(host, port) and relay each to upstream=(host, port) until stop_event is set."""
    srv = socket.create_server(listen, reuse_port=False)
    srv.listen(64)
    srv.settimeout(0.25)
    if on_listen:
        on_listen(srv.getsockname())
    if ready_event:
        ready_event.set()
    conn_ids = itertools.count(1)
    try:
        while not (stop_event and stop_event.is_set()):
            try:
                client, _ = srv.accept()
            except socket.timeout:
                continue
            client.settimeout(None)
            threading.Thread(target=handle, args=(client, upstream, next(conn_ids), ctx), daemon=True).start()
    finally:
        srv.close()
