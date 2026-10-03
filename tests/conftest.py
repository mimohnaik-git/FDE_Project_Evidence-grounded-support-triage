import socket

import pytest


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guarded_connect(self, address):
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return original_connect(self, address)
        if self.family == socket.AF_UNIX:
            return original_connect(self, address)
        raise AssertionError("External network forbidden in offline verification")

    def guarded_connect_ex(self, address):
        if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
            return original_connect_ex(self, address)
        raise AssertionError("External network forbidden in offline verification")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
