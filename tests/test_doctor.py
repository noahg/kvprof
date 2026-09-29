import os

from conftest import FIXTURES
from kvprof.cli import main
from kvprof.config import Settings
from kvprof.doctor import run_doctor


def test_doctor_ok(fake_upstream, capsys):
    rc = run_doctor(Settings(upstream=fake_upstream.url, server_log=os.path.join(FIXTURES, "server.log")))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "b10909-test (tested)" in out and "section profile chatml_qwen" in out and "pieces concatenate" in out
    assert "count_tokens" in out and "launch 4" in out and "result: all OK" in out


def test_doctor_generic_profile(fake_upstream_llama3, capsys):
    assert run_doctor(Settings(upstream=fake_upstream_llama3.url)) == 0
    out = capsys.readouterr().out
    assert "section profile generic" in out and "no --server-log" in out


def test_doctor_log_drift(fake_upstream, tmp_path, capsys):
    p = tmp_path / "server.log"
    with open(os.path.join(FIXTURES, "server.log")) as f:
        p.write_text(f.read() + "1.30.000.000 I slot launch_slot_: id  0 | task 99 | began work on task (format v2)\n")
    assert run_doctor(Settings(upstream=fake_upstream.url, server_log=str(p))) == 0
    out = capsys.readouterr().out
    assert "1 slot line(s) matched no pattern" in out and "began work" in out and "OK with warnings" in out


def test_doctor_wrong_log(fake_upstream, tmp_path, capsys):
    p = tmp_path / "other.log"
    p.write_text("hello\nworld\n")
    assert run_doctor(Settings(upstream=fake_upstream.url, server_log=str(p))) == 1
    assert "no slot lines recognized" in capsys.readouterr().out


def test_doctor_unreachable(capsys):
    assert run_doctor(Settings(upstream="http://127.0.0.1:9")) == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_doctor_exit_code(fake_upstream):
    import pytest
    with pytest.raises(SystemExit) as e:
        main(["doctor", "--upstream", fake_upstream.url])
    assert e.value.code == 0
