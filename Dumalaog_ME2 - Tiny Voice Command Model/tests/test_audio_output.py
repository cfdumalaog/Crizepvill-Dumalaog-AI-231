from types import SimpleNamespace

import tinyvcm.audio_output as audio_output
from tinyvcm.audio_output import SystemVolumeOutput


def test_pi_volume_prefers_a_real_pipewire_sink(monkeypatch):
    monkeypatch.setattr(audio_output.sys, "platform", "linux")
    monkeypatch.setattr(audio_output.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls = []

    def fake_run(command):
        calls.append(command)
        if command[1] == "status":
            return SimpleNamespace(returncode=0, stdout="Sinks:\n * 35. HDMI Speakers [vol: 0.50]\nSources:\n")
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(SystemVolumeOutput, "_run", staticmethod(fake_run))
    output = SystemVolumeOutput(enabled=True)

    assert output.mode == "pipewire"
    assert output.set_percent(65) is True
    assert calls[-1] == ["/usr/bin/wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "0.65"]


def test_pi_volume_falls_back_to_alsa_pcm_if_pipewire_is_dummy(monkeypatch):
    monkeypatch.setattr(audio_output.sys, "platform", "linux")
    monkeypatch.setattr(audio_output.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls = []

    def fake_run(command):
        calls.append(command)
        if command[1] == "status":
            return SimpleNamespace(returncode=0, stdout="Sinks:\n * 35. Dummy Output [vol: 1.00]\nSources:\n")
        if command[-1] == "scontrols":
            return SimpleNamespace(returncode=0, stdout="Simple mixer control 'PCM',0\n")
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(SystemVolumeOutput, "_run", staticmethod(fake_run))
    output = SystemVolumeOutput(enabled=True)

    assert output.mode == "alsa-card-0"
    assert output.set_percent(37) is True
    assert calls[-1] == ["/usr/bin/amixer", "-q", "-c", "0", "sset", "PCM", "37%"]


def test_non_pi_host_volume_is_left_to_the_browser(monkeypatch):
    monkeypatch.setattr(audio_output.sys, "platform", "win32")
    output = SystemVolumeOutput(enabled=True)
    assert output.mode == "browser"
    assert output.set_percent(70) is False
