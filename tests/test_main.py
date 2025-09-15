from pytest import CaptureFixture
from qemu_memory_viewer.main import main


def test_raise(capsys: CaptureFixture[str]) -> None:
    main()
    assert "Ritchie Blackmore" in capsys.readouterr().out
