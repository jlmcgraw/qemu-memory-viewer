import qemu_memory_viewer.main as main


class DummyQMP:
    def __init__(self, response: str) -> None:
        self.response = response
        self.commands: list[str] = []

    def hmp(self, command: str) -> str:  # pragma: no cover - trivial
        self.commands.append(command)
        return self.response


def test_extract_hex_bytes_handles_varied_formats() -> None:
    sample = (
        "0x0000000000000000: 0x41 0x07 43 1f\n"
        "0x0000000000000010: 50 aa bb cc dd ee\n"
    )
    assert main._extract_hex_bytes(sample, 8) == [0x41, 0x07, 0x43, 0x1F, 0x50, 0xAA, 0xBB, 0xCC]


def test_parse_register_dump_extracts_segment_base() -> None:
    dump = """
EAX=00000001 EBX=00000002
CS =0000 000f0000 0000ffff 00c09b00
EIP=00007c00
"""

    regs = main.parse_register_dump(dump)

    assert regs["EAX"] == 0x1
    assert regs["EBX"] == 0x2
    assert regs["EIP"] == 0x7C00
    assert regs["CS"] == 0x0
    assert regs["CS_BASE"] == 0x000F0000


def test_compute_pointer_address_uses_segment_base() -> None:
    regs = {"EIP": 0x7C00, "CS_BASE": 0x000F0000}
    spec = main.RegisterPointerSpec(label="IP", offset_keys=("EIP",), segment="CS")

    assert main.compute_pointer_address(regs, spec) == 0x000F7C00


def test_compute_pointer_address_falls_back_to_selector_shift() -> None:
    regs = {"IP": 0x1234, "CS": 0xF000}
    spec = main.RegisterPointerSpec(label="IP", offset_keys=("IP",), segment="CS")

    assert main.compute_pointer_address(regs, spec) == (0xF000 << 4) + 0x1234


def test_qmp_read_b800_pads_to_expected_size() -> None:
    sample = "0x0000000000000000: 0x41 0x07 0x42 0x17"
    qmp = DummyQMP(sample)
    result = main.qmp_read_b800(qmp)

    assert qmp.commands == [f"xp /{main.VGA_TEXT_BYTES}bx {main.VGA_TEXT_BASE}"]
    assert result.shape == (main.VGA_ROWS, main.VGA_COLS, 2)
    assert result.dtype == main.np.uint8
    assert tuple(result[0, 0]) == (0x41, 0x07)
    assert tuple(result[0, 1]) == (0x42, 0x17)
    # The helper should pad missing bytes with zeros so that the reshape always works
    assert tuple(result[0, 2]) == (0, 0)
