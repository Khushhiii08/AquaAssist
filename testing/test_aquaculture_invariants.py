import sys
sys.stdout.reconfigure(encoding="utf-8")

from generate_rlaif_data import verify_aquaculture_invariants


def test_valid_aquaculture_values():
    assert verify_aquaculture_invariants(
        "The pond water pH was 7.5 and dissolved oxygen was 5.2 mg/L."
    ) is True


def test_low_ph_rejected():
    assert verify_aquaculture_invariants(
        "The pond water pH was 5.5."
    ) is False


def test_high_ph_rejected():
    assert verify_aquaculture_invariants(
        "The pond water pH was 10.2."
    ) is False


def test_negative_do_rejected():
    assert verify_aquaculture_invariants(
        "Dissolved oxygen was -2.0 mg/L."
    ) is False


def test_boundary_ph_values_accepted():
    assert verify_aquaculture_invariants(
        "The pond pH was 6.0."
    ) is True

    assert verify_aquaculture_invariants(
        "The pond pH was 9.5."
    ) is True


def test_empty_text_is_safe():
    assert verify_aquaculture_invariants("") is True


if __name__ == "__main__":
    test_valid_aquaculture_values()
    test_low_ph_rejected()
    test_high_ph_rejected()
    test_negative_do_rejected()
    test_boundary_ph_values_accepted()
    test_empty_text_is_safe()

    print("ALL AQUACULTURE INVARIANT TESTS PASSED")
